# -*- coding: UTF-8 -*-
"""
BiGeaR_LightGCN: 带消融开关的 BiGeaR 算法实现

本文件是从 src/models/general/BiGeaR_LightGCN.py 复制并修改的版本，
添加了 use_distillation 参数用于消融实验。

修改说明：
    - 添加 --use_distillation 参数，默认为 True
    - 当 use_distillation=False 时，Stage 2 不使用 ID Loss

Reference:
    "Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement 
     for Top-K Recommendation"
    Yankai Chen et al., KDD 2022.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, Optional, Tuple, List

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from models.BaseModel import GeneralModel


# =============================================================================
# 核心组件 1: Normal_Ddelta 二值化函数
# =============================================================================

class NormalDdelta(torch.autograd.Function):
    """
    二值化函数，使用正态分布近似 Dirac delta 函数进行梯度估计。
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, norm_a: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.norm_a = norm_a
        
        binary_encoding = torch.sign(x)
        n = x.shape[-1]
        scale = x.abs().mean(dim=-1, keepdim=True)
        output = binary_encoding * scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        x, = ctx.saved_tensors
        norm_a = ctx.norm_a
        
        grad_scale = norm_a / math.sqrt(math.pi)
        grad_input = grad_scale * torch.exp(-(norm_a * x) ** 2)
        grad_input = torch.clamp(grad_input, min=1e-8, max=1.0)
        
        return grad_input * grad_output, None


class QuantLayer(nn.Module):
    """量化层：封装 NormalDdelta 二值化操作。"""
    
    def __init__(self, norm_a: float = 1.0):
        super(QuantLayer, self).__init__()
        self.norm_a = norm_a
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return NormalDdelta.apply(x, self.norm_a)
    
    def set_norm_a(self, norm_a: float):
        self.norm_a = norm_a


# =============================================================================
# 核心组件 2: 图构建工具
# =============================================================================

def build_sparse_graph(
    n_users: int, 
    n_items: int, 
    train_clicked_set: Dict[int, set],
    device: torch.device
) -> torch.sparse.FloatTensor:
    """构建用户-物品二部图的归一化邻接矩阵。"""
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for user, items in train_clicked_set.items():
        for item in items:
            R[user, item] = 1.0
    R = R.tocsr()
    
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    
    rowsum = np.array(adj_mat.sum(axis=1)).flatten() + 1e-10
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
    norm_adj = norm_adj.tocoo()
    
    indices = torch.LongTensor([norm_adj.row, norm_adj.col])
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    return sparse_tensor.to(device)


# =============================================================================
# 核心组件 3: BiGeaR_LightGCN 主模型（带消融开关）
# =============================================================================

class BiGeaR_LightGCN(GeneralModel):
    """
    BiGeaR 算法的 LightGCN 实现，添加消融开关。
    
    新增参数：
        --use_distillation: 是否使用知识蒸馏（ID Loss），默认 True
                           设为 False 时进行消融实验
    """
    
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'stage', 'lambda_id', 'top_r', 'use_distillation']
    
    @staticmethod
    def parse_model_args(parser):
        """解析模型特有的命令行参数"""
        parser.add_argument('--emb_size', type=int, default=256,
                            help='嵌入向量维度')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='图卷积层数')
        parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                            help='训练阶段：1=Teacher预训练, 2=Student蒸馏')
        parser.add_argument('--norm_a', type=float, default=1.0,
                            help='Dirac delta 近似参数')
        parser.add_argument('--lambda_id', type=float, default=1.0,
                            help='ID损失权重（仅阶段2有效）')
        parser.add_argument('--top_r', type=int, default=100,
                            help='ID损失中考虑的 Top-R 物品数')
        parser.add_argument('--lambda_decay', type=float, default=0.1,
                            help='ID损失中排名位置的指数衰减系数')
        parser.add_argument('--teacher_model_path', type=str, default='',
                            help='预训练 Teacher 模型路径（阶段2必需）')
        # 消融实验开关
        parser.add_argument('--use_distillation', type=int, default=1,
                            help='是否使用知识蒸馏 (1=使用, 0=不使用)')
        
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.stage = args.stage
        self.norm_a = args.norm_a
        
        self.lambda_id = args.lambda_id
        self.top_r = args.top_r
        self.lambda_decay = args.lambda_decay
        self.teacher_model_path = args.teacher_model_path
        
        # 消融开关：是否使用蒸馏
        self.use_distillation = bool(args.use_distillation)
        
        self.l2 = args.l2
        
        self._define_params()
        self.apply(self.init_weights)
        self._build_graph(corpus)
        self._compute_layer_weights()
        self._compute_ranking_weights()
        
        if self.stage == 2:
            self._setup_teacher()
    
    def _define_params(self):
        """定义模型参数"""
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size)
        
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)
        
        self.quant_layer = QuantLayer(norm_a=self.norm_a)
        self.register_buffer('teacher_top_r_indices', None)
    
    def _build_graph(self, corpus):
        """构建归一化邻接矩阵"""
        self.sparse_norm_adj = build_sparse_graph(
            n_users=self.user_num,
            n_items=self.item_num,
            train_clicked_set=corpus.train_clicked_set,
            device=self.device
        )
    
    def _compute_layer_weights(self):
        """计算层级聚合权重"""
        L = self.n_layers
        weights = torch.tensor(
            [(l + 1) / (L + 1) for l in range(L + 1)],
            dtype=torch.float32,
            device=self.device
        )
        self.register_buffer('layer_weights', weights)
    
    def _compute_ranking_weights(self):
        """预计算 ID 损失中的排名位置权重"""
        rank_weights = torch.exp(
            -torch.arange(1, self.top_r + 1, dtype=torch.float32, device=self.device) 
            * self.lambda_decay
        )
        self.register_buffer('rank_weights', rank_weights)
    
    def _setup_teacher(self):
        """设置 Teacher 模型（阶段2）"""
        if not self.teacher_model_path or not os.path.exists(self.teacher_model_path):
            raise ValueError(
                f"阶段2需要有效的 teacher_model_path，当前值: '{self.teacher_model_path}'"
            )
        
        teacher_state = torch.load(self.teacher_model_path, map_location=self.device)
        
        self.teacher_user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.teacher_item_emb = nn.Embedding(self.item_num, self.emb_size)
        
        if 'user_embedding.weight' in teacher_state:
            user_weight = teacher_state['user_embedding.weight']
            item_weight = teacher_state['item_embedding.weight']
        elif 'encoder.embedding_dict.user_emb' in teacher_state:
            user_weight = teacher_state['encoder.embedding_dict.user_emb']
            item_weight = teacher_state['encoder.embedding_dict.item_emb']
        else:
            raise ValueError(f"无法识别的 Teacher 权重格式")
        
        self.teacher_user_emb.weight.data.copy_(user_weight)
        self.teacher_item_emb.weight.data.copy_(item_weight)
        
        self.teacher_user_emb.weight.requires_grad = False
        self.teacher_item_emb.weight.requires_grad = False
        
        self.user_embedding.weight.data.copy_(user_weight)
        self.item_embedding.weight.data.copy_(item_weight)
        
        # 只有启用蒸馏时才计算 Teacher 排名
        if self.use_distillation:
            self._compute_teacher_rankings()
    
    @torch.no_grad()
    def _compute_teacher_rankings(self):
        """计算 Teacher 在每一层的 Top-R 物品排名"""
        user_emb = self.teacher_user_emb.weight.to(self.device)
        item_emb = self.teacher_item_emb.weight.to(self.device)
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        sparse_adj = self.sparse_norm_adj.to(self.device)
        
        layer_rankings = []
        current_emb = all_emb
        
        for layer in range(self.n_layers + 1):
            if layer > 0:
                current_emb = torch.sparse.mm(sparse_adj, current_emb)
            
            weighted_emb = current_emb * self.layer_weights[layer]
            
            user_layer_emb = weighted_emb[:self.user_num]
            item_layer_emb = weighted_emb[self.user_num:]
            
            top_r_indices = self._batch_topk(
                user_layer_emb, item_layer_emb, k=self.top_r
            )
            layer_rankings.append(top_r_indices)
        
        self.teacher_top_r_indices = torch.stack(layer_rankings, dim=2)
    
    def _batch_topk(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor, 
        k: int,
        batch_size: int = 1024
    ) -> torch.Tensor:
        """分批计算 Top-K 物品索引"""
        n_users = user_emb.size(0)
        all_indices = []
        
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_user_emb = user_emb[start:end]
            scores = torch.matmul(batch_user_emb, item_emb.t())
            _, indices = torch.topk(scores, k=k, dim=1, largest=True)
            all_indices.append(indices)
        
        return torch.cat(all_indices, dim=0)
    
    def _propagate_and_aggregate_teacher(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Teacher 的图传播和聚合（全精度）"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        layer_embs = []
        
        for layer in range(self.n_layers + 1):
            if layer == 0:
                current_emb = all_emb
            else:
                current_emb = torch.sparse.mm(self.sparse_norm_adj, current_emb)
            
            weighted_emb = current_emb * self.layer_weights[layer]
            layer_embs.append(weighted_emb)
        
        concat_emb = torch.cat(layer_embs, dim=1)
        
        user_agg_emb = concat_emb[:self.user_num]
        item_agg_emb = concat_emb[self.user_num:]
        
        return user_agg_emb, item_agg_emb
    
    def _propagate_and_quantize_student(self):
        """Student 的图传播、二值化和聚合"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        user_bin_layers = []
        item_bin_layers = []
        user_bin_weighted_layers = []
        item_bin_weighted_layers = []
        bin_weighted_embs = []
        
        current_emb = all_emb
        
        for layer in range(self.n_layers + 1):
            if layer > 0:
                current_emb = torch.sparse.mm(self.sparse_norm_adj, current_emb)
            
            bin_emb = self.quant_layer(current_emb)
            weighted_bin_emb = bin_emb * self.layer_weights[layer]
            bin_weighted_embs.append(weighted_bin_emb)
            
            user_bin = bin_emb[:self.user_num]
            item_bin = bin_emb[self.user_num:]
            user_bin_layers.append(user_bin)
            item_bin_layers.append(item_bin)
            
            user_bin_weighted = weighted_bin_emb[:self.user_num]
            item_bin_weighted = weighted_bin_emb[self.user_num:]
            user_bin_weighted_layers.append(user_bin_weighted)
            item_bin_weighted_layers.append(item_bin_weighted)
        
        concat_bin_emb = torch.cat(bin_weighted_embs, dim=1)
        
        user_bin_agg = concat_bin_emb[:self.user_num]
        item_bin_agg = concat_bin_emb[self.user_num:]
        
        return (user_bin_agg, item_bin_agg, user_bin_layers, item_bin_layers,
                user_bin_weighted_layers, item_bin_weighted_layers)
    
    def _bpr_loss(
        self, 
        user_emb: torch.Tensor, 
        pos_emb: torch.Tensor, 
        neg_emb: torch.Tensor
    ) -> torch.Tensor:
        """计算 BPR 排序损失"""
        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        
        if neg_emb.dim() == 3:
            neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=-1)
            neg_softmax = F.softmax(neg_scores, dim=1)
            neg_scores = (neg_scores * neg_softmax).sum(dim=1)
        else:
            neg_scores = (user_emb * neg_emb).sum(dim=-1)
        
        loss = F.softplus(neg_scores - pos_scores).mean()
        return loss
    
    def _id_loss(
        self,
        user_indices: torch.Tensor,
        user_bin_weighted_layers: List[torch.Tensor],
        item_bin_weighted_layers: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算 Inference Distillation (ID) 损失"""
        batch_size = user_indices.size(0)
        device = user_indices.device
        
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1)
        
        layer_losses = []
        
        for layer in range(self.n_layers + 1):
            teacher_topk = self.teacher_top_r_indices[user_indices, :, layer].long()
            user_bin_w = user_bin_weighted_layers[layer][user_indices]
            item_bin_w = item_bin_weighted_layers[layer]
            
            all_scores = torch.matmul(user_bin_w, item_bin_w.t())
            topk_scores = all_scores[batch_idx, teacher_topk]
            
            layer_loss = F.softplus(-topk_scores) * self.rank_weights.unsqueeze(0)
            layer_loss = layer_loss.mean(dim=-1).mean()
            
            layer_losses.append(layer_loss)
        
        return torch.stack(layer_losses).mean()
    
    def _reg_loss(
        self, 
        user_indices: torch.Tensor, 
        pos_indices: torch.Tensor, 
        neg_indices: torch.Tensor
    ) -> torch.Tensor:
        """计算 L2 正则化损失"""
        user_emb = self.user_embedding(user_indices)
        pos_emb = self.item_embedding(pos_indices)
        
        if neg_indices.dim() == 1:
            neg_emb = self.item_embedding(neg_indices)
        else:
            neg_emb = self.item_embedding(neg_indices.view(-1)).view(
                neg_indices.size(0), neg_indices.size(1), -1
            )
        
        reg_loss = (
            user_emb.norm(2).pow(2) + 
            pos_emb.norm(2).pow(2) + 
            neg_emb.norm(2).pow(2)
        ) / (2 * user_indices.size(0))
        
        return reg_loss
    
    def forward(self, feed_dict: Dict) -> Dict:
        """前向传播"""
        self.check_list = []
        
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        
        if self.stage == 1:
            user_agg_emb, item_agg_emb = self._propagate_and_aggregate_teacher()
            
            user_emb = user_agg_emb[users]
            item_emb = item_agg_emb[items.view(-1)].view(
                items.size(0), items.size(1), -1
            )
            
            prediction = (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)
            
            return {
                'prediction': prediction,
                'user_indices': users,
                'item_indices': items,
            }
        
        else:
            (user_bin_agg, item_bin_agg, 
             user_bin_layers, item_bin_layers,
             user_bin_weighted_layers, item_bin_weighted_layers) = self._propagate_and_quantize_student()
            
            user_emb = user_bin_agg[users]
            item_emb = item_bin_agg[items.view(-1)].view(
                items.size(0), items.size(1), -1
            )
            
            prediction = (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)
            
            out_dict = {
                'prediction': prediction,
                'user_indices': users,
                'item_indices': items,
                'user_bin_layers': user_bin_layers,
                'item_bin_layers': item_bin_layers,
                'user_bin_weighted_layers': user_bin_weighted_layers,
                'item_bin_weighted_layers': item_bin_weighted_layers,
            }
            
            return out_dict
    
    def loss(self, out_dict: Dict) -> torch.Tensor:
        """
        计算总损失。
        
        Stage 1: BPR损失 + L2正则化
        Stage 2: BPR损失 + ID损失（如果 use_distillation=True）+ L2正则化
        """
        prediction = out_dict['prediction']
        
        pos_score = prediction[:, 0]
        neg_score = prediction[:, 1:]
        
        if neg_score.size(1) == 1:
            neg_score = neg_score.squeeze(1)
            bpr_loss = F.softplus(neg_score - pos_score).mean()
        else:
            neg_softmax = F.softmax(neg_score, dim=1)
            weighted_neg = (neg_score * neg_softmax).sum(dim=1)
            bpr_loss = F.softplus(weighted_neg - pos_score).mean()
        
        if self.stage == 1:
            user_indices = out_dict['user_indices']
            item_indices = out_dict['item_indices']
            pos_indices = item_indices[:, 0]
            neg_indices = item_indices[:, 1:]
            reg_loss = self._reg_loss(user_indices, pos_indices, neg_indices)
            
            return bpr_loss + self.l2 * reg_loss
        
        else:
            user_indices = out_dict['user_indices']
            item_indices = out_dict['item_indices']
            user_bin_weighted_layers = out_dict['user_bin_weighted_layers']
            item_bin_weighted_layers = out_dict['item_bin_weighted_layers']
            
            # 消融开关：只有 use_distillation=True 时才计算 ID 损失
            if self.use_distillation:
                id_loss = self._id_loss(user_indices, user_bin_weighted_layers, item_bin_weighted_layers)
            else:
                id_loss = torch.tensor(0.0, device=prediction.device)
            
            pos_indices = item_indices[:, 0]
            neg_indices = item_indices[:, 1:]
            reg_loss = self._reg_loss(user_indices, pos_indices, neg_indices)
            
            total_loss = bpr_loss + self.lambda_id * id_loss + self.l2 * reg_loss
            
            return total_loss
    
    def save_model(self, model_path: str = None):
        """保存模型"""
        if model_path is None:
            model_path = self.model_path
        
        from utils import utils
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)
    
    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取所有用户和物品的聚合嵌入（用于评估）"""
        with torch.no_grad():
            if self.stage == 1:
                return self._propagate_and_aggregate_teacher()
            else:
                user_bin_agg, item_bin_agg, _, _, _, _ = self._propagate_and_quantize_student()
                return user_bin_agg, item_bin_agg
    
    def inference(self, feed_dict: Dict) -> Dict:
        """推理方法，用于评估阶段"""
        self.check_list = []
        
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        
        with torch.no_grad():
            user_emb, item_emb = self.get_all_embeddings()
            
            user_e = user_emb[users]
            item_e = item_emb[items.view(-1)].view(
                items.size(0), items.size(1), -1
            )
            
            prediction = (user_e.unsqueeze(1) * item_e).sum(dim=-1)
        
        return {'prediction': prediction}

