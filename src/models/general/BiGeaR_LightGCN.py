# -*- coding: UTF-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Dict, Optional, Tuple, List

from models.BaseModel import GeneralModel


# =============================================================================
# 核心组件 1: Normal_Ddelta 二值化函数
# =============================================================================

class NormalDdelta(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, norm_a: float) -> torch.Tensor:
        """
        Args:
            x: 输入张量，shape 任意
            norm_a: 正态分布近似参数，控制梯度的集中程度
            
        Returns:
            二值化后的张量，保持输入 shape
        """
        ctx.save_for_backward(x)
        ctx.norm_a = norm_a
        
        # 计算符号：{-1, 0, +1}
        binary_encoding = torch.sign(x)
        
        # 计算缩放因子：每个向量的 L1 范数均值
        # x.shape 可能是 [n, d] 或 [n, L+1, d]
        n = x.shape[-1]  # 嵌入维度
        
        # 对最后一个维度计算 L1 范数均值
        scale = x.abs().mean(dim=-1, keepdim=True)
        
        # 输出 = 符号 × 缩放因子
        output = binary_encoding * scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        使用正态分布近似 Dirac delta 函数计算梯度。
        
        Dirac delta 近似：δ(x) ≈ (a / √π) * exp(-(ax)²)
        其中 a = norm_a
        """
        x, = ctx.saved_tensors
        norm_a = ctx.norm_a
        
        # 正态分布近似的梯度
        grad_scale = norm_a / math.sqrt(math.pi)
        grad_input = grad_scale * torch.exp(-(norm_a * x) ** 2)
        
        # 梯度裁剪，防止数值问题
        grad_input = torch.clamp(grad_input, min=1e-8, max=1.0)
        
        return grad_input * grad_output, None


class QuantLayer(nn.Module):
    """
    量化层：封装 NormalDdelta 二值化操作。
    
    这个模块可以方便地在模型中使用，支持设置 norm_a 参数。
    """
    
    def __init__(self, norm_a: float = 1.0):

        super(QuantLayer, self).__init__()
        self.norm_a = norm_a
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行二值化。
        """
        return NormalDdelta.apply(x, self.norm_a)
    
    def set_norm_a(self, norm_a: float):
        """动态调整 norm_a 参数"""
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

    # 构建用户-物品交互矩阵 R
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)
    for user, items in train_clicked_set.items():
        for item in items:
            R[user, item] = 1.0
    R = R.tocsr()
    
    # 构建完整邻接矩阵
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()
    
    # 对称归一化：D^{-1/2} * A * D^{-1/2}
    rowsum = np.array(adj_mat.sum(axis=1)).flatten() + 1e-10
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
    norm_adj = norm_adj.tocoo()
    
    # 转换为 PyTorch 稀疏张量
    indices = torch.LongTensor([norm_adj.row, norm_adj.col])
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    
    sparse_tensor = torch.sparse.FloatTensor(indices, values, shape)
    return sparse_tensor.to(device)


# =============================================================================
# 核心组件 3: BiGeaR_LightGCN 主模型
# =============================================================================

class BiGeaR_LightGCN(GeneralModel):
    
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'stage', 'lambda_id', 'top_r']
    
    @staticmethod
    def parse_model_args(parser):
        """解析模型特有的命令行参数"""
        # 基础参数
        parser.add_argument('--emb_size', type=int, default=256,
                            help='嵌入向量维度')
        parser.add_argument('--n_layers', type=int, default=2,
                            help='图卷积层数')
        
        # 训练阶段控制
        parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                            help='训练阶段：1=Teacher预训练, 2=Student蒸馏')
        
        # 二值化参数
        parser.add_argument('--norm_a', type=float, default=1.0,
                            help='Dirac delta 近似参数，控制梯度集中程度')
        
        # Inference Distillation 参数
        parser.add_argument('--lambda_id', type=float, default=1.0,
                            help='ID损失权重（仅阶段2有效），原始BiGeaR默认1.0')
        parser.add_argument('--top_r', type=int, default=100,
                            help='ID损失中考虑的 Top-R 物品数（仅阶段2有效）')
        parser.add_argument('--lambda_decay', type=float, default=0.1,
                            help='ID损失中排名位置的指数衰减系数')
        
        # Teacher 模型路径（阶段2必需）
        parser.add_argument('--teacher_model_path', type=str, default='',
                            help='预训练 Teacher 模型路径（阶段2必需）')
        
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        """
        初始化 BiGeaR_LightGCN 模型。
        """
        super().__init__(args, corpus)
        
        # 基础配置
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.stage = args.stage
        self.norm_a = args.norm_a
        
        # ID损失配置
        self.lambda_id = args.lambda_id
        self.top_r = args.top_r
        self.lambda_decay = args.lambda_decay
        self.teacher_model_path = args.teacher_model_path
        
        # L2 正则化系数（从 GeneralModel 继承，通过 --l2 参数设置）
        self.l2 = args.l2
        
        # 初始化模型参数
        self._define_params()
        
        # 先初始化权重，确保 Stage1 从合理初始化开始
        # 注意：Stage2 会在加载 Teacher 后不再重新初始化，避免覆盖 Teacher 权重
        self.apply(self.init_weights)
        
        # 构建图结构
        self._build_graph(corpus)
        
        # 计算层级聚合权重：lambda[l] = (l+1) / (L+1)
        self._compute_layer_weights()
        
        # 计算 ID 损失的排名权重（指数衰减）
        self._compute_ranking_weights()
        
        # 如果是阶段2，加载 Teacher 并计算 Top-R 排名
        if self.stage == 2:
            self._setup_teacher()
    
    def _define_params(self):
        # 用户和物品嵌入
        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size)
        
        # 使用原始 BiGeaR 的初始化方式：std=0.1（而非 ReChorus 默认的 0.01）
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)
        
        # 量化层
        self.quant_layer = QuantLayer(norm_a=self.norm_a)
        
        # 用于存储 Teacher 的 Top-R 排名（阶段2）
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
        """
        计算层级聚合权重。
        """
        L = self.n_layers
        weights = torch.tensor(
            [(l + 1) / (L + 1) for l in range(L + 1)],
            dtype=torch.float32,
            device=self.device
        )
        self.register_buffer('layer_weights', weights)
    
    def _compute_ranking_weights(self):
        """
        预计算 ID 损失中的排名位置权重。
        """
        rank_weights = torch.exp(
            -torch.arange(1, self.top_r + 1, dtype=torch.float32, device=self.device) 
            * self.lambda_decay
        )
        self.register_buffer('rank_weights', rank_weights)
    
    def _setup_teacher(self):

        if not self.teacher_model_path or not os.path.exists(self.teacher_model_path):
            raise ValueError(
                f"阶段2需要有效的 teacher_model_path，当前值: '{self.teacher_model_path}'\n"
                f"请先运行阶段1训练：python main.py --model_name BiGeaR_LightGCN --stage 1 ..."
            )
        
        # 加载 Teacher 权重
        teacher_state = torch.load(self.teacher_model_path, map_location=self.device)
        
        # 创建 Teacher 嵌入（冻结）
        self.teacher_user_emb = nn.Embedding(self.user_num, self.emb_size)
        self.teacher_item_emb = nn.Embedding(self.item_num, self.emb_size)
        
        # 加载权重（支持两种格式）
        if 'user_embedding.weight' in teacher_state:
            # BiGeaR_LightGCN 阶段1格式
            user_weight = teacher_state['user_embedding.weight']
            item_weight = teacher_state['item_embedding.weight']
        elif 'encoder.embedding_dict.user_emb' in teacher_state:
            # LightGCN 格式
            user_weight = teacher_state['encoder.embedding_dict.user_emb']
            item_weight = teacher_state['encoder.embedding_dict.item_emb']
        else:
            available_keys = list(teacher_state.keys())[:10]
            raise ValueError(
                f"无法识别的 Teacher 权重格式。可用的 key: {available_keys}...\n"
                f"期望: 'user_embedding.weight' 或 'encoder.embedding_dict.user_emb'"
            )
        
        # 检查维度匹配
        if user_weight.size(0) != self.user_num or user_weight.size(1) != self.emb_size:
            raise ValueError(
                f"Teacher 嵌入维度不匹配！期望: ({self.user_num}, {self.emb_size})，"
                f"实际: {tuple(user_weight.shape)}"
            )
        if item_weight.size(0) != self.item_num or item_weight.size(1) != self.emb_size:
            raise ValueError(
                f"Teacher 嵌入维度不匹配！期望: ({self.item_num}, {self.emb_size})，"
                f"实际: {tuple(item_weight.shape)}"
            )
        
        # 加载权重到 Teacher
        self.teacher_user_emb.weight.data.copy_(user_weight)
        self.teacher_item_emb.weight.data.copy_(item_weight)
        
        # 冻结 Teacher 参数
        self.teacher_user_emb.weight.requires_grad = False
        self.teacher_item_emb.weight.requires_grad = False
        
        # 将 Student 嵌入初始化为 Teacher 的值
        self.user_embedding.weight.data.copy_(user_weight)
        self.item_embedding.weight.data.copy_(item_weight)
        
        # 计算 Teacher 的逐层 Top-R 排名
        self._compute_teacher_rankings()
    
    @torch.no_grad()
    def _compute_teacher_rankings(self):
        """
        计算 Teacher 在每一层的 Top-R 物品排名。
        
        关键：使用**加权后**的嵌入计算排名，与原始 BiGeaR 保持一致！
        原始代码参考：BiGeaR model.py summary() 方法第 110-125 行
            con_embed_i = con_original_embed * self.lambdas[layer]
            score_i = torch.matmul(con_origin_users_embed, con_origin_items_embed.t())
        
        这个排名在训练过程中保持不变，用于 ID 损失的计算。
        
        返回的形状：[n_users, top_r, n_layers+1]
        """
        # 获取 Teacher 的逐层嵌入，确保在正确的设备上
        user_emb = self.teacher_user_emb.weight.to(self.device)
        item_emb = self.teacher_item_emb.weight.to(self.device)
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # 确保图矩阵也在正确的设备上
        sparse_adj = self.sparse_norm_adj.to(self.device)
        
        layer_rankings = []
        current_emb = all_emb
        
        for layer in range(self.n_layers + 1):
            if layer > 0:
                current_emb = torch.sparse.mm(sparse_adj, current_emb)
            
            # 关键：使用加权后的嵌入计算排名！
            weighted_emb = current_emb * self.layer_weights[layer]
            
            # 分离用户和物品嵌入
            user_layer_emb = weighted_emb[:self.user_num]
            item_layer_emb = weighted_emb[self.user_num:]
            
            # 计算得分矩阵并获取 Top-R
            # 为了节省内存，分批处理
            top_r_indices = self._batch_topk(
                user_layer_emb, item_layer_emb, k=self.top_r
            )
            layer_rankings.append(top_r_indices)
        
        # 堆叠为 [n_users, top_r, n_layers+1]
        self.teacher_top_r_indices = torch.stack(layer_rankings, dim=2)
    
    def _batch_topk(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor, 
        k: int,
        batch_size: int = 1024
    ) -> torch.Tensor:
        """
        分批计算 Top-K 物品索引，避免 OOM。
        """
        n_users = user_emb.size(0)
        all_indices = []
        
        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_user_emb = user_emb[start:end]
            
            # 计算得分
            scores = torch.matmul(batch_user_emb, item_emb.t())
            
            # 获取 Top-K
            _, indices = torch.topk(scores, k=k, dim=1, largest=True)
            all_indices.append(indices)
        
        return torch.cat(all_indices, dim=0)
    
    # =========================================================================
    # 图传播方法
    # =========================================================================
    
    def _propagate_and_aggregate_teacher(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Teacher 的图传播和聚合（全精度）。
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # 存储每层的嵌入（加权后）
        layer_embs = []
        
        for layer in range(self.n_layers + 1):
            if layer == 0:
                current_emb = all_emb
            else:
                current_emb = torch.sparse.mm(self.sparse_norm_adj, current_emb)
            
            # 加权
            weighted_emb = current_emb * self.layer_weights[layer]
            layer_embs.append(weighted_emb)
        
        # 拼接所有层的嵌入
        concat_emb = torch.cat(layer_embs, dim=1)
        
        user_agg_emb = concat_emb[:self.user_num]
        item_agg_emb = concat_emb[self.user_num:]
        
        return user_agg_emb, item_agg_emb
    
    def _propagate_and_quantize_student(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], 
               List[torch.Tensor], List[torch.Tensor]]:
        """
        Student 的图传播、二值化和聚合。
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # 存储原始嵌入（用于正则化）
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # 存储每层的二值化嵌入
        user_bin_layers = []
        item_bin_layers = []
        user_bin_weighted_layers = []
        item_bin_weighted_layers = []
        bin_weighted_embs = []
        
        current_emb = all_emb
        
        for layer in range(self.n_layers + 1):
            if layer > 0:
                current_emb = torch.sparse.mm(self.sparse_norm_adj, current_emb)
            
            # 二值化当前层的嵌入
            bin_emb = self.quant_layer(current_emb)
            
            # 加权
            weighted_bin_emb = bin_emb * self.layer_weights[layer]
            bin_weighted_embs.append(weighted_bin_emb)
            
            # 分离用户和物品（未加权）
            user_bin = bin_emb[:self.user_num]
            item_bin = bin_emb[self.user_num:]
            user_bin_layers.append(user_bin)
            item_bin_layers.append(item_bin)
            
            # 分离用户和物品（加权后，用于ID损失）
            user_bin_weighted = weighted_bin_emb[:self.user_num]
            item_bin_weighted = weighted_bin_emb[self.user_num:]
            user_bin_weighted_layers.append(user_bin_weighted)
            item_bin_weighted_layers.append(item_bin_weighted)
        
        # 拼接所有层
        concat_bin_emb = torch.cat(bin_weighted_embs, dim=1)
        
        user_bin_agg = concat_bin_emb[:self.user_num]
        item_bin_agg = concat_bin_emb[self.user_num:]
        
        return (user_bin_agg, item_bin_agg, user_bin_layers, item_bin_layers,
                user_bin_weighted_layers, item_bin_weighted_layers)
    
    # =========================================================================
    # 损失函数
    # =========================================================================
    
    def _bpr_loss(
        self, 
        user_emb: torch.Tensor, 
        pos_emb: torch.Tensor, 
        neg_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 BPR 排序损失。
        """
        # 正样本得分
        pos_scores = (user_emb * pos_emb).sum(dim=-1)
        
        # 负样本得分
        if neg_emb.dim() == 3:
            # 多个负样本：[B, n_neg, d]
            neg_scores = (user_emb.unsqueeze(1) * neg_emb).sum(dim=-1)  # [B, n_neg]
            # 使用 softmax 加权的方式处理多个负样本
            neg_softmax = F.softmax(neg_scores, dim=1)
            neg_scores = (neg_scores * neg_softmax).sum(dim=1)
        else:
            neg_scores = (user_emb * neg_emb).sum(dim=-1)
        
        # BPR 损失
        loss = F.softplus(neg_scores - pos_scores).mean()
        
        return loss
    
    def _id_loss(
        self,
        user_indices: torch.Tensor,
        user_bin_weighted_layers: List[torch.Tensor],
        item_bin_weighted_layers: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算 Inference Distillation (ID) 损失。        
        """
        batch_size = user_indices.size(0)
        device = user_indices.device
        
        # 批次索引
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1)
        
        layer_losses = []
        
        for layer in range(self.n_layers + 1):
            # 获取 Teacher 的 Top-R 索引：[B, top_r]
            teacher_topk = self.teacher_top_r_indices[user_indices, :, layer].long()
            
            # 获取 Student 的**加权**用户嵌入：[B, d]
            user_bin_w = user_bin_weighted_layers[layer][user_indices]
            
            # 获取 Student 的**加权**物品嵌入：[n_items, d]
            item_bin_w = item_bin_weighted_layers[layer]
            
            # 计算 Student 对所有物品的得分（使用加权嵌入）
            # 得分 = w² × (bin_u · bin_v)，与原始 BiGeaR 一致
            all_scores = torch.matmul(user_bin_w, item_bin_w.t())  # [B, n_items]
            
            # 获取 Teacher Top-R 物品的得分
            topk_scores = all_scores[batch_idx, teacher_topk]  # [B, top_r]
            
            # 计算损失：鼓励 Student 给 Teacher 的 Top-R 物品高分
            # 使用 softplus(-score) 实现数值稳定的 -log(sigmoid(score))
            # loss = mean(mean(softplus(-score) * weight))
            layer_loss = F.softplus(-topk_scores) * self.rank_weights.unsqueeze(0)
            layer_loss = layer_loss.mean(dim=-1).mean()  # 先对 top_r 平均，再对 batch 平均
            
            layer_losses.append(layer_loss)
        
        # 返回所有层损失的平均值
        return torch.stack(layer_losses).mean()
    
    def _reg_loss(
        self, 
        user_indices: torch.Tensor, 
        pos_indices: torch.Tensor, 
        neg_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 L2 正则化损失。
        """
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
    
    # =========================================================================
    # Forward 和 Loss 方法
    # =========================================================================
    
    def forward(self, feed_dict: Dict) -> Dict:
        """
        前向传播。
        """
        self.check_list = []
        
        users = feed_dict['user_id']  # [B]
        items = feed_dict['item_id']  # [B, 1+n_neg]
        
        if self.stage == 1:
            # Stage 1: Teacher 预训练，使用全精度嵌入
            user_agg_emb, item_agg_emb = self._propagate_and_aggregate_teacher()
            
            user_emb = user_agg_emb[users]  # [B, d*(L+1)]
            item_emb = item_agg_emb[items.view(-1)].view(
                items.size(0), items.size(1), -1
            )  # [B, 1+n_neg, d*(L+1)]
            
            # 计算得分
            prediction = (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)  # [B, 1+n_neg]
            
            return {
                'prediction': prediction,
                'user_indices': users,
                'item_indices': items,
            }
        
        else:
            # Stage 2: Student 蒸馏训练，使用二值化嵌入
            (user_bin_agg, item_bin_agg, 
             user_bin_layers, item_bin_layers,
             user_bin_weighted_layers, item_bin_weighted_layers) = self._propagate_and_quantize_student()
            
            user_emb = user_bin_agg[users]  # [B, d*(L+1)]
            item_emb = item_bin_agg[items.view(-1)].view(
                items.size(0), items.size(1), -1
            )  # [B, 1+n_neg, d*(L+1)]
            
            # 计算得分
            prediction = (user_emb.unsqueeze(1) * item_emb).sum(dim=-1)  # [B, 1+n_neg]
            
            # 保存中间结果用于损失计算
            # 注意：ID损失使用加权后的嵌入，与原始BiGeaR一致
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
        """
        prediction = out_dict['prediction']
        
        # 获取正负样本得分
        pos_score = prediction[:, 0]
        neg_score = prediction[:, 1:]
        
        # BPR 损失
        if neg_score.size(1) == 1:
            neg_score = neg_score.squeeze(1)
            bpr_loss = F.softplus(neg_score - pos_score).mean()
        else:
            # 多负样本的加权 BPR
            neg_softmax = F.softmax(neg_score, dim=1)
            weighted_neg = (neg_score * neg_softmax).sum(dim=1)
            bpr_loss = F.softplus(weighted_neg - pos_score).mean()
        
        if self.stage == 1:
            # Stage 1: BPR 损失 + 正则化（与原始 BiGeaR 一致）
            user_indices = out_dict['user_indices']
            item_indices = out_dict['item_indices']
            pos_indices = item_indices[:, 0]
            neg_indices = item_indices[:, 1:]
            reg_loss = self._reg_loss(user_indices, pos_indices, neg_indices)
            
            return bpr_loss + self.l2 * reg_loss
        
        else:
            # Stage 2: BPR + ID + Reg
            user_indices = out_dict['user_indices']
            item_indices = out_dict['item_indices']
            # 使用加权后的嵌入计算 ID 损失（与原始 BiGeaR 一致）
            user_bin_weighted_layers = out_dict['user_bin_weighted_layers']
            item_bin_weighted_layers = out_dict['item_bin_weighted_layers']
            
            # ID 损失（使用加权嵌入）
            id_loss = self._id_loss(user_indices, user_bin_weighted_layers, item_bin_weighted_layers)
            
            # 正则化损失
            pos_indices = item_indices[:, 0]
            neg_indices = item_indices[:, 1:]
            reg_loss = self._reg_loss(user_indices, pos_indices, neg_indices)
            
            # 总损失（原始 BiGeaR: loss = bpr + id + l2*reg，其中 id 权重=1.0）
            total_loss = bpr_loss + self.lambda_id * id_loss + self.l2 * reg_loss
            
            return total_loss
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def save_model(self, model_path: str = None):
        """
        保存模型。
        
        Stage 1 结束后，保存的模型可以作为 Stage 2 的 Teacher。
        """
        if model_path is None:
            model_path = self.model_path
        
        from utils import utils
        utils.check_dir(model_path)
        torch.save(self.state_dict(), model_path)
    
    def get_all_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取所有用户和物品的聚合嵌入（用于评估）。
        """
        with torch.no_grad():
            if self.stage == 1:
                return self._propagate_and_aggregate_teacher()
            else:
                user_bin_agg, item_bin_agg, _, _, _, _ = self._propagate_and_quantize_student()
                return user_bin_agg, item_bin_agg
    
    def inference(self, feed_dict: Dict) -> Dict:
        """
        推理方法，用于评估阶段。
        """
        self.check_list = []
        
        users = feed_dict['user_id']  # [B]
        items = feed_dict['item_id']  # [B, n_candidates]
        
        with torch.no_grad():
            user_emb, item_emb = self.get_all_embeddings()
            
            user_e = user_emb[users]  # [B, d*(L+1)]
            item_e = item_emb[items.view(-1)].view(
                items.size(0), items.size(1), -1
            )  # [B, n_candidates, d*(L+1)]
            
            # 计算得分
            prediction = (user_e.unsqueeze(1) * item_e).sum(dim=-1)  # [B, n_candidates]
        
        return {'prediction': prediction}

