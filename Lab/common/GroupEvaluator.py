# -*- coding: UTF-8 -*-
"""
GroupEvaluator: 分组评估器

用于实验二（用户活跃度分析）和实验三（物品热门度分析）的分组评估。

功能：
    1. 支持按用户分组评估（用户活跃度分析）
    2. 支持按物品分组评估（物品热门度分析）
    3. 计算分组 Hit@K 和 NDCG@K 指标
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


class GroupEvaluator:
    """
    分组评估器：支持用户分组和物品分组的评估。
    
    使用方法：
        evaluator = GroupEvaluator(corpus, group_type='user')
        evaluator.compute_groups(percentile_low=20, percentile_high=80)
        results = evaluator.evaluate_grouped(predictions, test_data, topks=[10, 20])
    """
    
    def __init__(self, corpus, group_type: str = 'user'):
        """
        初始化分组评估器。
        
        Args:
            corpus: ReChorus 的 corpus 对象，包含 train_clicked_set 等
            group_type: 'user' 或 'item'，指定分组类型
        """
        self.corpus = corpus
        self.group_type = group_type
        self.groups = {}  # {'high': set(), 'low': set(), 'medium': set()}
        self.statistics = {}  # 统计信息
        
        # 计算基础统计
        self._compute_statistics()
    
    def _compute_statistics(self):
        """计算用户活跃度或物品热门度的基础统计"""
        if self.group_type == 'user':
            # 统计每个用户的交互次数
            self.statistics = {}
            for user, items in self.corpus.train_clicked_set.items():
                self.statistics[user] = len(items)
        elif self.group_type == 'item':
            # 统计每个物品的被交互次数
            self.statistics = defaultdict(int)
            for user, items in self.corpus.train_clicked_set.items():
                for item in items:
                    self.statistics[item] += 1
            self.statistics = dict(self.statistics)
        else:
            raise ValueError(f"不支持的分组类型: {self.group_type}")
    
    def compute_groups(
        self, 
        percentile_low: float = 20, 
        percentile_high: float = 80,
        include_medium: bool = False
    ):
        """
        计算分组。
        
        Args:
            percentile_low: 低活跃/冷门组的百分位阈值（取后 percentile_low%）
            percentile_high: 高活跃/热门组的百分位阈值（取前 100-percentile_high%）
            include_medium: 是否包含中间组
        
        分组逻辑：
            - 高活跃/热门组：统计值 >= percentile_high 百分位
            - 低活跃/冷门组：统计值 <= percentile_low 百分位
            - 中间组：其余
        """
        values = list(self.statistics.values())
        
        threshold_low = np.percentile(values, percentile_low)
        threshold_high = np.percentile(values, percentile_high)
        
        self.groups = {
            'high': set(),
            'low': set(),
        }
        if include_medium:
            self.groups['medium'] = set()
        
        for entity_id, count in self.statistics.items():
            if count >= threshold_high:
                self.groups['high'].add(entity_id)
            elif count <= threshold_low:
                self.groups['low'].add(entity_id)
            elif include_medium:
                self.groups['medium'].add(entity_id)
        
        # 记录分组信息
        self.group_info = {
            'threshold_low': threshold_low,
            'threshold_high': threshold_high,
            'percentile_low': percentile_low,
            'percentile_high': percentile_high,
            'group_sizes': {k: len(v) for k, v in self.groups.items()},
        }
        
        return self.group_info
    
    def get_group_statistics(self) -> Dict:
        """获取分组的详细统计信息"""
        stats = {
            'type': self.group_type,
            'total_entities': len(self.statistics),
            'group_info': self.group_info,
        }
        
        # 计算每个组的平均统计值
        for group_name, group_set in self.groups.items():
            if len(group_set) > 0:
                group_values = [self.statistics[e] for e in group_set if e in self.statistics]
                stats[f'{group_name}_avg'] = np.mean(group_values) if group_values else 0
                stats[f'{group_name}_std'] = np.std(group_values) if group_values else 0
                stats[f'{group_name}_min'] = np.min(group_values) if group_values else 0
                stats[f'{group_name}_max'] = np.max(group_values) if group_values else 0
        
        return stats
    
    def evaluate_grouped(
        self, 
        predictions: np.ndarray, 
        test_data: pd.DataFrame,
        topks: List[int] = [10, 20],
        metrics: List[str] = ['HR', 'NDCG']
    ) -> Dict[str, Dict[str, float]]:
        """
        分组评估。
        
        Args:
            predictions: 预测得分矩阵，shape = (n_test_samples, n_candidates)
                        第一列是正样本的得分
            test_data: 测试集 DataFrame，包含 user_id, item_id 列
            topks: Top-K 列表
            metrics: 指标列表
        
        Returns:
            分组评估结果，格式为 {'high': {'HR@10': ..., 'NDCG@10': ...}, 'low': {...}, 'all': {...}}
        """
        results = {}
        
        # 根据分组类型确定分组依据
        if self.group_type == 'user':
            group_key = 'user_id'
        else:
            group_key = 'item_id'
        
        # 为每个样本分配组别
        sample_groups = []
        for idx, row in test_data.iterrows():
            entity_id = row[group_key]
            group = 'other'
            for group_name, group_set in self.groups.items():
                if entity_id in group_set:
                    group = group_name
                    break
            sample_groups.append(group)
        
        sample_groups = np.array(sample_groups)
        
        # 计算每个组的指标
        for group_name in list(self.groups.keys()) + ['all']:
            if group_name == 'all':
                mask = np.ones(len(predictions), dtype=bool)
            else:
                mask = sample_groups == group_name
            
            if mask.sum() == 0:
                results[group_name] = {f'{m}@{k}': 0.0 for m in metrics for k in topks}
                continue
            
            group_predictions = predictions[mask]
            group_results = self._evaluate_method(group_predictions, topks, metrics)
            results[group_name] = group_results
            results[group_name]['n_samples'] = int(mask.sum())
        
        return results
    
    def _evaluate_method(
        self, 
        predictions: np.ndarray, 
        topks: List[int], 
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        计算评估指标。
        
        使用与 ReChorus BaseRunner.evaluate_method 相同的逻辑。
        
        Args:
            predictions: 预测得分，第一列是正样本
            topks: Top-K 列表
            metrics: 指标列表
        
        Returns:
            指标字典
        """
        evaluations = {}
        
        # 计算 ground-truth 的排名（第一列是正样本）
        gt_rank = (predictions >= predictions[:, 0].reshape(-1, 1)).sum(axis=-1)
        
        for k in topks:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = f'{metric}@{k}'
                if metric == 'HR':
                    evaluations[key] = float(hit.mean())
                elif metric == 'NDCG':
                    evaluations[key] = float((hit / np.log2(gt_rank + 1)).mean())
                else:
                    raise ValueError(f'未定义的评估指标: {metric}')
        
        return evaluations


def evaluate_model_grouped(
    model,
    corpus,
    test_dataset,
    runner,
    group_type: str = 'user',
    percentile_low: float = 20,
    percentile_high: float = 80,
    topks: List[int] = [10, 20],
    metrics: List[str] = ['HR', 'NDCG']
) -> Tuple[Dict, Dict]:
    """
    便捷函数：对模型进行分组评估。
    
    Args:
        model: 模型对象
        corpus: 数据集对象
        test_dataset: 测试数据集
        runner: Runner 对象
        group_type: 分组类型 ('user' 或 'item')
        percentile_low: 低分组百分位
        percentile_high: 高分组百分位
        topks: Top-K 列表
        metrics: 指标列表
    
    Returns:
        (分组评估结果, 分组统计信息)
    """
    # 创建分组评估器
    evaluator = GroupEvaluator(corpus, group_type=group_type)
    evaluator.compute_groups(percentile_low, percentile_high)
    
    # 获取预测结果
    predictions = runner.predict(test_dataset)
    
    # 获取测试数据
    test_data = corpus.data_df['test']
    
    # 分组评估
    results = evaluator.evaluate_grouped(
        predictions, test_data, topks=topks, metrics=metrics
    )
    
    # 获取统计信息
    stats = evaluator.get_group_statistics()
    
    return results, stats


def print_grouped_results(results: Dict, title: str = "分组评估结果"):
    """打印分组评估结果的辅助函数"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print('='*60)
    
    # 表头
    metrics = [k for k in results.get('all', {}).keys() if k != 'n_samples']
    header = f"{'组别':<12}" + "".join([f"{m:<12}" for m in metrics]) + f"{'样本数':<10}"
    print(header)
    print('-' * len(header))
    
    # 数据行
    for group_name in ['high', 'medium', 'low', 'all']:
        if group_name not in results:
            continue
        group_data = results[group_name]
        n_samples = group_data.get('n_samples', 0)
        
        row = f"{group_name:<12}"
        for m in metrics:
            value = group_data.get(m, 0)
            row += f"{value:.4f}      "
        row += f"{n_samples:<10}"
        print(row)
    
    print('='*60)

