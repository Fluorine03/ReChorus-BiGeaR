# -*- coding: UTF-8 -*-
"""
data_statistics.py: 数据统计工具

用于计算用户活跃度和物品热门度的统计信息。
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def compute_user_activity(corpus) -> Dict[int, int]:
    """
    计算用户活跃度（训练集中的交互次数）。
    
    Args:
        corpus: ReChorus corpus 对象
    
    Returns:
        字典：{user_id: interaction_count}
    """
    user_activity = {}
    for user, items in corpus.train_clicked_set.items():
        user_activity[user] = len(items)
    return user_activity


def compute_item_popularity(corpus) -> Dict[int, int]:
    """
    计算物品热门度（训练集中的被交互次数）。
    
    Args:
        corpus: ReChorus corpus 对象
    
    Returns:
        字典：{item_id: interaction_count}
    """
    item_popularity = defaultdict(int)
    for user, items in corpus.train_clicked_set.items():
        for item in items:
            item_popularity[item] += 1
    return dict(item_popularity)


def get_statistics_summary(statistics: Dict[int, int], name: str = "统计") -> Dict:
    """
    获取统计摘要信息。
    
    Args:
        statistics: 统计字典
        name: 统计名称
    
    Returns:
        摘要字典
    """
    values = list(statistics.values())
    
    summary = {
        'name': name,
        'count': len(values),
        'sum': sum(values),
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'percentile_10': np.percentile(values, 10),
        'percentile_20': np.percentile(values, 20),
        'percentile_50': np.percentile(values, 50),
        'percentile_80': np.percentile(values, 80),
        'percentile_90': np.percentile(values, 90),
    }
    
    return summary


def get_group_entities(
    statistics: Dict[int, int],
    percentile_low: float = 20,
    percentile_high: float = 80
) -> Tuple[set, set, set]:
    """
    根据百分位数分组。
    
    Args:
        statistics: 统计字典
        percentile_low: 低组百分位阈值
        percentile_high: 高组百分位阈值
    
    Returns:
        (high_group, medium_group, low_group) 三个集合
    """
    values = list(statistics.values())
    threshold_low = np.percentile(values, percentile_low)
    threshold_high = np.percentile(values, percentile_high)
    
    high_group = set()
    medium_group = set()
    low_group = set()
    
    for entity_id, count in statistics.items():
        if count >= threshold_high:
            high_group.add(entity_id)
        elif count <= threshold_low:
            low_group.add(entity_id)
        else:
            medium_group.add(entity_id)
    
    return high_group, medium_group, low_group


def print_statistics_summary(summary: Dict):
    """打印统计摘要"""
    print(f"\n{'='*50}")
    print(f"{summary['name']} 统计摘要")
    print('='*50)
    print(f"实体数量: {summary['count']}")
    print(f"总交互数: {summary['sum']}")
    print(f"平均值: {summary['mean']:.2f}")
    print(f"标准差: {summary['std']:.2f}")
    print(f"最小值: {summary['min']}")
    print(f"最大值: {summary['max']}")
    print(f"中位数: {summary['median']:.2f}")
    print(f"10%分位: {summary['percentile_10']:.2f}")
    print(f"20%分位: {summary['percentile_20']:.2f}")
    print(f"80%分位: {summary['percentile_80']:.2f}")
    print(f"90%分位: {summary['percentile_90']:.2f}")
    print('='*50)


def analyze_dataset(corpus, dataset_name: str = "数据集") -> Dict:
    """
    分析数据集的用户活跃度和物品热门度分布。
    
    Args:
        corpus: ReChorus corpus 对象
        dataset_name: 数据集名称
    
    Returns:
        分析结果字典
    """
    print(f"\n{'#'*60}")
    print(f"# {dataset_name} 数据统计分析")
    print('#'*60)
    
    # 用户活跃度
    user_activity = compute_user_activity(corpus)
    user_summary = get_statistics_summary(user_activity, "用户活跃度")
    print_statistics_summary(user_summary)
    
    # 物品热门度
    item_popularity = compute_item_popularity(corpus)
    item_summary = get_statistics_summary(item_popularity, "物品热门度")
    print_statistics_summary(item_summary)
    
    # 分组统计
    print(f"\n{'='*50}")
    print("分组统计 (20%/80% 百分位)")
    print('='*50)
    
    user_high, user_medium, user_low = get_group_entities(user_activity, 20, 80)
    print(f"用户分组: 高活跃 {len(user_high)}, 中等 {len(user_medium)}, 低活跃 {len(user_low)}")
    
    item_high, item_medium, item_low = get_group_entities(item_popularity, 20, 80)
    print(f"物品分组: 热门 {len(item_high)}, 中等 {len(item_medium)}, 冷门 {len(item_low)}")
    
    return {
        'user_activity': user_activity,
        'user_summary': user_summary,
        'item_popularity': item_popularity,
        'item_summary': item_summary,
        'user_groups': (user_high, user_medium, user_low),
        'item_groups': (item_high, item_medium, item_low),
    }


if __name__ == '__main__':
    # 测试代码
    import pickle
    
    # 加载 corpus
    corpus_path = '../../data/Grocery_and_Gourmet_Food/BaseReader.pkl'
    if os.path.exists(corpus_path):
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        
        analyze_dataset(corpus, "Grocery_and_Gourmet_Food")

