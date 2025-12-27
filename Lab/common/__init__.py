# -*- coding: UTF-8 -*-
"""
Lab/common 共享模块

包含：
    - BiGeaR_LightGCN: 带消融开关的 BiGeaR 模型
    - GroupEvaluator: 分组评估器
    - data_statistics: 数据统计工具
    - experiment_utils: 实验通用工具
"""

from .GroupEvaluator import GroupEvaluator, evaluate_model_grouped, print_grouped_results
from .data_statistics import (
    compute_user_activity,
    compute_item_popularity,
    get_statistics_summary,
    get_group_entities,
    analyze_dataset
)
from .experiment_utils import (
    setup_logging,
    load_corpus,
    load_model_weights,
    get_predictions,
    evaluate_model,
    save_results,
    format_results_table,
    get_experiment_timestamp,
    set_seed,
    get_device,
    ExperimentConfig,
    MODEL_PATHS,
    get_model_path
)

__all__ = [
    'GroupEvaluator',
    'evaluate_model_grouped',
    'print_grouped_results',
    'compute_user_activity',
    'compute_item_popularity',
    'get_statistics_summary',
    'get_group_entities',
    'analyze_dataset',
    'setup_logging',
    'load_corpus',
    'load_model_weights',
    'get_predictions',
    'evaluate_model',
    'save_results',
    'format_results_table',
    'get_experiment_timestamp',
    'set_seed',
    'get_device',
    'ExperimentConfig',
    'MODEL_PATHS',
    'get_model_path',
]

