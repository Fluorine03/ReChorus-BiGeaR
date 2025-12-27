# -*- coding: UTF-8 -*-
"""
experiment_utils.py: 实验通用工具

提供实验运行、模型加载、结果保存等通用功能。
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from helpers import *
from utils import utils


def setup_logging(log_file: str, level=logging.INFO):
    """设置日志"""
    utils.check_dir(log_file)
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def load_corpus(data_path: str, dataset: str) -> Any:
    """
    加载数据集的 corpus 对象。
    
    Args:
        data_path: 数据目录路径
        dataset: 数据集名称
    
    Returns:
        corpus 对象
    """
    corpus_path = os.path.join(data_path, dataset, 'BaseReader.pkl')
    
    if os.path.exists(corpus_path):
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        return corpus
    else:
        raise FileNotFoundError(f"Corpus 文件不存在: {corpus_path}")


def load_model_weights(model, model_path: str):
    """
    加载模型权重。
    
    Args:
        model: 模型对象
        model_path: 模型权重文件路径
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    state_dict = torch.load(model_path, map_location=model.device)
    model.load_state_dict(state_dict)
    logging.info(f"已加载模型权重: {model_path}")


def get_predictions(model, dataset, runner) -> np.ndarray:
    """
    获取模型在数据集上的预测结果。
    
    Args:
        model: 模型对象
        dataset: 数据集对象
        runner: Runner 对象
    
    Returns:
        预测得分矩阵
    """
    model.eval()
    predictions = runner.predict(dataset)
    return predictions


def evaluate_model(
    model,
    dataset,
    runner,
    topks: List[int] = [5, 10, 20, 50],
    metrics: List[str] = ['HR', 'NDCG']
) -> Dict[str, float]:
    """
    评估模型性能。
    
    Args:
        model: 模型对象
        dataset: 数据集对象
        runner: Runner 对象
        topks: Top-K 列表
        metrics: 指标列表
    
    Returns:
        评估结果字典
    """
    model.eval()
    predictions = runner.predict(dataset)
    results = runner.evaluate_method(predictions, topks, metrics)
    return results


def save_results(results: Dict, save_path: str, format: str = 'json'):
    """
    保存实验结果。
    
    Args:
        results: 结果字典
        save_path: 保存路径
        format: 保存格式 ('json' 或 'csv')
    """
    utils.check_dir(save_path)
    
    if format == 'json':
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif format == 'csv':
        df = pd.DataFrame([results])
        df.to_csv(save_path, index=False)
    
    logging.info(f"结果已保存到: {save_path}")


def format_results_table(
    results_list: List[Dict],
    model_names: List[str],
    metrics: List[str] = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
) -> str:
    """
    格式化结果为 Markdown 表格。
    
    Args:
        results_list: 结果列表
        model_names: 模型名称列表
        metrics: 指标列表
    
    Returns:
        Markdown 表格字符串
    """
    # 表头
    header = "| 模型 | " + " | ".join(metrics) + " |"
    separator = "|" + "---|" * (len(metrics) + 1)
    
    # 数据行
    rows = []
    for name, results in zip(model_names, results_list):
        row = f"| {name} | "
        row += " | ".join([f"{results.get(m, 0):.4f}" for m in metrics])
        row += " |"
        rows.append(row)
    
    table = "\n".join([header, separator] + rows)
    return table


def get_experiment_timestamp() -> str:
    """获取实验时间戳"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int = 0):
    """设置随机种子"""
    utils.init_seed(seed)
    logging.info(f"随机种子设置为: {seed}")


def get_device(gpu: str = '0') -> torch.device:
    """获取设备"""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = torch.device('cpu')
    if gpu != '' and torch.cuda.is_available():
        device = torch.device('cuda')
    logging.info(f"使用设备: {device}")
    return device


class ExperimentConfig:
    """实验配置类"""
    
    def __init__(
        self,
        experiment_name: str,
        dataset: str = 'Grocery_and_Gourmet_Food',
        data_path: str = '../../data',
        seed: int = 0,
        gpu: str = '0',
        **kwargs
    ):
        self.experiment_name = experiment_name
        self.dataset = dataset
        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), data_path))
        self.seed = seed
        self.gpu = gpu
        self.timestamp = get_experiment_timestamp()
        
        # 额外参数
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()
    
    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def run_baseline_evaluation(
    model_class,
    model_path: str,
    corpus,
    device: torch.device,
    runner_class,
    topks: List[int] = [10, 20],
    metrics: List[str] = ['HR', 'NDCG'],
    **model_args
) -> Dict[str, float]:
    """
    运行基线模型评估。
    
    Args:
        model_class: 模型类
        model_path: 模型权重路径
        corpus: 数据集对象
        device: 设备
        runner_class: Runner 类
        topks: Top-K 列表
        metrics: 指标列表
        **model_args: 模型参数
    
    Returns:
        评估结果
    """
    # 创建参数
    parser = argparse.ArgumentParser()
    parser = model_class.parse_model_args(parser)
    parser = runner_class.parse_runner_args(parser)
    
    # 设置默认参数
    defaults = {
        'lr': 0.001,
        'l2': 1e-6,
        'batch_size': 256,
        'eval_batch_size': 256,
        'epoch': 200,
        'early_stop': 10,
        'topk': ','.join(map(str, topks)),
        'metric': ','.join(metrics),
        'num_workers': 0,
        'pin_memory': 0,
        'train': 0,  # 不训练
    }
    defaults.update(model_args)
    
    args = argparse.Namespace(**defaults)
    args.device = device
    
    # 创建模型
    model = model_class(args, corpus).to(device)
    
    # 加载权重
    load_model_weights(model, model_path)
    
    # 创建数据集
    test_dataset = model_class.Dataset(model, corpus, 'test')
    test_dataset.prepare()
    
    # 创建 runner
    runner = runner_class(args)
    
    # 评估
    results = evaluate_model(model, test_dataset, runner, topks, metrics)
    
    return results


# 预定义的模型路径（相对于 Lab 目录）
MODEL_PATHS = {
    'BiGeaR_Stage1': '../record/BiGeaR_LightGCN_v2/controlled_stage1_models/BiGeaR_LightGCN_v2__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__stage=1.pt',
    'BiGeaR_Stage2': '../record/BiGeaR_LightGCN_v2/controlled_stage2_models/BiGeaR_LightGCN_v2__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__stage=2__lambda_id=1.0__top_r=100.pt',
    'LightGCN': '../record/LightGCN/controlled_models/LightGCN__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__batch_size=256.pt',
}


def get_model_path(model_name: str, base_dir: str = None) -> str:
    """获取模型路径"""
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    
    if model_name in MODEL_PATHS:
        return os.path.abspath(os.path.join(base_dir, MODEL_PATHS[model_name]))
    else:
        return model_name  # 假设传入的是绝对路径

