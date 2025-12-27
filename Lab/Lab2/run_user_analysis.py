# -*- coding: UTF-8 -*-
"""
实验二：用户活跃度分层分析

目标：分析 BiGeaR 在不同活跃度用户群体上的表现差异

实验设计：
    1. 统计每个用户在训练集中的交互次数
    2. 按交互次数排序，取前 20% 为高活跃组，后 20% 为低活跃组
    3. 分别评估 BiGeaR、LightGCN 等模型在各组上的性能
"""

import os
import sys
import json
import argparse
import logging
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from common.GroupEvaluator import GroupEvaluator, print_grouped_results
from common.data_statistics import compute_user_activity, get_statistics_summary, print_statistics_summary
from helpers.BaseRunner import BaseRunner
from utils import utils

# 导入模型
from models.general.LightGCN import LightGCN
from models.general.BPRMF import BPRMF


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='用户活跃度分层分析')
    
    parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')
    parser.add_argument('--data_path', type=str, default='../../data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    
    # 分组参数
    parser.add_argument('--percentile_low', type=float, default=20,
                        help='低活跃组百分位阈值（后 X%）')
    parser.add_argument('--percentile_high', type=float, default=80,
                        help='高活跃组百分位阈值（前 100-X%）')
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--topks', type=str, default='10,20')
    parser.add_argument('--metrics', type=str, default='HR,NDCG')
    
    return parser.parse_args()


def load_and_evaluate_bigear(corpus, device, model_path, stage, args, teacher_path=''):
    """加载并评估 BiGeaR 模型"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from common.BiGeaR_LightGCN import BiGeaR_LightGCN
    
    # Stage 2 需要 teacher_model_path
    if stage == 2 and not teacher_path:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        teacher_path = os.path.abspath(os.path.join(base_dir, 
            '../../record/BiGeaR_LightGCN_v2/controlled_stage1_models/BiGeaR_LightGCN_v2__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__stage=1.pt'))
    
    model_args = argparse.Namespace(
        emb_size=64,
        n_layers=2,
        stage=stage,
        norm_a=1.0,
        lambda_id=1.0,
        top_r=100,
        lambda_decay=0.1,
        teacher_model_path=teacher_path,
        use_distillation=1,
        lr=0.001,
        l2=1e-6,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_neg=1,
        dropout=0,
        train=0,
        test_all=0,
        buffer=1,
        num_workers=0,
        pin_memory=0,
        topk=args.topks,
        metric=args.metrics,
        main_metric='NDCG@10',
        epoch=1,
        early_stop=10,
        check_epoch=1,
        test_epoch=-1,
        optimizer='Adam',
        verbose=logging.INFO,
        random_seed=args.seed,
        model_path=model_path,
        log_file='',
    )
    model_args.device = device
    
    model = BiGeaR_LightGCN(model_args, corpus).to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"已加载 BiGeaR Stage {stage} 模型: {model_path}")
    else:
        logging.warning(f"模型文件不存在: {model_path}")
        return None, None
    
    # 创建数据集
    test_dataset = BiGeaR_LightGCN.Dataset(model, corpus, 'test')
    test_dataset.prepare()
    
    # 创建 Runner
    runner = BaseRunner(model_args)
    
    return model, test_dataset, runner


def load_and_evaluate_lightgcn(corpus, device, model_path, args):
    """加载并评估 LightGCN 模型"""
    model_args = argparse.Namespace(
        emb_size=64,
        n_layers=2,
        lr=0.001,
        l2=1e-6,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_neg=1,
        dropout=0,
        train=0,
        test_all=0,
        buffer=1,
        num_workers=0,
        pin_memory=0,
        topk=args.topks,
        metric=args.metrics,
        main_metric='NDCG@10',
        epoch=1,
        early_stop=10,
        check_epoch=1,
        test_epoch=-1,
        optimizer='Adam',
        verbose=logging.INFO,
        random_seed=args.seed,
        model_path=model_path,
        log_file='',
    )
    model_args.device = device
    
    model = LightGCN(model_args, corpus).to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"已加载 LightGCN 模型: {model_path}")
    else:
        logging.warning(f"模型文件不存在: {model_path}")
        return None, None, None
    
    test_dataset = LightGCN.Dataset(model, corpus, 'test')
    test_dataset.prepare()
    
    runner = BaseRunner(model_args)
    
    return model, test_dataset, runner


def evaluate_model_by_groups(model, test_dataset, runner, evaluator, corpus, topks, metrics):
    """按用户分组评估模型"""
    model.eval()
    
    # 获取预测结果
    predictions = runner.predict(test_dataset)
    
    # 获取测试数据
    test_data = corpus.data_df['test']
    
    # 分组评估
    results = evaluator.evaluate_grouped(predictions, test_data, topks=topks, metrics=metrics)
    
    return results


def main():
    args = setup_args()
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_dir, f'user_analysis_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*60)
    logging.info("实验二：用户活跃度分层分析")
    logging.info("="*60)
    
    # 设置随机种子和设备
    utils.init_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if args.gpu != '' and torch.cuda.is_available() else 'cpu')
    logging.info(f"设备: {device}")
    
    # 加载数据
    data_path = os.path.abspath(os.path.join(base_dir, args.data_path))
    corpus_path = os.path.join(data_path, args.dataset, 'BaseReader.pkl')
    
    logging.info(f"加载数据集: {corpus_path}")
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    
    # 用户活跃度统计
    logging.info("\n" + "="*60)
    logging.info("用户活跃度统计")
    logging.info("="*60)
    
    user_activity = compute_user_activity(corpus)
    user_summary = get_statistics_summary(user_activity, "用户活跃度")
    print_statistics_summary(user_summary)
    
    # 创建分组评估器
    evaluator = GroupEvaluator(corpus, group_type='user')
    group_info = evaluator.compute_groups(
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high
    )
    
    logging.info(f"\n分组信息:")
    logging.info(f"  低活跃阈值: <= {group_info['threshold_low']:.0f} 次交互")
    logging.info(f"  高活跃阈值: >= {group_info['threshold_high']:.0f} 次交互")
    logging.info(f"  高活跃组用户数: {group_info['group_sizes']['high']}")
    logging.info(f"  低活跃组用户数: {group_info['group_sizes']['low']}")
    
    # 解析评估参数
    topks = [int(x) for x in args.topks.split(',')]
    metrics = [m.strip().upper() for m in args.metrics.split(',')]
    
    # 模型路径
    model_paths = {
        'BiGeaR_Stage1': os.path.abspath(os.path.join(base_dir, 
            '../../record/BiGeaR_LightGCN_v2/controlled_stage1_models/BiGeaR_LightGCN_v2__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__stage=1.pt')),
        'BiGeaR_Stage2': os.path.abspath(os.path.join(base_dir,
            '../../record/BiGeaR_LightGCN_v2/controlled_stage2_models/BiGeaR_LightGCN_v2__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__stage=2__lambda_id=1.0__top_r=100.pt')),
        'LightGCN': os.path.abspath(os.path.join(base_dir,
            '../../record/LightGCN/controlled_models/LightGCN__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__batch_size=256.pt')),
    }
    
    # 评估各模型
    all_results = {}
    
    # BiGeaR Stage 1
    logging.info("\n" + "="*60)
    logging.info("评估 BiGeaR Stage 1 (Teacher)")
    logging.info("="*60)
    
    if os.path.exists(model_paths['BiGeaR_Stage1']):
        model, test_dataset, runner = load_and_evaluate_bigear(
            corpus, device, model_paths['BiGeaR_Stage1'], stage=1, args=args
        )
        if model is not None:
            results = evaluate_model_by_groups(model, test_dataset, runner, evaluator, corpus, topks, metrics)
            all_results['BiGeaR_Stage1'] = results
            print_grouped_results(results, "BiGeaR Stage 1 分组评估结果")
    
    # BiGeaR Stage 2
    logging.info("\n" + "="*60)
    logging.info("评估 BiGeaR Stage 2 (Binary Student)")
    logging.info("="*60)
    
    if os.path.exists(model_paths['BiGeaR_Stage2']):
        model, test_dataset, runner = load_and_evaluate_bigear(
            corpus, device, model_paths['BiGeaR_Stage2'], stage=2, args=args
        )
        if model is not None:
            results = evaluate_model_by_groups(model, test_dataset, runner, evaluator, corpus, topks, metrics)
            all_results['BiGeaR_Stage2'] = results
            print_grouped_results(results, "BiGeaR Stage 2 分组评估结果")
    
    # LightGCN
    logging.info("\n" + "="*60)
    logging.info("评估 LightGCN")
    logging.info("="*60)
    
    if os.path.exists(model_paths['LightGCN']):
        model, test_dataset, runner = load_and_evaluate_lightgcn(
            corpus, device, model_paths['LightGCN'], args
        )
        if model is not None:
            results = evaluate_model_by_groups(model, test_dataset, runner, evaluator, corpus, topks, metrics)
            all_results['LightGCN'] = results
            print_grouped_results(results, "LightGCN 分组评估结果")
    
    # 汇总对比
    logging.info("\n" + "="*60)
    logging.info("用户活跃度分层分析汇总")
    logging.info("="*60)
    
    # 打印对比表格
    metric_keys = [f'{m}@{k}' for m in metrics for k in topks]
    
    for group in ['high', 'low', 'all']:
        logging.info(f"\n{group.upper()} 活跃度组:")
        header = f"{'模型':<20} | " + " | ".join([f"{m:<10}" for m in metric_keys])
        logging.info(header)
        logging.info("-" * len(header))
        
        for model_name, results in all_results.items():
            if group in results:
                row = f"{model_name:<20} | "
                row += " | ".join([f"{results[group].get(m, 0):.4f}    " for m in metric_keys])
                logging.info(row)
    
    # 分析高低活跃用户的性能差距
    logging.info("\n" + "="*60)
    logging.info("高活跃 vs 低活跃 性能差异分析")
    logging.info("="*60)
    
    for model_name, results in all_results.items():
        logging.info(f"\n{model_name}:")
        for m in metric_keys:
            high_val = results.get('high', {}).get(m, 0)
            low_val = results.get('low', {}).get(m, 0)
            if low_val > 0:
                ratio = high_val / low_val
                diff = (high_val - low_val) / low_val * 100
                logging.info(f"  {m}: 高/低 = {ratio:.2f}x, 差异 = {diff:+.1f}%")
    
    # 保存结果
    save_results = {
        'experiment': '用户活跃度分层分析',
        'timestamp': timestamp,
        'config': {
            'dataset': args.dataset,
            'percentile_low': args.percentile_low,
            'percentile_high': args.percentile_high,
        },
        'group_info': {
            'threshold_low': float(group_info['threshold_low']),
            'threshold_high': float(group_info['threshold_high']),
            'group_sizes': group_info['group_sizes'],
        },
        'user_statistics': {
            'count': user_summary['count'],
            'mean': float(user_summary['mean']),
            'std': float(user_summary['std']),
            'min': float(user_summary['min']),
            'max': float(user_summary['max']),
        },
        'results': {}
    }
    
    for model_name, results in all_results.items():
        save_results['results'][model_name] = {}
        for group, group_results in results.items():
            save_results['results'][model_name][group] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v 
                for k, v in group_results.items()
            }
    
    result_file = os.path.join(result_dir, f'user_analysis_results_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n结果已保存到: {result_file}")
    logging.info("\n实验二完成！")


if __name__ == '__main__':
    main()

