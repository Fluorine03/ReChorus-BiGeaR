# -*- coding: UTF-8 -*-
"""
实验三：物品热门度分层分析

目标：分析 BiGeaR 对热门物品和冷门物品的推荐能力差异

实验设计：
    1. 统计每个物品在训练集中的被交互次数
    2. 按热门度排序，取前 20% 为热门组，后 50% 为冷门组
    3. 按测试集中目标物品的热门度分组评估各模型性能
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
from common.data_statistics import compute_item_popularity, get_statistics_summary, print_statistics_summary
from helpers.BaseRunner import BaseRunner
from utils import utils

from models.general.LightGCN import LightGCN
from models.general.BPRMF import BPRMF


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='物品热门度分层分析')
    
    parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')
    parser.add_argument('--data_path', type=str, default='../../data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    
    # 分组参数
    parser.add_argument('--percentile_low', type=float, default=50,
                        help='冷门组百分位阈值（后 X%）')
    parser.add_argument('--percentile_high', type=float, default=80,
                        help='热门组百分位阈值（前 100-X%）')
    
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
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f"已加载 BiGeaR Stage {stage} 模型: {model_path}")
    else:
        logging.warning(f"模型文件不存在: {model_path}")
        return None, None, None
    
    test_dataset = BiGeaR_LightGCN.Dataset(model, corpus, 'test')
    test_dataset.prepare()
    
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
    """按物品分组评估模型"""
    model.eval()
    
    predictions = runner.predict(test_dataset)
    test_data = corpus.data_df['test']
    results = evaluator.evaluate_grouped(predictions, test_data, topks=topks, metrics=metrics)
    
    return results


def main():
    args = setup_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(result_dir, f'item_analysis_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*60)
    logging.info("实验三：物品热门度分层分析")
    logging.info("="*60)
    
    utils.init_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if args.gpu != '' and torch.cuda.is_available() else 'cpu')
    logging.info(f"设备: {device}")
    
    data_path = os.path.abspath(os.path.join(base_dir, args.data_path))
    corpus_path = os.path.join(data_path, args.dataset, 'BaseReader.pkl')
    
    logging.info(f"加载数据集: {corpus_path}")
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    
    # 物品热门度统计
    logging.info("\n" + "="*60)
    logging.info("物品热门度统计")
    logging.info("="*60)
    
    item_popularity = compute_item_popularity(corpus)
    item_summary = get_statistics_summary(item_popularity, "物品热门度")
    print_statistics_summary(item_summary)
    
    # 创建分组评估器（按物品分组）
    evaluator = GroupEvaluator(corpus, group_type='item')
    group_info = evaluator.compute_groups(
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high
    )
    
    logging.info(f"\n分组信息:")
    logging.info(f"  冷门阈值: <= {group_info['threshold_low']:.0f} 次被交互")
    logging.info(f"  热门阈值: >= {group_info['threshold_high']:.0f} 次被交互")
    logging.info(f"  热门物品数: {group_info['group_sizes']['high']}")
    logging.info(f"  冷门物品数: {group_info['group_sizes']['low']}")
    
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
    logging.info("物品热门度分层分析汇总")
    logging.info("="*60)
    
    metric_keys = [f'{m}@{k}' for m in metrics for k in topks]
    
    for group in ['high', 'low', 'all']:
        group_name = {'high': '热门物品', 'low': '冷门物品', 'all': '全部物品'}[group]
        logging.info(f"\n{group_name}组:")
        header = f"{'模型':<20} | " + " | ".join([f"{m:<10}" for m in metric_keys])
        logging.info(header)
        logging.info("-" * len(header))
        
        for model_name, results in all_results.items():
            if group in results:
                row = f"{model_name:<20} | "
                row += " | ".join([f"{results[group].get(m, 0):.4f}    " for m in metric_keys])
                logging.info(row)
    
    # 分析热门vs冷门性能差异
    logging.info("\n" + "="*60)
    logging.info("热门物品 vs 冷门物品 性能差异分析")
    logging.info("="*60)
    
    for model_name, results in all_results.items():
        logging.info(f"\n{model_name}:")
        for m in metric_keys:
            hot_val = results.get('high', {}).get(m, 0)
            cold_val = results.get('low', {}).get(m, 0)
            if cold_val > 0:
                ratio = hot_val / cold_val
                diff = (hot_val - cold_val) / cold_val * 100
                logging.info(f"  {m}: 热门/冷门 = {ratio:.2f}x, 差异 = {diff:+.1f}%")
    
    # 长尾推荐能力分析
    logging.info("\n" + "="*60)
    logging.info("长尾推荐能力对比")
    logging.info("="*60)
    
    logging.info("\n冷门物品（长尾）上的表现对比:")
    if 'low' in all_results.get('BiGeaR_Stage2', {}):
        for m in metric_keys:
            logging.info(f"\n{m}:")
            for model_name in ['BiGeaR_Stage2', 'BiGeaR_Stage1', 'LightGCN']:
                if model_name in all_results and 'low' in all_results[model_name]:
                    val = all_results[model_name]['low'].get(m, 0)
                    logging.info(f"  {model_name}: {val:.4f}")
    
    # 保存结果
    save_results = {
        'experiment': '物品热门度分层分析',
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
        'item_statistics': {
            'count': item_summary['count'],
            'mean': float(item_summary['mean']),
            'std': float(item_summary['std']),
            'min': float(item_summary['min']),
            'max': float(item_summary['max']),
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
    
    result_file = os.path.join(result_dir, f'item_analysis_results_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n结果已保存到: {result_file}")
    logging.info("\n实验三完成！")


if __name__ == '__main__':
    main()

