# -*- coding: UTF-8 -*-
"""
实验四：超参数敏感性分析 - Embedding 维度

目标：探索 Embedding 维度对 BiGeaR 性能的影响

实验设计：
    1. 测试维度：16, 32, 64, 128, 256
    2. 每个维度完整运行 Stage 1 + Stage 2
    3. 记录性能指标和训练时间
"""

import os
import sys
import json
import argparse
import logging
import time
import pickle
import torch
import numpy as np
from datetime import datetime

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from common.BiGeaR_LightGCN import BiGeaR_LightGCN
from helpers.BaseRunner import BaseRunner
from utils import utils


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='超参数敏感性分析 - Embedding 维度')
    
    parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')
    parser.add_argument('--data_path', type=str, default='../../data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    
    # 维度列表
    parser.add_argument('--emb_sizes', type=str, default='16,32,64,128',
                        help='要测试的 Embedding 维度列表')
    
    # 其他模型参数
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lambda_id', type=float, default=1.0)
    parser.add_argument('--top_r', type=int, default=100)
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10)
    
    return parser.parse_args()


def train_model(args, corpus, emb_size, stage, save_dir, teacher_path=None):
    """
    训练指定维度的模型
    
    Args:
        args: 命令行参数
        corpus: 数据集
        emb_size: Embedding 维度
        stage: 训练阶段 (1 或 2)
        save_dir: 保存目录
        teacher_path: Teacher 模型路径 (Stage 2 需要)
    
    Returns:
        结果字典
    """
    logging.info(f"\n{'='*60}")
    logging.info(f"训练 Stage {stage}, emb_size = {emb_size}")
    logging.info(f"{'='*60}")
    
    # 模型保存路径
    model_path = os.path.join(save_dir, f'BiGeaR_emb{emb_size}_stage{stage}.pt')
    
    model_args = argparse.Namespace(
        emb_size=emb_size,
        n_layers=args.n_layers,
        stage=stage,
        norm_a=1.0,
        lambda_id=args.lambda_id,
        top_r=args.top_r,
        lambda_decay=0.1,
        teacher_model_path=teacher_path if teacher_path else '',
        use_distillation=1,
        
        lr=args.lr,
        l2=args.l2,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        epoch=args.epoch,
        early_stop=args.early_stop,
        
        num_neg=1,
        dropout=0,
        train=1,
        test_all=0,
        buffer=1,
        num_workers=0,
        pin_memory=0,
        check_epoch=1,
        test_epoch=-1,
        optimizer='Adam',
        topk='5,10,20,50',
        metric='NDCG,HR',
        main_metric='NDCG@10',
        verbose=logging.INFO,
        random_seed=args.seed,
        model_path=model_path,
        log_file='',
    )
    
    device = torch.device('cuda' if args.gpu != '' and torch.cuda.is_available() else 'cpu')
    model_args.device = device
    
    # 创建模型
    model = BiGeaR_LightGCN(model_args, corpus).to(device)
    logging.info(f"模型参数量: {model.count_variables()}")
    
    # 创建数据集
    data_dict = {}
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = BiGeaR_LightGCN.Dataset(model, corpus, phase)
        data_dict[phase].prepare()
    
    runner = BaseRunner(model_args)
    
    # 训练
    start_time = time.time()
    runner.train(data_dict)
    training_time = time.time() - start_time
    
    # 加载最佳模型
    model.load_model()
    
    # 评估
    dev_result = runner.evaluate(data_dict['dev'], runner.topk, runner.metrics)
    test_result = runner.evaluate(data_dict['test'], runner.topk, runner.metrics)
    
    logging.info(f"训练完成: emb_size={emb_size}, stage={stage}")
    logging.info(f"训练时间: {training_time:.2f}s")
    logging.info(f"Test 结果: {utils.format_metric(test_result)}")
    
    return {
        'emb_size': emb_size,
        'stage': stage,
        'dev': dev_result,
        'test': test_result,
        'training_time': training_time,
        'model_path': model_path,
        'n_params': model.count_variables(),
    }


def main():
    args = setup_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'models')
    log_dir = os.path.join(base_dir, 'logs')
    result_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'hyperparam_experiment_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*60)
    logging.info("实验四：超参数敏感性分析 - Embedding 维度")
    logging.info("="*60)
    
    # 解析维度列表
    emb_sizes = [int(x) for x in args.emb_sizes.split(',')]
    logging.info(f"测试维度: {emb_sizes}")
    
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
    
    logging.info(f"用户数: {corpus.n_users}, 物品数: {corpus.n_items}")
    
    # 存储所有结果
    all_results = {
        'stage1': {},
        'stage2': {},
    }
    
    # 对每个维度进行实验
    for emb_size in emb_sizes:
        logging.info(f"\n{'#'*60}")
        logging.info(f"# Embedding 维度 = {emb_size}")
        logging.info(f"{'#'*60}")
        
        # 重置随机种子
        utils.init_seed(args.seed)
        
        # Stage 1: Teacher 预训练
        result_s1 = train_model(
            args, corpus, emb_size, stage=1, save_dir=save_dir
        )
        all_results['stage1'][emb_size] = result_s1
        
        # 重置随机种子
        utils.init_seed(args.seed)
        
        # Stage 2: Student 蒸馏
        result_s2 = train_model(
            args, corpus, emb_size, stage=2, save_dir=save_dir,
            teacher_path=result_s1['model_path']
        )
        all_results['stage2'][emb_size] = result_s2
    
    # 汇总结果
    logging.info("\n" + "="*60)
    logging.info("实验结果汇总")
    logging.info("="*60)
    
    metrics = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
    
    # Stage 1 结果
    logging.info("\n=== Stage 1 (Teacher) ===")
    header = f"{'维度':<10} | " + " | ".join([f"{m:<10}" for m in metrics]) + " | 训练时间"
    logging.info(header)
    logging.info("-" * len(header))
    
    for emb_size in emb_sizes:
        result = all_results['stage1'][emb_size]
        test = result['test']
        row = f"{emb_size:<10} | "
        row += " | ".join([f"{test.get(m, 0):.4f}    " for m in metrics])
        row += f" | {result['training_time']:.1f}s"
        logging.info(row)
    
    # Stage 2 结果
    logging.info("\n=== Stage 2 (Binary Student) ===")
    logging.info(header)
    logging.info("-" * len(header))
    
    for emb_size in emb_sizes:
        result = all_results['stage2'][emb_size]
        test = result['test']
        row = f"{emb_size:<10} | "
        row += " | ".join([f"{test.get(m, 0):.4f}    " for m in metrics])
        row += f" | {result['training_time']:.1f}s"
        logging.info(row)
    
    # Stage 2 相对 Stage 1 的保持率
    logging.info("\n=== Stage 2 / Stage 1 保持率 ===")
    header2 = f"{'维度':<10} | " + " | ".join([f"{m:<10}" for m in metrics])
    logging.info(header2)
    logging.info("-" * len(header2))
    
    for emb_size in emb_sizes:
        s1_test = all_results['stage1'][emb_size]['test']
        s2_test = all_results['stage2'][emb_size]['test']
        row = f"{emb_size:<10} | "
        for m in metrics:
            s1_val = s1_test.get(m, 0)
            s2_val = s2_test.get(m, 0)
            if s1_val > 0:
                ratio = s2_val / s1_val * 100
                row += f"{ratio:.1f}%     "
            else:
                row += "N/A       "
        logging.info(row)
    
    # 维度 vs 参数量
    logging.info("\n=== 维度 vs 模型参数量 ===")
    for emb_size in emb_sizes:
        n_params = all_results['stage1'][emb_size]['n_params']
        logging.info(f"  emb_size={emb_size}: {n_params:,} 参数")
    
    # 找到最佳维度
    best_dim = max(emb_sizes, key=lambda d: all_results['stage2'][d]['test'].get('NDCG@10', 0))
    best_ndcg = all_results['stage2'][best_dim]['test'].get('NDCG@10', 0)
    logging.info(f"\n最佳维度: {best_dim} (NDCG@10 = {best_ndcg:.4f})")
    
    # 保存结果
    save_results = {
        'experiment': '超参数敏感性分析 - Embedding 维度',
        'timestamp': timestamp,
        'config': {
            'dataset': args.dataset,
            'seed': args.seed,
            'emb_sizes': emb_sizes,
            'n_layers': args.n_layers,
            'lambda_id': args.lambda_id,
        },
        'stage1': {},
        'stage2': {},
        'best_dimension': best_dim,
    }
    
    for stage_name in ['stage1', 'stage2']:
        for emb_size in emb_sizes:
            result = all_results[stage_name][emb_size]
            save_results[stage_name][emb_size] = {
                'test': {k: float(v) for k, v in result['test'].items()},
                'dev': {k: float(v) for k, v in result['dev'].items()},
                'training_time': result['training_time'],
                'n_params': result['n_params'],
            }
    
    result_file = os.path.join(result_dir, f'hyperparam_results_{timestamp}.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n结果已保存到: {result_file}")
    logging.info("\n实验四完成！")


if __name__ == '__main__':
    main()

