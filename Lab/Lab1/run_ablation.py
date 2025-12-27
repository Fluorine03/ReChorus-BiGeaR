# -*- coding: UTF-8 -*-
"""
实验一：消融实验 - 知识蒸馏模块有效性验证

目标：对比有/无蒸馏模块时 BiGeaR Stage 2 的性能差异

实验设计：
    1. 使用已有的 Stage 1 Teacher 模型
    2. 训练完整版 Stage 2（use_distillation=1）
    3. 训练无蒸馏版 Stage 2（use_distillation=0）
    4. 对比两者的性能指标
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
from helpers.BaseReader import BaseReader
from utils import utils


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description='消融实验：蒸馏模块有效性验证')
    
    # 基础参数
    parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food')
    parser.add_argument('--data_path', type=str, default='../../data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    
    # 模型参数
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--lambda_id', type=float, default=1.0)
    parser.add_argument('--top_r', type=int, default=100)
    parser.add_argument('--lambda_decay', type=float, default=0.1)
    parser.add_argument('--norm_a', type=float, default=1.0)
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=10)
    
    # Teacher 模型路径
    parser.add_argument('--teacher_model_path', type=str, 
                        default='../../record/BiGeaR_LightGCN_v2/controlled_stage1_models/BiGeaR_LightGCN_v2__Grocery_and_Gourmet_Food__seed0__lr=0.001__l2=1e-06__emb_size=64__n_layers=2__stage=1.pt')
    
    return parser.parse_args()


def train_stage2(args, corpus, use_distillation: bool, save_dir: str):
    """
    训练 Stage 2 模型
    
    Args:
        args: 命令行参数
        corpus: 数据集对象
        use_distillation: 是否使用蒸馏
        save_dir: 模型保存目录
    
    Returns:
        测试集评估结果
    """
    distill_str = "with_distill" if use_distillation else "no_distill"
    logging.info(f"\n{'='*60}")
    logging.info(f"开始训练 Stage 2 ({distill_str})")
    logging.info(f"{'='*60}")
    
    # 设置模型参数
    model_args = argparse.Namespace(
        # 模型参数
        emb_size=args.emb_size,
        n_layers=args.n_layers,
        stage=2,  # Stage 2
        norm_a=args.norm_a,
        lambda_id=args.lambda_id,
        top_r=args.top_r,
        lambda_decay=args.lambda_decay,
        teacher_model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), args.teacher_model_path)),
        use_distillation=1 if use_distillation else 0,  # 消融开关
        
        # 训练参数
        lr=args.lr,
        l2=args.l2,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        epoch=args.epoch,
        early_stop=args.early_stop,
        
        # GeneralModel 需要的参数
        num_neg=1,
        dropout=0,
        test_all=0,
        
        # 其他参数
        train=1,
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
        model_path=os.path.join(save_dir, f'BiGeaR_Stage2_{distill_str}.pt'),
        log_file=os.path.join(save_dir, f'BiGeaR_Stage2_{distill_str}.log'),
    )
    model_args.device = torch.device('cuda' if args.gpu != '' and torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = BiGeaR_LightGCN(model_args, corpus).to(model_args.device)
    logging.info(f"模型参数量: {model.count_variables()}")
    logging.info(f"use_distillation = {use_distillation}")
    
    # 创建数据集
    data_dict = {}
    for phase in ['train', 'dev', 'test']:
        data_dict[phase] = BiGeaR_LightGCN.Dataset(model, corpus, phase)
        data_dict[phase].prepare()
    
    # 创建 Runner
    runner = BaseRunner(model_args)
    
    # 训练前测试
    logging.info(f"训练前测试: {runner.print_res(data_dict['test'])}")
    
    # 训练
    start_time = time.time()
    runner.train(data_dict)
    training_time = time.time() - start_time
    
    # 加载最佳模型
    model.load_model()
    
    # 最终评估
    dev_result = runner.evaluate(data_dict['dev'], runner.topk, runner.metrics)
    test_result = runner.evaluate(data_dict['test'], runner.topk, runner.metrics)
    
    logging.info(f"\n训练完成 ({distill_str})")
    logging.info(f"训练时间: {training_time:.2f}s")
    logging.info(f"Dev 结果: {utils.format_metric(dev_result)}")
    logging.info(f"Test 结果: {utils.format_metric(test_result)}")
    
    return {
        'use_distillation': use_distillation,
        'dev': dev_result,
        'test': test_result,
        'training_time': training_time,
        'model_path': model_args.model_path,
    }


def main():
    args = setup_args()
    
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, 'models')
    log_dir = os.path.join(base_dir, 'logs')
    result_dir = os.path.join(base_dir, 'results')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'ablation_experiment_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*60)
    logging.info("实验一：消融实验 - 知识蒸馏模块有效性验证")
    logging.info("="*60)
    logging.info(f"数据集: {args.dataset}")
    logging.info(f"随机种子: {args.seed}")
    logging.info(f"Embedding 维度: {args.emb_size}")
    logging.info(f"GNN 层数: {args.n_layers}")
    logging.info(f"Lambda_ID: {args.lambda_id}")
    
    # 设置随机种子
    utils.init_seed(args.seed)
    
    # 设置 GPU
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
    
    # 检查 Teacher 模型
    teacher_path = os.path.abspath(os.path.join(base_dir, args.teacher_model_path))
    if not os.path.exists(teacher_path):
        logging.error(f"Teacher 模型不存在: {teacher_path}")
        return
    logging.info(f"Teacher 模型: {teacher_path}")
    
    # 实验结果
    results = {}
    
    # 1. 训练有蒸馏版本
    logging.info("\n" + "="*60)
    logging.info("Part 1: 训练有蒸馏版本 (use_distillation=True)")
    logging.info("="*60)
    result_with_distill = train_stage2(args, corpus, use_distillation=True, save_dir=save_dir)
    results['with_distillation'] = result_with_distill
    
    # 重新设置随机种子确保公平对比
    utils.init_seed(args.seed)
    
    # 2. 训练无蒸馏版本
    logging.info("\n" + "="*60)
    logging.info("Part 2: 训练无蒸馏版本 (use_distillation=False)")
    logging.info("="*60)
    result_no_distill = train_stage2(args, corpus, use_distillation=False, save_dir=save_dir)
    results['without_distillation'] = result_no_distill
    
    # 3. 结果对比
    logging.info("\n" + "="*60)
    logging.info("实验结果对比")
    logging.info("="*60)
    
    metrics = ['HR@10', 'HR@20', 'NDCG@10', 'NDCG@20']
    
    logging.info("\n测试集结果:")
    logging.info(f"{'模型':<25} | " + " | ".join([f"{m:<10}" for m in metrics]))
    logging.info("-" * 80)
    
    with_test = result_with_distill['test']
    no_test = result_no_distill['test']
    
    logging.info(f"{'BiGeaR (有蒸馏)':<25} | " + " | ".join([f"{with_test.get(m, 0):.4f}    " for m in metrics]))
    logging.info(f"{'BiGeaR (无蒸馏)':<25} | " + " | ".join([f"{no_test.get(m, 0):.4f}    " for m in metrics]))
    
    # 计算相对变化
    logging.info("\n相对变化 (无蒸馏 vs 有蒸馏):")
    for m in metrics:
        with_val = with_test.get(m, 0)
        no_val = no_test.get(m, 0)
        if with_val > 0:
            change = (no_val - with_val) / with_val * 100
            logging.info(f"  {m}: {change:+.2f}%")
    
    # 保存结果
    result_file = os.path.join(result_dir, f'ablation_results_{timestamp}.json')
    
    # 转换为可序列化格式
    save_results = {
        'experiment': '消融实验 - 知识蒸馏模块有效性',
        'timestamp': timestamp,
        'config': {
            'dataset': args.dataset,
            'seed': args.seed,
            'emb_size': args.emb_size,
            'n_layers': args.n_layers,
            'lambda_id': args.lambda_id,
        },
        'with_distillation': {
            'test': {k: float(v) for k, v in result_with_distill['test'].items()},
            'dev': {k: float(v) for k, v in result_with_distill['dev'].items()},
            'training_time': result_with_distill['training_time'],
        },
        'without_distillation': {
            'test': {k: float(v) for k, v in result_no_distill['test'].items()},
            'dev': {k: float(v) for k, v in result_no_distill['dev'].items()},
            'training_time': result_no_distill['training_time'],
        },
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n结果已保存到: {result_file}")
    logging.info("\n实验一完成！")


if __name__ == '__main__':
    main()

