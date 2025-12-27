# 实验结果文件说明

本文件夹包含 BiGeaR 算法复现的核心实验结果，所有实验均在相同条件下进行（控制变量）。

## 实验配置

| 参数 | 值 |
|------|-----|
| 数据集 | Grocery_and_Gourmet_Food |
| 嵌入维度 | 64 |
| 图卷积层数 | 2 |
| 学习率 | 0.001 |
| L2 正则化 | 1e-6 |
| 随机种子 | 0 |

## 文件列表

### 1. BiGeaR_Stage1_training.log

**BiGeaR 算法阶段1（Teacher 预训练）训练日志**

- 训练方式：标准 LightGCN + BPR 损失 + 层级加权聚合
- 最终结果：
  - Dev NDCG@10: 0.3712
  - **Test NDCG@10: 0.3114**

### 2. BiGeaR_Stage2_training.log

**BiGeaR 算法阶段2（Binary Student 蒸馏）训练日志**

- 训练方式：二值化嵌入 + BPR 损失 + ID 蒸馏损失
- 关键参数：lambda_id=1.0, top_r=100
- 最终结果：
  - Dev NDCG@10: 0.3608
  - **Test NDCG@10: 0.3005**
- **性能保持率：96.5%**（达到论文声称的 95% 标准）

### 3. LightGCN_baseline_training.log

**LightGCN 基线模型训练日志**

- 标准 LightGCN 实现（mean 聚合）
- 最终结果：
  - Dev NDCG@10: 0.3715
  - **Test NDCG@10: 0.3101**

### 4. BPRMF_baseline_training.log

**BPR-MF 基线模型训练日志**

- 经典矩阵分解 + BPR 损失
- 最终结果：
  - Dev NDCG@10: 0.3333
  - **Test NDCG@10: 0.2737**

## 结果汇总

| 模型 | Test NDCG@5 | Test NDCG@10 | Test NDCG@20 |
|------|-------------|--------------|--------------|
| BPRMF (基线) | 0.2380 | 0.2737 | 0.3012 |
| LightGCN (基线) | 0.2702 | 0.3101 | 0.3382 |
| **BiGeaR Stage 1** (Teacher) | **0.2730** | **0.3114** | **0.3404** |
| BiGeaR Stage 2 (Binary) | 0.2601 | 0.3005 | 0.3289 |

## 核心结论

✅ **复现成功**：二值化 Student 达到 Teacher 性能的 **96.5%**，符合原论文 95% 的核心声明，同时存储空间减少 32 倍。
