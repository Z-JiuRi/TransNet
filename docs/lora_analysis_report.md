# TransNet LoRA 实验效果分析报告

## 一、实验概况

实验在 COST2100 CSI 反馈压缩任务上测试 LoRA (Low-Rank Adaptation) 微调效果。数据集包含室内 (indoor) 和室外 (outdoor) 两种场景，模型为 TransNet——基于 Transformer 的自编码器，压缩比 CR=4。

### 实验结构

| 实验组 | 场景 | Batch Size | 种子 | 是否 LoRA | 预训练 |
|--------|------|------------|------|-----------|--------|
| COST2100_bs32/base | in/out | 32 | 42, 2026, 3407 | 否 | 无 |
| COST2100_bs512/base | in/out | 512 | 42, 2026, 3407 | 否 | 无 |
| COST2100_lora | in only | 32 | 0 | 是 (fc_encoder+fc_decoder) | bs512/in/seed42/best_nmse |

LoRA 实验共 12 组，参数组合: rank ∈ {8, 16}, alpha ∈ {8, 16, 32}, lr ∈ {5e-5, 1e-4, 2e-4, 3e-4}。

---

## 二、核心数据对比

### 2.1 Base 模型 (从头训练, 无 LoRA, 无预训练)

| 场景 | Batch Size | Seed 42 | Seed 2026 | Seed 3407 | 平均 NMSE |
|------|-----------|---------|-----------|-----------|-----------|
| Indoor | 32 | **-34.37 dB** | -33.85 dB | -33.74 dB | **-33.99 dB** |
| Indoor | 512 | -23.91 dB | -22.64 dB | -22.89 dB | -23.15 dB |
| Outdoor | 32 | -18.23 dB | **-18.58 dB** | -18.53 dB | **-18.44 dB** |
| Outdoor | 512 | -13.39 dB | -12.96 dB | -13.32 dB | -13.22 dB |

### 2.2 LoRA 微调模型 (预训练自 bs512/in/seed42, NMSE=-23.91 dB)

| Rank | Alpha | LR | 最佳 NMSE | vs 预训练源 | vs 从头训练 bs32 |
|------|-------|-----|-----------|-------------|-------------------|
| 8 | 8 | 1e-4 | -24.23 dB | +0.32 | -10.14 (差) |
| 8 | 8 | 2e-4 | -24.26 dB | +0.35 | -10.11 (差) |
| 8 | 16 | 5e-5 | -24.24 dB | +0.33 | -10.13 (差) |
| 8 | 16 | 1e-4 | -24.26 dB | +0.35 | -10.11 (差) |
| 8 | 16 | 2e-4 | -24.28 dB | +0.37 | -10.09 (差) |
| 8 | 16 | 3e-4 | -24.29 dB | +0.38 | -10.08 (差) |
| 16 | 16 | 1e-4 | -24.28 dB | +0.37 | -10.09 (差) |
| 16 | 16 | 2e-4 | -24.33 dB | +0.41 | -10.04 (差) |
| 16 | 32 | 5e-5 | -24.29 dB | +0.38 | -10.08 (差) |
| 16 | 32 | 1e-4 | -24.32 dB | +0.41 | -10.05 (差) |
| 16 | 32 | 2e-4 | -24.35 dB | +0.44 | -10.02 (差) |
| **16** | **32** | **3e-4** | **-24.36 dB** | **+0.45** | **-10.01 (差)** |

---

## 三、结论：LoRA 几乎无效

两项关键对比：

```
LoRA 最佳:    -24.36 dB  (预训练 bs512 模型 + LoRA 微调)
预训练源:     -23.91 dB  (bs512/in/seed42 从头训练)  
从头训练 bs32: -34.37 dB  (同 batch size，从头训练)
```

1. **对比预训练源**: LoRA 仅提升 **0.45 dB**——边际改善，远低于预期。
2. **对比同条件从头训练**: LoRA 落后 **~10 dB**——差距巨大，说明 LoRA 远不如从头训练。

两种 batch size 都不适合 LoRA：bs512 模型性能差 (-23.91)，LoRA 微调只能勉强修复到 -24.36；而 bs32 从头训练直接达到 -34.37，说明 LoRA 策略完全失败。

---

## 四、根因分析

### 4.1 模型架构与 LoRA 作用位置

TransNet 的结构是：

```
输入 CSI (2×32×32)
  │
  ▼
Transformer Encoder (2 layers)    ← 特征提取 (注意力 + FFN)
  self-attention + FFN              参数 ~556K
  │
  ▼
fc_encoder: Linear(2048→512)      ← 压缩瓶颈 (纯线性)
  参数 ~1.05M                        ★ LoRA 作用于此
  │
  ▼
fc_decoder: Linear(512→2048)      ← 解压瓶颈 (纯线性)
  参数 ~1.05M                        ★ LoRA 作用于此
  │
  ▼
Transformer Decoder (2 layers)    ← 重建 (自注意力 + 交叉注意力 + FFN)
  self-attn + cross-attn + FFN      参数 ~588K
  │
  ▼
输出 CSI (2×32×32)
```

**关键事实**: fc_encoder 和 fc_decoder 虽然占模型参数的 \~65%（\~2.1M / 3.2M），但它们只是**线性投影层**。真正负责学习和提取 CSI 特征的是 Transformer Encoder 的注意力机制，负责重建的是 Transformer Decoder。

### 4.2 根因一：冻结策略本末倒置

LoRA 冻结了 Transformer Encoder + Decoder（非线性、高表示能力的部分），只微调 fc_encoder + fc_decoder（线性、低表示能力的部分）。

- **Transformer Encoder** (被冻结): 负责理解 CSI 的空间/延迟结构。如果这里学得差，后续全部白费。
- **fc_encoder/fc_decoder** (被微调): 只做线性压缩/解压。不论怎么调，都补不回来 Encoder 输出的信息损失。

这就像：你用一张模糊的照片（差的 Transformer 编码），然后试图在显影/放大阶段（fc_encoder/fc_decoder）恢复清晰度——线性变换做不到这件事。

### 4.3 根因二：预训练模型质量太差

bs512 训练的 Transformer 权重只达到 -23.91 dB NMSE，与最优的 -34.37 dB 相差超过 10 dB。原因是大 batch 导致每 epoch 只有 196 步梯度更新（bs32 有 938 步），模型收敛到次优的局部极小点。

LoRA 面临的问题是：**下游参数再优化，也弥补不了上游的 10 dB 信息损失**。

### 4.4 根因三：线性层表示能力不足

即使 LoRA rank 从 8 提到 16，alpha 从 8 提到 32，lr 从 5e-5 提到 3e-4，12 组实验的最佳/最差差距仅 **0.13 dB**（-24.36 到 -24.23）。这说明：

- 所有 LoRA 变体在狭窄范围内波动，没有一个能突破瓶颈
- 瓶颈不在 LoRA 超参数，而在冻结策略本身
- fc_encoder (2048×512) 上的 rank-16 更新最多只能在 16 维子空间内修正权重——远不足以纠正预训练模型的系统性问题

### 4.5 根因四：LoRA 作用位置的梯度传播受阻

反向传播路径：

```
Loss → Output → Decoder(冻结) → fc_decoder(LoRA) → fc_encoder(LoRA) → Encoder(冻结) → Input
```

梯度从 Loss 出发，穿过了**整个冻结的 Decoder** 才到达 fc_decoder 的 LoRA 参数。冻结的 Decoder 权重会扭曲梯度信号——LoRA 参数收到的梯度经过了 2 层冻结的 self-attention + cross-attention + FFN，这是一个被固定的、不可学习的非线性变换。

---

## 五、改进建议

### 5.1 将 LoRA 应用到 Transformer 层 (推荐)

对 Transformer Encoder/Decoder 的 self-attention、cross-attention、FFN 层的 Q/K/V 投影矩阵和 FFN 线性层施加 LoRA，冻结 fc_encoder/fc_decoder：

```
fc_encoder, fc_decoder → 冻结 (保持压缩比语义)
encoder attention/FFN → LoRA 微调 (调整特征提取)
decoder attention/FFN → LoRA 微调 (调整重建质量)
```

这样 LoRA 作用于模型的**非线性核心**（注意力机制），能真正改变模型对 CSI 的编码策略。

### 5.2 使用高质量预训练模型

当前使用 bs512 模型 (-23.91 dB) 作为预训练源。更好的选择：
- 使用 bs32 模型 (-34.37 dB) 作为预训练源，然后 LoRA 微调做**域适应**（如 indoor → outdoor 迁移）
- 如果目标是用 LoRA 提升已有模型，起始点不能太差

### 5.3 重新评估 LoRA 的适用场景

LoRA 在这个任务上的合理用途可能是：
- **跨场景迁移**: indoor 预训练 → LoRA → outdoor 微调
- **跨压缩比迁移**: CR=4 预训练 → LoRA → CR=8 微调
- **少量域数据的快速适应**: 用少量目标域数据微调已有模型

从头训练性能已经很好 (-34 dB NMSE) 的情况下，LoRA 不适合用于同场景的性能提升。

### 5.4 增补实验建议

| 实验 | 目的 |
|------|------|
| LoRA on encoder/decoder attention, freeze fc_layers | 验证根因一 |
| LoRA on all linear layers | 确定最大 LoRA 效果上界 |
| Pretrain from bs32 seed42, LoRA fine-tune on outdoor | 验证域适应场景 |
| bs512 模型 + full fine-tune (不冻结) bs32 | 对比 LoRA vs 全参数微调 |

---

## 六、总结

LoRA 在 TransNet COST2100 实验上几乎无效，不是因为 LoRA 作为技术本身有问题，而是因为应用策略存在三个致命缺陷：

1. **冻错了位置** —— 冻结了需要调整的 Transformer，微调了无需改动的线性瓶颈层
2. **起点太差** —— 预训练模型本身收敛不充分，线性层无法弥补 10 dB 的差距
3. **任务不匹配** —— 从头训练已经能达到 -34 dB，不需要 LoRA 来做同域微调

最可能有效的改进是：将 LoRA 施加到 Transformer 的注意力和 FFN 层，冻结 fc_encoder/fc_decoder，并用高质量的 bs32 预训练模型做跨域迁移实验。
