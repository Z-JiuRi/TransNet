# WAIRD seed797 scenario\_2 torch\_independent 实验分析

生成时间：2026-06-08

## 范围与数据来源

本文档分析当前仓库中
`exps/WAIRD/seed797/scenario_2/` 下目录名以 `torch_independent` 结尾的全部实验结果和相关代码。

共发现 11 个目标实验：

- `01105_torch_independent`
- `01105_encoder_ffn_8_64_torch_independent`
- `01105_decoder_ffn_8_64_torch_independent`
- `01105_fc_encoder_64_128_torch_independent`
- `01105_fc_decoder_64_128_torch_independent`
- `01105_fc_encoder_256_1024_torch_independent`
- `01105_fc_encoder_256_256_torch_independent`
- `06401_encoder_ffn_8_64_torch_independent`
- `06401_decoder_ffn_8_64_torch_independent`
- `09957_encoder_ffn_8_64_torch_independent`
- `09957_decoder_ffn_8_64_torch_independent`

解析依据为每个实验目录下的 `args.json`、`run.log`、`train.out`。所有 11 个实验目录均存在
`checkpoints/best_nmse.pth` 和 `encoder_output.pth`。

## 总体结论

1. `01105_torch_independent` 是唯一的普通全量微调实验，最终 NMSE 为 `-22.003 dB`，显著优于所有 LoRA 实验。
2. LoRA 实验中最优的是 `01105_fc_decoder_64_128_torch_independent`，best NMSE 为 `-20.877 dB`，但仍比 01105 全量微调差约 `1.126 dB`。
3. 对 01105 而言，`fc_decoder` LoRA 明显强于 `decoder_ffn`、`fc_encoder` 和 `encoder_ffn`；`encoder_ffn` 最差。
4. 对 06401 而言，`encoder_ffn` 与 `decoder_ffn` 几乎没有带来有效提升，best NMSE 只比 LoRA 初始化状态好约 `0.003~0.006 dB`。
5. 对 09957 而言，`decoder_ffn` 优于 `encoder_ffn`，但提升幅度仍有限。
6. 所有目标实验都使用 `seed797/base/torch_independent/checkpoints/best_nmse.pth` 作为预训练权重，模型口径一致：`transformer_backend=torch`、`layer_sharing=independent`、`nt=64`、`nc=64`、`cr=4`、`d_model=64`、`dim_feedforward=2048`。

## 实验配置

公共配置如下：

| 字段                   | 值                                                                     |
| -------------------- | --------------------------------------------------------------------- |
| seed                 | `797`                                                                 |
| channel              | `2`                                                                   |
| nt / nc              | `64 / 64`                                                             |
| input\_dim           | `2 * 64 * 64 = 8192`                                                  |
| cr                   | `4`                                                                   |
| codeword length      | `8192 / 4 = 2048`                                                     |
| d\_model             | `64`                                                                  |
| Transformer 序列形状     | `(B, 128, 64)`                                                        |
| dim\_feedforward     | `2048`                                                                |
| transformer\_backend | `torch`                                                               |
| layer\_sharing       | `independent`                                                         |
| scheduler            | `cosine`                                                              |
| epochs               | `200`                                                                 |
| pretrained           | `exps/WAIRD/seed797/base/torch_independent/checkpoints/best_nmse.pth` |

训练数据按实验 ID 区分：

| 数据 ID | train\_path                      | val\_path / test\_path          |
| ----- | -------------------------------- | ------------------------------- |
| 01105 | `data/scenario_2/01105/train.pt` | `data/scenario_2/01105/test.pt` |
| 06401 | `data/scenario_2/06401/train.pt` | `data/scenario_2/06401/test.pt` |
| 09957 | `data/scenario_2/09957/train.pt` | `data/scenario_2/09957/test.pt` |

全量微调与 LoRA 的训练超参不同：

| 类型                        | batch\_size | lr\_init | weight\_decay | LoRA           |
| ------------------------- | ----------: | -------: | ------------: | -------------- |
| `01105_torch_independent` |       `256` |   `2e-4` |        `1e-3` | 无              |
| 所有 LoRA 实验                |        `32` |   `5e-4` |           `0` | 只训练 `lora_` 参数 |

## 结果汇总

NMSE 越低越好。下表按 final NMSE 排序。

| 排名 | 实验                                            | 数据 ID | 训练方式               | rank / alpha |  初始化 NMSE | best NMSE | best epoch |   final loss | final NMSE |
| -: | --------------------------------------------- | ----- | ------------------ | ------------ | --------: | --------: | ---------: | -----------: | ---------: |
|  1 | `01105_torch_independent`                     | 01105 | 全量微调               | -            |         - | `-22.003` |        200 | `3.1523e-03` |  `-22.003` |
|  2 | `01105_fc_decoder_64_128_torch_independent`   | 01105 | LoRA `fc_decoder`  | 64 / 128     | `-18.604` | `-20.877` |        162 | `4.0906e-03` |  `-20.872` |
|  3 | `01105_decoder_ffn_8_64_torch_independent`    | 01105 | LoRA `decoder_ffn` | 8 / 64       | `-18.604` | `-20.177` |        200 | `4.8000e-03` |  `-20.177` |
|  4 | `01105_fc_encoder_256_256_torch_independent`  | 01105 | LoRA `fc_encoder`  | 256 / 256    | `-18.604` | `-20.045` |        164 | `4.9545e-03` |  `-20.040` |
|  5 | `01105_fc_encoder_256_1024_torch_independent` | 01105 | LoRA `fc_encoder`  | 256 / 1024   | `-18.604` | `-20.037` |        200 | `4.9579e-03` |  `-20.037` |
|  6 | `01105_fc_encoder_64_128_torch_independent`   | 01105 | LoRA `fc_encoder`  | 64 / 128     | `-18.604` | `-19.854` |        200 | `5.1708e-03` |  `-19.854` |
|  7 | `01105_encoder_ffn_8_64_torch_independent`    | 01105 | LoRA `encoder_ffn` | 8 / 64       | `-18.604` | `-19.560` |        200 | `5.5330e-03` |  `-19.560` |
|  8 | `09957_decoder_ffn_8_64_torch_independent`    | 09957 | LoRA `decoder_ffn` | 8 / 64       | `-18.669` | `-19.193` |        199 | `6.0210e-03` |  `-19.193` |
|  9 | `09957_encoder_ffn_8_64_torch_independent`    | 09957 | LoRA `encoder_ffn` | 8 / 64       | `-18.669` | `-19.088` |        192 | `6.1679e-03` |  `-19.088` |
| 10 | `06401_decoder_ffn_8_64_torch_independent`    | 06401 | LoRA `decoder_ffn` | 8 / 64       | `-17.823` | `-17.829` |        194 | `8.2438e-03` |  `-17.828` |
| 11 | `06401_encoder_ffn_8_64_torch_independent`    | 06401 | LoRA `encoder_ffn` | 8 / 64       | `-17.823` | `-17.826` |        184 | `8.2484e-03` |  `-17.826` |

说明：

- LoRA 的“初始化 NMSE”来自 `LoRA monitor [before_train]`，此时 `lora_B` 为零，理论上应接近加载 base 后的原模型输出。
- 全量微调没有 LoRA monitor，因此没有初始化 NMSE。
- 目标日志中 final test 是训练结束后重新调用 `Tester` 得到的结果；best NMSE 来自训练过程中每轮或每 10 轮测试后的 checkpoint 选择。

## 按数据 ID 分析

### 01105

01105 的实验最完整，包含 1 个全量微调和 6 个 LoRA 变体。

| 实验                             | best NMSE |   相比 LoRA 初始化 |       相比全量微调 |
| ------------------------------ | --------: | ------------: | -----------: |
| 全量微调 `01105_torch_independent` | `-22.003` |             - |           基线 |
| LoRA `fc_decoder`, r64/a128    | `-20.877` | 提升 `2.273 dB` | 差 `1.126 dB` |
| LoRA `decoder_ffn`, r8/a64     | `-20.177` | 提升 `1.573 dB` | 差 `1.826 dB` |
| LoRA `fc_encoder`, r256/a256   | `-20.045` | 提升 `1.441 dB` | 差 `1.958 dB` |
| LoRA `fc_encoder`, r256/a1024  | `-20.037` | 提升 `1.433 dB` | 差 `1.966 dB` |
| LoRA `fc_encoder`, r64/a128    | `-19.854` | 提升 `1.250 dB` | 差 `2.149 dB` |
| LoRA `encoder_ffn`, r8/a64     | `-19.560` | 提升 `0.956 dB` | 差 `2.443 dB` |

结论：

- 全量微调仍是当前最强配置。
- 只改 `fc_decoder` 的 LoRA 是最有效的参数高效适配方向。
- `fc_encoder` 从 r64/a128 增大到 r256/a256 有收益，但继续把 alpha 从 256 提到 1024 没有带来可见改善。
- `encoder_ffn` LoRA 收益最弱，说明只调 encoder FFN 不足以适配 01105。

### 06401

06401 只有两个 FFN LoRA 变体。

| 实验                                         |  初始化 NMSE | best NMSE |         提升 |
| ------------------------------------------ | --------: | --------: | ---------: |
| `06401_decoder_ffn_8_64_torch_independent` | `-17.823` | `-17.829` | `0.006 dB` |
| `06401_encoder_ffn_8_64_torch_independent` | `-17.823` | `-17.826` | `0.003 dB` |

结论：

- 两个实验几乎没有学习到有效增益。
- 训练 200 epoch 后 final NMSE 与初始化状态基本相同，说明单独 FFN LoRA 对 06401 迁移不敏感，或者该数据 ID 与 base 的差异无法由这两个小 adapter 表达。

### 09957

09957 也只有两个 FFN LoRA 变体。

| 实验                                         |  初始化 NMSE | best NMSE |         提升 |
| ------------------------------------------ | --------: | --------: | ---------: |
| `09957_decoder_ffn_8_64_torch_independent` | `-18.669` | `-19.193` | `0.524 dB` |
| `09957_encoder_ffn_8_64_torch_independent` | `-18.669` | `-19.088` | `0.419 dB` |

结论：

- `decoder_ffn` 比 `encoder_ffn` 更好，但差距只有约 `0.105 dB`。
- 相比 01105，09957 的 LoRA 收益较小；相比 06401，仍有明确改善。

## 代码路径分析

### 入口与训练流程

`main.py` 的流程如下：

1. 创建 `exps/{exp_name}/checkpoints` 和 `tensorboard`。
2. 将 CLI 参数保存为 `args.json`。
3. `MyDataLoader(...)` 读取 `train_path`、`val_path`、`test_path`。
4. `init_model(args)` 构建模型、加载预训练权重、应用 LoRA 或冻结配置。
5. 使用 `nn.MSELoss()` 和 `AdamW` 训练。
6. `Trainer.loop(...)` 训练 200 epoch，并根据 test NMSE 保存 `best_nmse.pth`。
7. 训练结束后保存 `encoder_output.pth`，再做 final test。

### 数据加载

`dataloader/dataloader.py` 中 `MyDataLoader` 直接用 `torch.load(..., map_location='cpu')` 读取 `.pt` 文件，再封装为单元素 `TensorDataset`。

当前代码不会在 dataloader 内显式 reshape 或校验 `(channel, nt, nc)`，目标实验依赖数据文件本身已经是模型期望的 `(N, 2, 64, 64)` 形状。

### torch\_independent 模型

`models/TransNet.py` 中：

- `transformer_backend='torch'` 时，encoder/decoder layer 使用 PyTorch 原生 `nn.TransformerEncoderLayer` 和 `nn.TransformerDecoderLayer`。
- `layer_sharing='independent'` 时，encoder 使用 `nn.TransformerEncoder`，decoder 使用 `nn.TransformerDecoder`，两层 Transformer 不共享参数。
- 对 64x64 输入，`input_dim=8192`，`feature_shape=(128, 64)`。
- 前向传播为：
  - 输入 `src`: `(B, 2, 64, 64)`
  - `view`: `(B, 128, 64)`
  - encoder 输出: `(B, 128, 64)`
  - `fc_encoder`: `(B, 8192) -> (B, 2048)`
  - `fc_decoder`: `(B, 2048) -> (B, 8192)`
  - decoder 输出 reshape 回 `(B, 2, 64, 64)`

### LoRA 实现

`utils/init.py` 中 `LoRALinear` 将一个 `nn.Linear` 替换为：

```text
base_layer(x) + lora_B(lora_A(x)) * (alpha / rank)
```

其中 base layer 参数被冻结，`lora_A` 使用 Kaiming 初始化，`lora_B` 初始化为 0。

`lora_component(...)` 的关键行为：

- 如果启用 `--lora_component`，先冻结所有模型参数。
- 按组件替换目标线性层：
  - `encoder_ffn`: `encoder.layers.{0,1}.linear1/linear2`
  - `decoder_ffn`: `decoder.layers.{0,1}.linear1/linear2`
  - `fc_encoder`: `fc_encoder`
  - `fc_decoder`: `fc_decoder`
- 最后只允许名字中包含 `lora_` 的参数训练。

可训练标量参数量如下：

| LoRA 组件       | rank / alpha           | target modules         |        trainable / total |
| ------------- | ---------------------- | ---------------------- | -----------------------: |
| `encoder_ffn` | 8 / 64                 | 4 个 encoder FFN Linear |    `67,584 / 34,790,656` |
| `decoder_ffn` | 8 / 64                 | 4 个 decoder FFN Linear |    `67,584 / 34,790,656` |
| `fc_encoder`  | 64 / 128               | 1 个压缩 Linear           |   `655,360 / 35,378,432` |
| `fc_decoder`  | 64 / 128               | 1 个解压 Linear           |   `655,360 / 35,378,432` |
| `fc_encoder`  | 256 / 256 或 256 / 1024 | 1 个压缩 Linear           | `2,621,440 / 37,344,512` |

### 指标计算

`utils/solver.py` 的 `Tester` 使用 `evaluator_ratio(...)` 先按样本计算归一化误差比例，再在整个测试集上累计 ratio，最终输出：

```text
10 * log10(sum(sample_nmse_ratio) / sample_count)
```

`utils/statics.py` 当前没有对 `sparse_gt` 和 `sparse_pred` 做 `-0.5` 去中心化；相关代码已被注释。因此本文档中的 NMSE 是基于原始 sparse tensor 数值计算的结果。

## 可能原因与建议

### 为什么全量微调明显更强

全量微调会更新全部约 3,400 万参数，而 LoRA 只更新 6.8 万到 262 万 adapter 参数。WAIRD scenario\_2 的不同 ID 之间可能涉及较强分布迁移，只调单个组件时容量不足，尤其是只调 Transformer FFN 时，无法直接重塑压缩码字与重建空间之间的映射。

### 为什么 fc\_decoder LoRA 最值得继续

01105 上 `fc_decoder` r64/a128 明显优于所有其它 LoRA 配置。TransNet 的压缩码字到重建矩阵主要由 `fc_decoder` 完成，目标场景适配很可能更依赖解码侧重建映射，而不是只调 encoder FFN 或 fc\_encoder。

### 后续实验建议

1. 对 06401、09957 补跑 `fc_decoder_64_128_torch_independent` 和 `fc_encoder_256_256_torch_independent`，确认 01105 上的规律是否可迁移。
2. 在 01105 上尝试组合 LoRA，例如同时启用 `fc_encoder fc_decoder`，或 `decoder_ffn fc_decoder`。
3. 为 LoRA 增加 epoch 0 的普通 test 记录，方便明确 base 初始化性能与 `LoRA monitor [before_train]` 是否完全一致。
4. 对 LoRA 与全量微调使用相同 `batch_size` 复评 best checkpoint，排除 batch size 与训练后 final test 口径差异。
5. 如果目标是逼近全量微调，优先扩展解码侧 LoRA，而不是继续单独增大 `fc_encoder` alpha。

## 限制

- 本文档只基于当前仓库已有日志和配置，没有重新训练或重新评估 checkpoint。
- 当前数据文件和 checkpoint 是否可在其它机器复现，取决于 `data/` 与 `exps/` 下权重文件是否完整。
- 06401 和 09957 缺少全量微调、`fc_encoder`、`fc_decoder` LoRA 对照，因此跨数据 ID 的结论应视为当前材料下的阶段性判断。

