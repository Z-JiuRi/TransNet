# WAIRD seed796/seed797 LoRA 结果分析

生成时间：2026-06-07

## 结论摘要

基于当前仓库已有的 `args.json` 和日志，`seed797/test` 确实是把没有在
`scenario_2/01109` 上微调过的 base checkpoint 直接拿到 01109 测试集上评估：

- 路径：`exps/WAIRD/seed797/test`
- checkpoint：
  `/storage/hujiacong/zxd/Huawei/TransNet/exps/WAIRD/seed797/base/torch_shared/checkpoints/best_nmse.pth`
- 测试数据：
  `/storage/hujiacong/zxd/datasets/WAIRD/data/scenario_2/01109/test.pt`
- 日志结果：`loss=5.6805e-03`，`NMSE=-21.000 dB`

`seed797` 下 12 个 LoRA 实验全部比这个 direct base 测试结果差。最佳 LoRA 是
`decoder_ffn + original_independent`，best/final NMSE 为 `-19.790 dB`，仍比 direct
base 差约 `1.210 dB`。如果只看同为 `torch_shared` 的公平架构口径，三个 LoRA
分支分别是：

| 实验 | LoRA 组件 | best/final NMSE | 相比 direct base |
| --- | --- | ---: | ---: |
| `seed797/test` | 无，直接 base | `-21.000` | 基线 |
| `01109_decoder_ffn_8_16_torch_shared` | `decoder_ffn` | `-19.714` | 差 `1.286 dB` |
| `01109_fc_encoder_8_16_torch_shared` | `fc_encoder` | `-19.709` | 差 `1.291 dB` |
| `01109_encoder_ffn_8_16_torch_shared` | `encoder_ffn` | `-19.702` | 差 `1.298 dB` |

所以，对 `seed797` 可以明确回答：是的，当前 LoRA 结果连 direct base on 01109
都比不过。

对 `seed796` 不能做同样强结论，因为当前没有 `seed796/test` 这种 direct base on
01109 的日志，而且 `seed796` 的 base/LoRA 配置是 `nt=32,nc=32`，`seed797` 是
`nt=64,nc=64`。两个 seed 的实验口径不一致。

## 当前材料限制

我只基于当前可见材料分析：

- 当前仓库里 `exps/WAIRD/seed796` 和 `exps/WAIRD/seed797` 没有 `.pth` 权重文件。
- 日志里的绝对路径 `/storage/hujiacong/zxd/...` 在当前机器不可访问。
- 因此不能复跑评估、不能检查真实数据 tensor shape、不能读取 checkpoint 内容。

解析使用的 Python 环境是：

```text
/home/z-jiuri/.envs/miniconda3/envs/torch/bin/python
Python 3.11.15
```

## 实验口径

### base 训练结果不是 01109 direct test

`seed796/base/*` 和 `seed797/base/*` 都是在
`/storage/hujiacong/zxd/datasets/WAIRD/data/base/{train,test}.pt` 上训练和测试，不是
01109。

| 实验 | 维度 | final NMSE |
| --- | ---: | ---: |
| `seed796/base/original_independent` | `32x32` | `-21.428` |
| `seed796/base/original_shared` | `32x32` | `-21.446` |
| `seed796/base/torch_independent` | `32x32` | `-21.399` |
| `seed796/base/torch_shared` | `32x32` | `-21.465` |
| `seed797/base/original_independent` | `64x64` | `-22.533` |
| `seed797/base/original_shared` | `64x64` | `-22.530` |
| `seed797/base/torch_independent` | `64x64` | `-22.506` |
| `seed797/base/torch_shared` | `64x64` | `-22.516` |

这些结果只能说明 base 数据集上的训练表现，不能直接当作 01109 适配效果。

### direct base on 01109

当前只有一个 direct base on 01109 实验：

| 实验 | checkpoint | 维度 | batch size | NMSE |
| --- | --- | ---: | ---: | ---: |
| `seed797/test` | `seed797/base/torch_shared/best_nmse.pth` | `64x64` | `32` | `-21.000` |

注意：这个实验是 `evaluate=True`，日志中参数 `requires_grad=True` 只是模型参数标志，
没有优化器训练步骤。

## LoRA 结果汇总

### seed796

`seed796` 的 LoRA 全部配置为 `nt=32,nc=32`，路径是 01109：

| 实验 | 组件 | backend | sharing | best/final NMSE | best epoch |
| --- | --- | --- | --- | ---: | ---: |
| `01109_decoder_ffn_8_16_original_independent` | `decoder_ffn` | original | independent | `-19.292` | 400 |
| `01109_decoder_ffn_8_16_original_shared` | `decoder_ffn` | original | shared | `-19.172` | 400 |
| `01109_decoder_ffn_8_16_torch_independent` | `decoder_ffn` | torch | independent | `-19.287` | 400 |
| `01109_decoder_ffn_8_16_torch_shared` | `decoder_ffn` | torch | shared | `-19.181` | 400 |
| `01109_encoder_ffn_8_16_original_independent` | `encoder_ffn` | original | independent | `-19.280` | 390 |
| `01109_encoder_ffn_8_16_original_shared` | `encoder_ffn` | original | shared | `-19.167` | 390 |
| `01109_encoder_ffn_8_16_torch_independent` | `encoder_ffn` | torch | independent | `-19.251` | 10 |
| `01109_encoder_ffn_8_16_torch_shared` | `encoder_ffn` | torch | shared | `-19.149` | 10 |
| `01109_fc_encoder_8_16_original_independent` | `fc_encoder` | original | independent | `-19.266` | 380 |
| `01109_fc_encoder_8_16_original_shared` | `fc_encoder` | original | shared | `-19.161` | 390 |
| `01109_fc_encoder_8_16_torch_independent` | `fc_encoder` | torch | independent | `-19.261` | 400 |
| `01109_fc_encoder_8_16_torch_shared` | `fc_encoder` | torch | shared | `-19.170` | 390 |

按组件聚合：

| 组件 | best NMSE 范围 | 平均 best NMSE |
| --- | ---: | ---: |
| `decoder_ffn` | `-19.292` 到 `-19.172` | `-19.233` |
| `encoder_ffn` | `-19.280` 到 `-19.149` | `-19.212` |
| `fc_encoder` | `-19.266` 到 `-19.161` | `-19.215` |

### seed797

`seed797` 的 LoRA 全部配置为 `nt=64,nc=64`，路径是 01109：

| 实验 | 组件 | backend | sharing | best/final NMSE | best epoch |
| --- | --- | --- | --- | ---: | ---: |
| `01109_decoder_ffn_8_16_original_independent` | `decoder_ffn` | original | independent | `-19.790` | 400 |
| `01109_decoder_ffn_8_16_original_shared` | `decoder_ffn` | original | shared | `-19.725` | 390 |
| `01109_decoder_ffn_8_16_torch_independent` | `decoder_ffn` | torch | independent | `-19.754` | 400 |
| `01109_decoder_ffn_8_16_torch_shared` | `decoder_ffn` | torch | shared | `-19.714` | 390 |
| `01109_encoder_ffn_8_16_original_independent` | `encoder_ffn` | original | independent | `-19.781` | 400 |
| `01109_encoder_ffn_8_16_original_shared` | `encoder_ffn` | original | shared | `-19.722` | 400 |
| `01109_encoder_ffn_8_16_torch_independent` | `encoder_ffn` | torch | independent | `-19.735` | 10 |
| `01109_encoder_ffn_8_16_torch_shared` | `encoder_ffn` | torch | shared | `-19.702` | 10 |
| `01109_fc_encoder_8_16_original_independent` | `fc_encoder` | original | independent | `-19.775` | 350 |
| `01109_fc_encoder_8_16_original_shared` | `fc_encoder` | original | shared | `-19.721` | 360 |
| `01109_fc_encoder_8_16_torch_independent` | `fc_encoder` | torch | independent | `-19.741` | 310 |
| `01109_fc_encoder_8_16_torch_shared` | `fc_encoder` | torch | shared | `-19.709` | 200 |

按组件聚合：

| 组件 | best NMSE 范围 | 平均 best NMSE |
| --- | ---: | ---: |
| `decoder_ffn` | `-19.790` 到 `-19.714` | `-19.746` |
| `encoder_ffn` | `-19.781` 到 `-19.702` | `-19.735` |
| `fc_encoder` | `-19.775` 到 `-19.709` | `-19.737` |

## 为什么 LoRA 效果这么差

### 1. seed797 的 LoRA 训练明显把 direct base 拉坏了

`seed797/test` 显示 direct base 在 01109 上已有 `-21.000 dB`。LoRA 日志第一次测试在
epoch 10，已经只有约 `-19.70~-19.78 dB`，后续 400 epoch 基本只小幅波动：

- `encoder_ffn + torch_shared`：epoch 10 best 就停在 `-19.702 dB`
- `encoder_ffn + torch_independent`：epoch 10 best 就停在 `-19.735 dB`
- `decoder_ffn` 和 `fc_encoder` 分支后续有小幅改善，但最终仍只有约 `-19.71~-19.79 dB`

由于 LoRA 初始化时通常应接近原 base 输出，缺少 epoch 0 评估会遮住“什么时候坏掉”的精确位置。但当前日志已经能说明：前 10 个 epoch 的 LoRA 更新就足以让 NMSE 从 direct base 的 `-21.000 dB` 掉到约 `-19.7 dB`。

### 2. 当前 LoRA 容量太小，且只改单个组件

当前实现中，只要启用 `--lora_component`，代码会先冻结所有 base 参数，然后只训练名字里带
`lora_` 的参数：

```python
for param in model.parameters():
    param.requires_grad = False
...
for name, param in model.named_parameters():
    param.requires_grad = "lora_" in name
```

日志里的 `4/42 params trainable` 是“参数张量个数”，不是标量参数个数。按当前模型结构估算，
`seed797` 的 64x64 模型大约 3414 万到 3472 万 base 参数，而单组件 LoRA 只有：

| 维度 | sharing | 组件 | 可训练标量参数 | 占 LoRA 后总参数比例 |
| --- | --- | --- | ---: | ---: |
| `64x64` | shared | `encoder_ffn` / `decoder_ffn` | 33,792 | 0.099% |
| `64x64` | independent | `encoder_ffn` / `decoder_ffn` | 67,584 | 0.194% |
| `64x64` | shared | `fc_encoder` | 81,920 | 0.239% |
| `64x64` | independent | `fc_encoder` | 81,920 | 0.235% |

TransNet 的主要容量在 `fc_encoder` 和 `fc_decoder` 这两个大线性层上。当前实验要么只改
Transformer FFN，要么只改 `fc_encoder`，没有同时适配 `fc_decoder`，也没有适配注意力 QKV
或 LayerNorm。对于 01109 这种场景迁移，单组件 rank=8 adapter 很可能表达能力不足。

### 3. 优化目标是 MSE，但报告指标是 NMSE，二者可能冲突

训练损失是 `nn.MSELoss()`，直接在稀疏域做全局 MSE。当前 `utils/statics.py::evaluator()`
计算的是每个样本的归一化误差，再对 batch 内样本取平均后转 dB：

```python
nmse = 10 * torch.log10((mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])).mean())
```

这会带来两个问题：

- MSE 下降不等价于 NMSE 下降。LoRA 某些实验 final loss 与 direct base loss 很接近，甚至略低，但 NMSE 仍明显更差。
- `Tester` 用 `AverageMeter.update(nmse)`，没有传入 batch 样本数，所以是对 batch 级 dB 结果等权平均，不是严格的样本级加权平均。`seed797/test` 用 `batch_size=32`，LoRA final test 用 `batch_size=256`，比较时存在 batch size 口径差异。

这个口径差异不太可能单独解释约 `1.2 dB` 的差距，但建议用相同 batch size 重评 direct base
和 LoRA checkpoint，确认差距的精确值。

### 4. seed796 可能存在 32/64 维度口径错误

`seed796` 的 base 和 LoRA 配置都是 `nt=32,nc=32`，但它们的 LoRA 数据路径与 `seed797`
一样指向 `scenario_2/01109`。`seed797` 对同一路径使用的是 `nt=64,nc=64`。

当前 `dataloader/dataloader.py` 只是 `torch.load()` 后直接构造 `TensorDataset`，没有 shape
校验。更关键的是 `models/TransNet.py` 前向里使用：

```python
src_shape = src.shape
src = src.view(-1, self.feature_shape[0], self.feature_shape[1])
...
output = output.reshape(src_shape)
```

如果 01109 真实 tensor 是 `(B, 2, 64, 64)`，而模型配置是 `nt=32,nc=32`，这里不会报错。
它会把一个 64x64 样本静默拆成 4 个 32x32 伪样本送入模型，再 reshape 回原 shape。这会改变
batch 语义和编码语义，导致结果看起来“能跑”，但不是标准的 64x64 TransNet 评估。

因为当前数据文件不可访问，我不能直接确认 01109 的真实 shape。但从 `seed797` 的配置看，
01109 很可能应按 64x64 处理。因此 `seed796` 结果不建议与 `seed797` 或 64x64 direct base
混合比较。

### 5. 当前没有证据表明是 checkpoint 加载失败

我检查了 seed796/seed797 的 LoRA 日志，没有发现：

- `Missing keys`
- `Unexpected keys`
- `Traceback`
- `RuntimeError`

日志都显示 `pretrained model loaded from ...`，所以现有材料不支持“权重没加载上”这个解释。
更像是加载成功后，LoRA 单组件微调把 01109 上的 NMSE 往坏方向优化了。

## 建议补做的验证

如果后续能下载权重和数据，建议按下面顺序验证：

1. 对 `seed797/base/torch_shared/best_nmse.pth` 在 01109 上分别用 `batch_size=32` 和 `256`
   复评，确认 direct base 的 NMSE 是否稳定在 `-21 dB` 附近。
2. 对每个 LoRA 实验增加 epoch 0 测试。LoRA 初始化输出应等价或接近 base；如果 epoch 0 是
   `-21 dB`，epoch 10 掉到 `-19.7 dB`，就能确认是 LoRA 训练早期破坏了 base。
3. 用同一套 checkpoint 做样本级加权 NMSE 统计，避免 batch size 差异影响结论。
4. 确认 `scenario_2/01109/train.pt` 和 `test.pt` 的真实 shape。如果是 `(N,2,64,64)`，
   `seed796` 的 32x32 实验需要标记为不同口径，最好不要作为有效 01109 结论。
5. 尝试更保守的 LoRA 超参：`lr=2e-4` 或 `1e-4`，加入 epoch 0/best checkpoint 评估，
   不只看 final model。
6. 尝试组合式适配，而不是单组件适配，例如同时启用 `fc_encoder fc_decoder`，或
   `encoder_ffn decoder_ffn fc_encoder fc_decoder`。当前只改 `fc_encoder` 不改
   `fc_decoder`，对自编码重建任务限制很强。
7. 给数据加载或模型前向加显式 shape assert，例如要求输入 `sparse_gt.shape[1:] ==
   (channel, nt, nc)`，并在模型中优先使用 `src.view(src.size(0), ...)`，避免 32/64 维度错误被
   `view(-1, ...)` 静默吞掉。

## 需要权重时能进一步分析什么

如果你下载权重，优先需要这些文件：

- `exps/WAIRD/seed797/base/torch_shared/checkpoints/best_nmse.pth`
- `exps/WAIRD/seed797/scenario_2/*/checkpoints/best_nmse.pth`
- 如果要补 `seed796` direct base：`exps/WAIRD/seed796/base/*/checkpoints/best_nmse.pth`
- 01109 的 `train.pt` 和 `test.pt`，至少能读 shape 和样本数即可

有权重后可以进一步做：

- base 与 LoRA 的同 batch size、同 evaluator 复评；
- epoch 0 LoRA 等价性检查；
- LoRA adapter 权重范数和输出差异分析；
- 每样本 NMSE 分布，定位是整体变差还是低能量样本被严重拉坏；
- 32x32 配置在 64x64 数据上是否发生静默拆样本。
