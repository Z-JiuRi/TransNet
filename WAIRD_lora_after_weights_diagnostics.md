# WAIRD LoRA 权重下载后的诊断结论

生成时间：2026-06-07

## 结论

现在有了本地 `data/` 和 checkpoint 后，可以确认之前“LoRA 比 direct base 差很多”的主要结论需要修正：

1. `seed797/test` 的 `-21.000 dB` 与 LoRA 日志里的约 `-19.70 dB` 不是公平比较，核心原因是当前 `Tester/evaluator` 的 NMSE 统计依赖 batch size。
2. 正确加载 `seed797` 的 LoRA checkpoint 后，大多数 LoRA 分支并没有变差，反而有很小改善。
3. `torch` backend 的 `encoder_ffn` LoRA 有一个真实代码问题：`model.eval()` 时 PyTorch Transformer fast path 会绕过 PEFT 替换的 `encoder.layer.linear1/linear2`，导致验证/测试时 adapter 不生效。
4. `seed796` 的 01109 实验配置是 `nt=32,nc=32`，但本地 `data/scenario_2/01109/*.pt` 实际是 `(N,2,64,64)`；由于模型里用了 `view(-1, ...)`，它不会报错，而是把一个样本拆成 4 个伪样本。因此 `seed796` 的 01109 结果口径不可靠。

## 本地数据确认

```text
data/base/train.pt                 (90000, 2, 64, 64)
data/base/test.pt                  (10000, 2, 64, 64)
data/scenario_2/01109/train.pt     (9000, 2, 64, 64)
data/scenario_2/01109/test.pt      (1000, 2, 64, 64)
```

这说明 01109 应按 `nt=64,nc=64` 跑。`seed797` 是这个口径，`seed796` 不是。

## 诊断脚本

新增脚本：

```text
scripts/lora_diagnostics.py
```

示例：

```bash
/home/z-jiuri/.envs/miniconda3/envs/torch/bin/python scripts/lora_diagnostics.py \
  --eval-only \
  --device cuda \
  --batch-size 256 \
  --lora-component decoder_ffn \
  --lora-checkpoint exps/WAIRD/seed797/scenario_2/01109_decoder_ffn_8_16_torch_shared/checkpoints/best_nmse.pth
```

检查 `torch encoder_ffn` LoRA 的真实 eval 效果时要加：

```bash
--disable-fastpath
```

## 问题 1：NMSE 统计依赖 batch size

当前 `utils/statics.py::evaluator()` 是对每个 batch 计算：

```python
10 * log10(mean(sample_nmse_ratio_in_this_batch))
```

然后 `utils/solver.py::Tester` 又对 batch 的 dB 值做等权平均，没有按样本数聚合。因此同一个模型、同一个 test set，batch size 不同会得到不同 NMSE。

对 `seed797/base/torch_shared` 在同一个 `01109/test.pt` 上复评：

| 口径 | batch size | NMSE |
| --- | ---: | ---: |
| 当前日志式 batch 平均 | 32 | `-21.000 dB` |
| 当前日志式 batch 平均 | 256 | `-19.702 dB` |
| 全数据 `10log10(mean ratio)` | 1000 | `-19.363 dB` |
| 样本 dB 后平均 | 任意 | `-26.325 dB` |

所以 `seed797/test` 的 `-21.000 dB` 看起来比 LoRA `-19.70 dB` 好，主要是 batch size 从 32 变成 256 造成的统计口径差异，不是 LoRA 真把模型拉坏 1.3 dB。

## 问题 2：`torch encoder_ffn` LoRA 在 eval 时被 fast path 绕过

对 `seed797/base/torch_shared` 加 `encoder_ffn` LoRA，强制把 LoRA 权重填成很大的值：

| 模式 | fast path | 输出相对 base 偏移 |
| --- | --- | ---: |
| `eval()` | 开 | `3.27e-7` |
| `train()` | 开 | `1.07` |
| `eval()` | 关 | `1.07` |

这说明：

- 训练时 `encoder_ffn` LoRA 参与计算，有梯度。
- 但测试/验证时 `model.eval()` 会走 PyTorch fast path，绕过 PEFT 替换后的 `linear1/linear2`。
- 关闭 `torch.backends.mha.set_fastpath_enabled(False)` 后，eval 也能看到 LoRA 效果。

已保存的 `seed797/scenario_2/01109_encoder_ffn_8_16_torch_shared` checkpoint：

| 口径 | base global NMSE | LoRA global NMSE | 变化 |
| --- | ---: | ---: | ---: |
| fast path 开 | `-19.3631` | `-19.3631` | `0.0000 dB` |
| fast path 关 | `-19.3631` | `-19.3660` | 改善 `0.0029 dB` |

也就是说这个分支不是变差，而是原日志测试时基本没测到 adapter。

## seed797 全部 LoRA checkpoint 的真实样本级结果

下面用本地 checkpoint 正确加载 LoRA，并用样本 dB 平均口径比较每个 LoRA 与对应 base。`delta_db = lora_nmse - base_nmse`，越负越好。

| 实验 | base NMSE | LoRA NMSE | delta |
| --- | ---: | ---: | ---: |
| `decoder_ffn original independent` | `-26.8368` | `-26.9624` | `-0.1256` |
| `decoder_ffn original shared` | `-26.3723` | `-26.4160` | `-0.0437` |
| `decoder_ffn torch independent` | `-26.8174` | `-26.9323` | `-0.1149` |
| `decoder_ffn torch shared` | `-26.3250` | `-26.3722` | `-0.0472` |
| `encoder_ffn original independent` | `-26.8368` | `-26.8989` | `-0.0622` |
| `encoder_ffn original shared` | `-26.3723` | `-26.4002` | `-0.0278` |
| `encoder_ffn torch independent` | `-26.8174` | `-26.8174` | `0.0000` |
| `encoder_ffn torch shared` | `-26.3250` | `-26.3250` | `0.0000` |
| `fc_encoder original independent` | `-26.8368` | `-26.8696` | `-0.0328` |
| `fc_encoder original shared` | `-26.3723` | `-26.4025` | `-0.0302` |
| `fc_encoder torch independent` | `-26.8174` | `-26.8493` | `-0.0319` |
| `fc_encoder torch shared` | `-26.3250` | `-26.3584` | `-0.0334` |

`torch encoder_ffn` 的 `0.0000` 是 fast path 绕过造成的；关闭 fast path 后也有极小改善。

## 逐 epoch probe：LoRA 是否失败

我挑了 `seed797/torch_shared/encoder_ffn`，从 base checkpoint 重新挂 LoRA 跑 10 epoch，每轮输出 LoRA 范数、输出偏移和测试 NMSE。

现象：

- LoRA 参数范数从 `2.337` 增加到 `2.396`，说明参数确实在训练。
- 在默认 eval fast path 开启时，测试输出对 base 的偏移一直显示为 `0`，说明测试路径没看到 adapter。
- 训练 loss 在 `6.09e-3 ~ 6.33e-3` 附近，和日志一致。

因此这个分支的问题不是“adapter 学坏了”，而是“训练时 adapter 生效，但 eval/test 时 adapter 被 fast path 绕过”。

## 建议修复

### 1. 修 NMSE 聚合

建议不要在 `evaluator()` 里直接返回 batch dB 后再平均。更稳妥的是返回每个样本的 ratio，最终在整个 test set 上统一聚合：

```python
ratio = mse.sum(dim=[1, 2]) / power_gt.sum(dim=[1, 2])
nmse_global = 10 * torch.log10(ratio.mean())
nmse_sample_mean = (10 * torch.log10(ratio)).mean()
```

至少要保证 direct base 和 LoRA 使用同一个 batch size，否则当前日志里的 NMSE 不能直接比较。

### 2. LoRA + torch backend 时关闭 fast path

只要启用 LoRA，尤其是 `encoder_ffn` 目标模块，建议初始化时关闭：

```python
torch.backends.mha.set_fastpath_enabled(False)
```

或者避免用 PyTorch 原生 `TransformerEncoderLayer` fast path，改用 `original` backend 或自定义 encoder layer。

### 3. 加输入 shape 校验

`models/TransNet.py` 里不要用：

```python
src = src.view(-1, self.feature_shape[0], self.feature_shape[1])
```

至少应先 assert：

```python
assert tuple(src.shape[1:]) == (self.channel, self.nt, self.nc)
src = src.view(src.size(0), self.feature_shape[0], self.feature_shape[1])
```

这能防止 `seed796` 这种 32x32 配置静默处理 64x64 数据。

### 4. LoRA checkpoint 加载要先构造 LoRA 再 load

LoRA checkpoint 的 key 是 `base_model.model...lora_A...`，不能直接通过当前 `--pretrained` 加载到裸 TransNet。正确顺序是：

1. 构造 base TransNet；
2. 加载 base checkpoint；
3. 调用 `lora_component()` 挂 adapter；
4. 加载 LoRA checkpoint。

`scripts/lora_diagnostics.py` 已按这个顺序实现。
