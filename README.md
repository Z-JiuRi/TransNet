# TransNet CSI Feedback

本仓库是 TransNet 在 CSI 反馈任务上的 PyTorch 实现，并在原始 COST2100
自编码重建流程基础上扩展了多数据集训练、可切换 Transformer 后端、层共享策略、
冻结微调、手写 LoRA、latent adapter、低秩 FC 层、DeepMIMO 数据转换和码字提取。

原始论文：

```bibtex
@ARTICLE{9705497,
  author={Cui, Yaodong and Guo, Aihuang and Song, Chunlin},
  journal={IEEE Wireless Communications Letters},
  title={TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System},
  year={2022},
  volume={11},
  number={5},
  pages={903-907},
  doi={10.1109/LWC.2022.3149416}}
```

## 功能概览

- 基础 TransNet 自编码训练与评估：输入稀疏 CSI，输出同形状重建结果。
- 数据格式支持：`.pt`、`.pth`、`.npy`、单数组 `.npz`，运行时统一转为
  `float32`。
- 可配置 CSI 维度：通过 `--channel`、`--nt`、`--nc` 控制输入形状。
- 可配置模型维度：`--d_model`、`--dim_feedforward`、`--cr`。
- 两种 Transformer 后端：`--transformer_backend torch|original`。
- 两种层策略：`--layer_sharing shared|independent`。
- 迁移学习：支持 `--pretrained` 加载基础模型，`--resume` 恢复完整训练状态。
- 冻结微调：通过 `--freeze_components` 冻结指定模块。
- 手写 LoRA：支持在 FFN、FC 层或 latent adapter 上做参数高效微调。
- 低秩 FC 替换：`--fc_lora` 将 `fc_encoder/fc_decoder` 替换为低秩分解层。
- 训练日志：每个实验保存 `args.json`、`run.log`、TensorBoard 文件和 checkpoint。
- 工具脚本：批量训练/测试、DeepMIMO `.npy -> .pt` 转换、码字提取、LoRA 诊断。

## 项目结构

```text
.
├── main.py                         # 训练/评估主入口
├── models/TransNet.py              # TransNet、Transformer 后端、LoRA/adapter 相关模块
├── dataloader/dataloader.py        # .pt/.pth/.npy/.npz 数据加载
├── utils/
│   ├── parser.py                   # CLI 参数
│   ├── init.py                     # 设备、模型、冻结、LoRA 初始化
│   ├── solver.py                   # Trainer / Tester
│   ├── scheduler.py                # const / cosine 调度器
│   ├── statics.py                  # MSE / NMSE 指标
│   └── logger.py                   # 日志
├── scripts/
│   ├── train.sh                    # 基础训练脚本
│   ├── test.sh                     # 基础评估脚本
│   ├── train_lora.sh               # LoRA 微调脚本
│   ├── lora.sh                     # LoRA 实验脚本
│   ├── train_on_base_pretrained.sh # 加载基础模型后冻结/微调
│   ├── convert_deepmimo_npy_to_pt.py
│   ├── convert_deepmimo_npy_to_pt.sh
│   ├── extract_codewords.py
│   └── codewords.sh
└── env.yaml                        # Conda 环境
```

请不要提交数据集、日志、TensorBoard 文件或 checkpoint。推荐将数据放在仓库外部，
或放在已忽略的本地目录中，例如 `data/`、`COST2100/`、`Experiments/`。

## 环境安装

当前代码在本地验证环境中使用 PyTorch 2.5.1。由于代码中使用了
`torch.load(..., weights_only=False)`，不建议直接使用过旧的 PyTorch 版本。

推荐准备一个 Python 3.8+ 环境，并安装：

```bash
pip install torch torchvision tensorboard scipy numpy thop
```

如果只做手写 LoRA/adapter 实验，不需要 PEFT；`peft/transformers/accelerate`
只作为兼容旧实验环境的可选依赖。

仓库仍保留了原始 `env.yaml`：

```bash
conda env create -f env.yaml
conda activate transnet
```

但该文件固定在 Python 3.8、PyTorch 1.6.0、CUDA 10.1，属于 legacy 环境定义；
若使用当前代码，建议升级 PyTorch 或同步更新 `env.yaml`。

## 数据格式

主程序要求显式传入训练、验证和测试数据：

```text
--train_path /path/to/train.pt
--val_path   /path/to/val.pt
--test_path  /path/to/test.pt
```

每个文件应保存一个形状为 `(N, channel, nt, nc)` 的数组或张量。默认 COST2100
设置是：

```text
(N, 2, 32, 32)
```

WAIRD/DeepMIMO 实验中常用：

```text
(N, 2, 64, 64)
```

注意：当前 dataloader 不自动 reshape 数据，也不自动推断 `nt/nc`。数据尾部维度
必须与命令行中的 `--channel --nt --nc` 保持一致。

## 模型与数据流

`models/TransNet.py::transnet()` 创建 Transformer 自编码器：

```text
input_dim = channel * nt * nc
code_dim  = input_dim / cr
seq_len   = input_dim / d_model
```

以 COST2100 默认参数 `channel=2, nt=32, nc=32, d_model=64, cr=4` 为例：

```text
Input sparse CSI:      (B, 2, 32, 32)
Flatten / tokenize:   (B, 32, 64)
Transformer encoder:  (B, 32, 64)
fc_encoder:           (B, 512)
latent adapter:       (B, 512)       # 默认 Identity，启用 adapter 时为残差瓶颈
fc_decoder:           (B, 2048)
Transformer decoder:  (B, 32, 64)
Output sparse CSI:    (B, 2, 32, 32)
```

维度约束：

- `channel * nt * nc` 必须能被 `--d_model` 整除。
- `channel * nt * nc` 必须能被 `--cr` 整除。
- 压缩率写作 `1 / cr`，例如 `--cr 4` 表示码字长度为原始展平维度的 `1/4`。

## 训练、验证与指标

训练由 `utils/solver.py::Trainer` 执行：

- DataLoader batch 为单元素 tuple：`(sparse_gt,)`。
- 模型输入 `sparse_gt`，输出 `sparse_pred`，两者形状相同。
- 训练损失是 `nn.MSELoss()`。
- `--scheduler const` 使用固定学习率。
- `--scheduler cosine` 使用 warmup cosine 调度。
- 验证默认每 10 个 epoch 执行一次，测试默认每 10 个 epoch 执行一次。
- 最优测试 NMSE checkpoint 保存为 `exps/{exp_name}/checkpoints/best_nmse.pth`。

测试由 `utils/solver.py::Tester` 执行，报告：

- `loss`：稀疏域 MSE。
- `NMSE`：对每个样本计算
  `sum(|pred - gt|^2) / sum(|gt|^2)`，再对全测试集求平均并转换为 dB。

## 常用参数

### 数据与运行模式

| 参数 | 说明 |
| --- | --- |
| `--train_path` | 训练数据路径，必填 |
| `--val_path` | 验证数据路径，必填 |
| `--test_path` | 测试数据路径，必填 |
| `--batch_size` | batch size，必填 |
| `--workers` | DataLoader worker 数，必填 |
| `--evaluate` | 只评估，不训练 |
| `--pretrained` | 加载模型权重，通常用于评估或微调 |
| `--resume` | 恢复训练 checkpoint，包括 optimizer/scheduler |
| `--exp_name` | 输出目录名，实际路径为 `./exps/{exp_name}` |
| `--seed` | 随机种子，并启用 deterministic 设置 |
| `--gpu` | 指定 GPU id |
| `--cpu` | 强制 CPU |

### 模型结构

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--channel` | `2` | CSI 通道数，通常为实部/虚部 |
| `--nt` | `32` | 天线维 |
| `--nc` | `32` | 延迟/频域维 |
| `--cr` | `4` | 压缩率分母 |
| `--d_model` | `64` | Transformer token 维度 |
| `--dim_feedforward` | `2048` | FFN 隐层维度 |
| `--transformer_backend` | `torch` | `torch` 或 `original` |
| `--layer_sharing` | `shared` | `shared` 或 `independent` |
| `--fc_lora` | 关闭 | 用低秩 Linear 替换 FC 编码/解码层 |
| `--fc_lora_rank` | 自动 | 低秩 FC rank，默认 `code_dim // 4` |

### 微调参数

| 参数 | 说明 |
| --- | --- |
| `--freeze_components` | 冻结指定组件，适合加载基础模型后做部分微调 |
| `--lora_component` | 启用手写 LoRA/adapter，启用后基础参数会被冻结 |
| `--lora_pretrained` | 在 base `--pretrained` 后加载 LoRA checkpoint |
| `--lora_rank` | LoRA 或 adapter rank |
| `--lora_alpha` | LoRA scaling 中的 alpha；`adapter` 不使用 alpha |

`--freeze_components` 可选：

```text
encoder_self_attn encoder_ffn decoder_self_attn decoder_cross_attn decoder_ffn fc_encoder fc_decoder
```

`--lora_component` 可选：

```text
encoder_ffn decoder_ffn fc_encoder fc_decoder adapter
```

注意：当前手写 LoRA 只包裹 FFN 和 FC 中的 `nn.Linear`；attention LoRA 不在
`--lora_component` 选项中。若同时传入 `--lora_component` 和
`--freeze_components`，冻结参数会被忽略，因为 LoRA 初始化会统一冻结基础参数，
只训练 `lora_*` 或 `adapter_*` 参数。

## 示例命令

### 1. COST2100 从头训练

```bash
python main.py \
  --exp_name COST2100/in_cr4 \
  --train_path ./COST2100/in_train.pt \
  --val_path ./COST2100/in_val.pt \
  --test_path ./COST2100/in_test.pt \
  --epochs 400 \
  --batch_size 200 \
  --workers 0 \
  --channel 2 \
  --nt 32 \
  --nc 32 \
  --cr 4 \
  --d_model 64 \
  --dim_feedforward 2048 \
  --scheduler const \
  --lr_init 1e-4 \
  --gpu 0 \
  --seed 42
```

### 2. COST2100 checkpoint 评估

```bash
python main.py \
  --exp_name COST2100/eval_in_cr4 \
  --train_path ./COST2100/in_train.pt \
  --val_path ./COST2100/in_val.pt \
  --test_path ./COST2100/in_test.pt \
  --pretrained ./checkpoints/4_in.pth \
  --evaluate \
  --batch_size 200 \
  --workers 0 \
  --channel 2 \
  --nt 32 \
  --nc 32 \
  --cr 4 \
  --cpu
```

评估时 `--cr --channel --nt --nc --d_model --dim_feedforward` 必须与 checkpoint
训练时一致。

### 3. WAIRD/DeepMIMO 64x64 基础训练

```bash
python main.py \
  --exp_name WAIRD/seed797/base/torch_independent \
  --train_path data/WAIRD/base/train.pt \
  --val_path data/WAIRD/base/test.pt \
  --test_path data/WAIRD/base/test.pt \
  --epochs 400 \
  --batch_size 256 \
  --workers 0 \
  --channel 2 \
  --nt 64 \
  --nc 64 \
  --cr 4 \
  --d_model 64 \
  --dim_feedforward 2048 \
  --transformer_backend torch \
  --layer_sharing independent \
  --scheduler cosine \
  --lr_init 2e-4 \
  --weight_decay 1e-3 \
  --gpu 0 \
  --seed 797
```

等价脚本入口：

```bash
scen_name=base \
data_root=data/WAIRD \
epochs=400 \
batch_size=256 \
seed=797 \
gpu=0 \
nt=64 \
nc=64 \
transformer_backend=torch \
layer_sharing=independent \
bash scripts/train.sh
```

`scripts/train.sh` 默认后台运行，并将 stdout/stderr 写入
`exps/${exp_name}/train.out`。

### 4. 评估 WAIRD/DeepMIMO 基础模型

```bash
python main.py \
  --exp_name WAIRD/seed797/test/scenario_2_01109_torch_independent \
  --train_path data/WAIRD/scenario_2/01109/train.pt \
  --val_path data/WAIRD/scenario_2/01109/test.pt \
  --test_path data/WAIRD/scenario_2/01109/test.pt \
  --pretrained exps/WAIRD/seed797/base/torch_independent/checkpoints/best_nmse.pth \
  --evaluate \
  --batch_size 256 \
  --workers 0 \
  --channel 2 \
  --nt 64 \
  --nc 64 \
  --cr 4 \
  --d_model 64 \
  --dim_feedforward 2048 \
  --transformer_backend torch \
  --layer_sharing independent \
  --gpu 0 \
  --seed 797
```

脚本入口：

```bash
seed=797 \
scen_name=scenario_2/01109 \
data_root=data/WAIRD \
transformer_backend=torch \
layer_sharing=independent \
gpu=0 \
bash scripts/test.sh
```

### 5. 加载基础模型后冻结部分模块微调

下面示例只训练 encoder 相关参数，冻结 decoder 和 FC：

```bash
python main.py \
  --exp_name WAIRD/seed797/scenario_2_01105_only_encoder \
  --train_path data/WAIRD/scenario_2/01105/train.pt \
  --val_path data/WAIRD/scenario_2/01105/test.pt \
  --test_path data/WAIRD/scenario_2/01105/test.pt \
  --pretrained exps/WAIRD/seed797/base/torch_independent/checkpoints/best_nmse.pth \
  --freeze_components decoder_self_attn decoder_cross_attn decoder_ffn fc_encoder fc_decoder \
  --epochs 200 \
  --batch_size 256 \
  --workers 0 \
  --nt 64 \
  --nc 64 \
  --cr 4 \
  --transformer_backend torch \
  --layer_sharing independent \
  --scheduler cosine \
  --lr_init 2e-4 \
  --gpu 0 \
  --seed 797
```

脚本入口：

```bash
scen_name=scenario_2/01105 \
seed=797 \
nt=64 \
nc=64 \
transformer_backend=torch \
layer_sharing=independent \
freeze_components="decoder_self_attn decoder_cross_attn decoder_ffn fc_encoder fc_decoder" \
bash scripts/train_on_base_pretrained.sh
```

### 6. LoRA 微调 FFN 或 FC

LoRA 会先加载 `--pretrained`，然后冻结基础模型，只训练 `lora_*` 参数。

```bash
python main.py \
  --exp_name WAIRD/seed797/scenario_2_01105_fc_encoder_lora \
  --train_path data/WAIRD/scenario_2/01105/train.pt \
  --val_path data/WAIRD/scenario_2/01105/test.pt \
  --test_path data/WAIRD/scenario_2/01105/test.pt \
  --pretrained exps/WAIRD/seed797/base/torch_independent/checkpoints/best_nmse.pth \
  --lora_component fc_encoder \
  --lora_rank 256 \
  --lora_alpha 1024 \
  --epochs 200 \
  --batch_size 32 \
  --workers 0 \
  --nt 64 \
  --nc 64 \
  --cr 4 \
  --transformer_backend torch \
  --layer_sharing independent \
  --scheduler cosine \
  --lr_init 5e-4 \
  --weight_decay 0 \
  --gpu 0 \
  --seed 797
```

多个组件可以一起传入：

```bash
--lora_component encoder_ffn decoder_ffn fc_encoder fc_decoder
```

脚本入口：

```bash
seed=797 \
gpu=0 \
scen_name=scenario_2/01105 \
data_root=data/WAIRD \
lora_component=fc_encoder \
lora_rank=256 \
lora_alpha=1024 \
epochs=200 \
batch_size=32 \
transformer_backend=torch \
layer_sharing=independent \
bash scripts/train_lora.sh
```

### 7. Latent adapter 微调

`adapter` 在压缩码字 `code_dim` 上添加残差瓶颈适配器，适合只调整 latent
codeword 空间：

```bash
python main.py \
  --exp_name WAIRD/seed797/scenario_2_01105_adapter_r64 \
  --train_path data/WAIRD/scenario_2/01105/train.pt \
  --val_path data/WAIRD/scenario_2/01105/test.pt \
  --test_path data/WAIRD/scenario_2/01105/test.pt \
  --pretrained exps/WAIRD/seed797/base/torch_shared/checkpoints/best_nmse.pth \
  --lora_component adapter \
  --lora_rank 64 \
  --lora_alpha 16 \
  --epochs 200 \
  --batch_size 256 \
  --workers 0 \
  --nt 64 \
  --nc 64 \
  --cr 4 \
  --transformer_backend torch \
  --layer_sharing shared \
  --scheduler cosine \
  --lr_init 2e-4 \
  --weight_decay 0 \
  --gpu 0 \
  --seed 797
```

### 8. 低秩 FC 替换

`--fc_lora` 与手写 LoRA 不同：它直接把 `fc_encoder/fc_decoder` 替换为
`LowRankLinear`，属于模型结构变化。训练和评估都必须使用同样的 `--fc_lora` 和
`--fc_lora_rank`。

```bash
python main.py \
  --exp_name WAIRD/seed797/base/torch_independent_fc_lora_r512 \
  --train_path data/WAIRD/base/train.pt \
  --val_path data/WAIRD/base/test.pt \
  --test_path data/WAIRD/base/test.pt \
  --epochs 400 \
  --batch_size 256 \
  --workers 0 \
  --nt 64 \
  --nc 64 \
  --cr 4 \
  --transformer_backend torch \
  --layer_sharing independent \
  --fc_lora \
  --fc_lora_rank 512 \
  --scheduler cosine \
  --lr_init 2e-4 \
  --gpu 0 \
  --seed 797
```

### 9. 恢复训练

```bash
python main.py \
  --exp_name WAIRD/seed797/base/torch_independent_resume \
  --train_path data/WAIRD/base/train.pt \
  --val_path data/WAIRD/base/test.pt \
  --test_path data/WAIRD/base/test.pt \
  --resume exps/WAIRD/seed797/base/torch_independent/checkpoints/best_nmse.pth \
  --epochs 500 \
  --batch_size 256 \
  --workers 0 \
  --nt 64 \
  --nc 64 \
  --cr 4 \
  --transformer_backend torch \
  --layer_sharing independent \
  --scheduler cosine \
  --gpu 0 \
  --seed 797
```

`--resume` 需要 checkpoint 中包含 `optimizer` 和 `scheduler` 状态。本仓库保存的
`best_nmse.pth` 满足这个格式。

## DeepMIMO 数据转换

如果 DeepMIMO 数据以 `.npy` 保存，可使用转换脚本批量转为 `.pt`，目录结构会被
保留：

```bash
python scripts/convert_deepmimo_npy_to_pt.py \
  --src_root /storage/hujiacong/zxd/datasets/deepmimo/data \
  --dst_root ./data/DeepMIMO \
  --workers 2 \
  --validate_existing size
```

先查看将要转换的文件：

```bash
python scripts/convert_deepmimo_npy_to_pt.py \
  --src_root /storage/hujiacong/zxd/datasets/deepmimo/data \
  --dst_root ./data/DeepMIMO \
  --dry_run
```

脚本入口：

```bash
src_root=/storage/hujiacong/zxd/datasets/deepmimo/data \
dst_root=./data/DeepMIMO \
workers=2 \
bash scripts/convert_deepmimo_npy_to_pt.sh
```

## 提取压缩码字

`scripts/extract_codewords.py` 调用 `model.encode()`，将输入 CSI 转成压缩码字并
保存为 `.pt`：

```bash
python scripts/extract_codewords.py \
  --data data/WAIRD/scenario_2/00032/train.pt \
  --checkpoint exps/WAIRD/seed42/base/torch_shared/checkpoints/best_nmse.pth \
  --output ./codewords.pt \
  --batch_size 200 \
  --gpu 0 \
  --d_model 64 \
  --dim_feedforward 2048 \
  --cr 4 \
  --channel 2 \
  --nt 64 \
  --nc 64
```

输出形状为 `(N, code_dim)`。例如 `(N, 2, 64, 64)` 且 `cr=4` 时，
`code_dim = 2 * 64 * 64 / 4 = 2048`。

## 输出目录

每次运行会写入：

```text
exps/{exp_name}/
├── args.json
├── run.log
├── train.out 或 test.out      # 使用脚本重定向时生成
├── tensorboard/
└── checkpoints/
    └── best_nmse.pth
```

查看 TensorBoard：

```bash
tensorboard --logdir exps
```

查看脚本后台训练日志：

```bash
tail -f exps/WAIRD/seed797/base/torch_independent/train.out
```

## 原始 COST2100 复现结果

下表为原始论文中 TransNet 的 COST2100 结果，供对照使用。复现时请确认数据预处理、
场景、压缩率、epoch 数和 checkpoint 配置一致。

400 epochs：

| Scenario | Compression Ratio | NMSE | FLOPs |
| --- | --- | --- | --- |
| indoor | 1/4 | -29.22 | 35.72M |
| indoor | 1/8 | -21.62 | 34.70M |
| indoor | 1/16 | -14.98 | 34.14M |
| indoor | 1/32 | -9.83 | 33.88M |
| indoor | 1/64 | -5.77 | 33.75M |
| outdoor | 1/4 | -13.99 | 35.72M |
| outdoor | 1/8 | -9.57 | 34.70M |
| outdoor | 1/16 | -6.90 | 34.14M |
| outdoor | 1/32 | -3.77 | 33.88M |
| outdoor | 1/64 | -2.20 | 33.75M |

1000 epochs：

| Scenario | Compression Ratio | NMSE | FLOPs |
| --- | --- | --- | --- |
| indoor | 1/4 | -32.38 | 35.72M |
| indoor | 1/8 | -22.91 | 34.70M |
| indoor | 1/16 | -15.00 | 34.14M |
| indoor | 1/32 | -10.49 | 33.88M |
| indoor | 1/64 | -6.08 | 33.75M |
| outdoor | 1/4 | -14.86 | 35.72M |
| outdoor | 1/8 | -9.99 | 34.70M |
| outdoor | 1/16 | -7.82 | 34.14M |
| outdoor | 1/32 | -4.13 | 33.88M |
| outdoor | 1/64 | -2.62 | 33.75M |

## 数据与 checkpoint

COST2100 预处理数据可参考 Chao-Kai Wen 和 Shi Jin 组公开的数据：

- Google Drive:
  https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing
- Baidu Netdisk:
  https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA

原始 TransNet checkpoint：

- Google Drive:
  https://drive.google.com/drive/folders/1eoxryQfrMOPVtbiMRdxXtp5KsBt13-hI?usp=sharing
- More checkpoints:
  https://drive.google.com/drive/folders/10AxRFCE1Nbiqc0JgcFdQZ8mxQV8YbR8F?usp=sharing

也可以根据 COST2100 开源库自行生成数据：

https://github.com/cost2100/cost2100

## 实验记录建议

报告实验结果时建议至少记录：

- 数据集和场景，例如 COST2100 indoor/outdoor、WAIRD `scenario_2/01105`、
  DeepMIMO city 名称。
- 数据路径和输入维度：`channel/nt/nc`。
- 压缩率：`cr`。
- 模型结构：`d_model`、`dim_feedforward`、`transformer_backend`、
  `layer_sharing`、是否启用 `fc_lora`。
- 微调方式：冻结组件、LoRA 组件、rank、alpha、adapter rank。
- 训练配置：epoch、batch size、scheduler、学习率、weight decay、seed、GPU。
- checkpoint 路径和最终 `loss/NMSE`。

## Acknowledgment

感谢 Chao-Kai Wen 和 Shi Jin 组提供预处理 COST2100 数据，其相关 CsiNet 工作可见：
https://github.com/sydney222/Python_CsiNet

感谢 CRNet 和 CLNet 等 CSI feedback 开源工作：

- https://github.com/Kylin9511/CRNet
- https://github.com/SIJIEJI/CLNet

原始 TransNet 实现参考了 Datawhale Transformer 教程：
https://github.com/datawhalechina/Learn-NLP-with-Transformers
