## Overview

This is the PyTorch implementation of paper ["TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System"](https://ieeexplore.ieee.org/document/9705497). You can cite our  paper by:

```
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
or
```
Y. Cui, A. Guo and C. Song, "TransNet: Full Attention Network for CSI Feedback in FDD Massive MIMO System," in IEEE Wireless Communications Letters, vol. 11, no. 5, pp. 903-907, May 2022, doi: 10.1109/LWC.2022.3149416.
```
## TransNet Architecture

TransNet is a Transformer autoencoder implemented in `models/TransNet.py`. The factory
`transnet(reduction=args.cr, d_model=args.d_model, channel=args.channel, nt=args.nt, nc=args.nc)`
builds a Transformer with **2 encoder layers**, **2 decoder layers**, **2 attention heads**,
and **dropout=0.0** by default. The compression ratio is controlled by `--cr` (passed as
`reduction`), and the CSI dimensions are controlled by `--channel`, `--nt`, and `--nc`.

Detailed data flow (default settings: `channel=2`, `nt=32`, `nc=32`, `d_model=64`, `cr=4`):
```
Input sparse CSI: (B, 2, 32, 32)
  |
  | flatten -> (B, 2048)
  v
Reshape to tokens: (B, 32, 64)   # seq_len = 2048 / 64, d_model = 64
  |
  v
Transformer Encoder x2
  - Multihead self-attn: nhead=2 (head_dim=32)
  - FFN: Linear(64 -> 256) + ReLU + Linear(256 -> 64)
  - Residual + LayerNorm (x2)
  |
  v
Flatten: (B, 2048)
  |
  v
fc_encoder: Linear(2048 -> 512)  # codeword size = input_dim / reduction
  |
  v
fc_decoder: Linear(512 -> 2048)
  |
  v
Reshape to tokens: (B, 32, 64)
  |
  v
Transformer Decoder x2
  - Multihead self-attn: nhead=2
  - Multihead cross-attn: tgt attends to memory
  - FFN: Linear(64 -> 256) + ReLU + Linear(256 -> 64)
  - Residual + LayerNorm (x3)
  |
  v
Output sparse CSI: (B, 2, 32, 32)
```

```
**********
Transformer(
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-1): 2 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
        )
        (linear1): Linear(in_features=64, out_features=2048, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (linear2): Linear(in_features=2048, out_features=64, bias=True)
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.0, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  )
  (decoder): TransformerDecoder(
    (layers): ModuleList(
      (0-1): 2 x TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
        )
        (multihead_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=64, out_features=64, bias=True)
        )
        (linear1): Linear(in_features=64, out_features=2048, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (linear2): Linear(in_features=2048, out_features=64, bias=True)
        (norm1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.0, inplace=False)
        (dropout2): Dropout(p=0.0, inplace=False)
        (dropout3): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  )
  (fc_encoder): Linear(in_features=2048, out_features=512, bias=True)
  (fc_decoder): Linear(in_features=512, out_features=2048, bias=True)
)
**********
```

```
encoder.layers.0.self_attn.in_proj_weight                         True     (192, 64)
encoder.layers.0.self_attn.in_proj_bias                           True     (192,)
encoder.layers.0.self_attn.out_proj.weight                        True     (64, 64)
encoder.layers.0.self_attn.out_proj.bias                          True     (64,)
encoder.layers.0.linear1.weight                                   True     (2048, 64)
encoder.layers.0.linear1.bias                                     True     (2048,)
encoder.layers.0.linear2.weight                                   True     (64, 2048)
encoder.layers.0.linear2.bias                                     True     (64,)
encoder.layers.0.norm1.weight                                     True     (64,)
encoder.layers.0.norm1.bias                                       True     (64,)
encoder.layers.0.norm2.weight                                     True     (64,)
encoder.layers.0.norm2.bias                                       True     (64,)
encoder.layers.1.self_attn.in_proj_weight                         True     (192, 64)
encoder.layers.1.self_attn.in_proj_bias                           True     (192,)
encoder.layers.1.self_attn.out_proj.weight                        True     (64, 64)
encoder.layers.1.self_attn.out_proj.bias                          True     (64,)
encoder.layers.1.linear1.weight                                   True     (2048, 64)
encoder.layers.1.linear1.bias                                     True     (2048,)
encoder.layers.1.linear2.weight                                   True     (64, 2048)
encoder.layers.1.linear2.bias                                     True     (64,)
encoder.layers.1.norm1.weight                                     True     (64,)
encoder.layers.1.norm1.bias                                       True     (64,)
encoder.layers.1.norm2.weight                                     True     (64,)
encoder.layers.1.norm2.bias                                       True     (64,)
encoder.norm.weight                                               True     (64,)
encoder.norm.bias                                                 True     (64,)
decoder.layers.0.self_attn.in_proj_weight                         True     (192, 64)
decoder.layers.0.self_attn.in_proj_bias                           True     (192,)
decoder.layers.0.self_attn.out_proj.weight                        True     (64, 64)
decoder.layers.0.self_attn.out_proj.bias                          True     (64,)
decoder.layers.0.multihead_attn.in_proj_weight                    True     (192, 64)
decoder.layers.0.multihead_attn.in_proj_bias                      True     (192,)
decoder.layers.0.multihead_attn.out_proj.weight                   True     (64, 64)
decoder.layers.0.multihead_attn.out_proj.bias                     True     (64,)
decoder.layers.0.linear1.weight                                   True     (2048, 64)
decoder.layers.0.linear1.bias                                     True     (2048,)
decoder.layers.0.linear2.weight                                   True     (64, 2048)
decoder.layers.0.linear2.bias                                     True     (64,)
decoder.layers.0.norm1.weight                                     True     (64,)
decoder.layers.0.norm1.bias                                       True     (64,)
decoder.layers.0.norm2.weight                                     True     (64,)
decoder.layers.0.norm2.bias                                       True     (64,)
decoder.layers.0.norm3.weight                                     True     (64,)
decoder.layers.0.norm3.bias                                       True     (64,)
decoder.layers.1.self_attn.in_proj_weight                         True     (192, 64)
decoder.layers.1.self_attn.in_proj_bias                           True     (192,)
decoder.layers.1.self_attn.out_proj.weight                        True     (64, 64)
decoder.layers.1.self_attn.out_proj.bias                          True     (64,)
decoder.layers.1.multihead_attn.in_proj_weight                    True     (192, 64)
decoder.layers.1.multihead_attn.in_proj_bias                      True     (192,)
decoder.layers.1.multihead_attn.out_proj.weight                   True     (64, 64)
decoder.layers.1.multihead_attn.out_proj.bias                     True     (64,)
decoder.layers.1.linear1.weight                                   True     (2048, 64)
decoder.layers.1.linear1.bias                                     True     (2048,)
decoder.layers.1.linear2.weight                                   True     (64, 2048)
decoder.layers.1.linear2.bias                                     True     (64,)
decoder.layers.1.norm1.weight                                     True     (64,)
decoder.layers.1.norm1.bias                                       True     (64,)
decoder.layers.1.norm2.weight                                     True     (64,)
decoder.layers.1.norm2.bias                                       True     (64,)
decoder.layers.1.norm3.weight                                     True     (64,)
decoder.layers.1.norm3.bias                                       True     (64,)
decoder.norm.weight                                               True     (64,)
decoder.norm.bias                                                 True     (64,)
fc_encoder.weight                                                 True     (512, 2048)
fc_encoder.bias                                                   True     (512,)
fc_decoder.weight                                                 True     (2048, 512)
fc_decoder.bias                                                   True     (2048,)
**********
```

Notes:
1. `dim_feedforward` defaults to `4 * d_model` when not specified.
2. `model.encode(src)` returns the compressed codeword from `fc_encoder`.
3. LoRA can be enabled on `encoder_self_attn`, `encoder_ffn`, `decoder_self_attn`,
   `decoder_cross_attn`, `decoder_ffn`, `fc_encoder`, and `fc_decoder` via
   `--lora_component`.

## Requirements

We support a env.yaml in our project, so you can simply run
```
conda env create -f environment.yaml
```
to get a useable environment. Or manually install and build your own environment, to use this project, you need to ensure the following main requirements are installed.


- Python >= 3.7
- scipy
- [1.2 =< PyTorch <= 1.6](https://pytorch.org/get-started/locally/)
- [thop==0.0.31-2005241907](https://github.com/Lyken17/pytorch-OpCounter) Note that the latest version leads to bug.
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- peft (for LoRA experiments; requires PyTorch >= 1.12 and transformers >= 4.31)

## Project Preparation

#### A. Data Preparation

The channel state information (CSI) matrix is generated from [COST2100](https://ieeexplore.ieee.org/document/6393523) model. Chao-Kai Wen and Shi Jin group provides a pre-processed version of COST2100 dataset in [Google Drive](https://drive.google.com/drive/folders/1_lAMLk_5k1Z8zJQlTr5NRnSD6ACaNRtj?usp=sharing), which is easier to use for the CSI feedback task; You can also download it from [Baidu Netdisk](https://pan.baidu.com/s/1Ggr6gnsXNwzD4ULbwqCmjA).

You can generate your own dataset according to the [open source library of COST2100](https://github.com/cost2100/cost2100) as well. The details of data pre-processing can be found in our paper.

#### B. Checkpoints Downloading

 You can check  the performance of indoor and outdoor scenarios by downloading checkpoints in [Google Drive](https://drive.google.com/drive/folders/1eoxryQfrMOPVtbiMRdxXtp5KsBt13-hI?usp=sharing). We support more detail checpoints in  [Google Drive](https://drive.google.com/drive/folders/10AxRFCE1Nbiqc0JgcFdQZ8mxQV8YbR8F?usp=sharing). You can also check the authenticity of our results by training a new TransNet yourself and see its performance, the test NMSE and training MSE loss will be printed during your training. A 400 epochs training dosen't take very long (about 3 and half hours on a single RTX 2060), and you are able to reproduce TransNet-400ep results in  Table 1 of our paper.



#### C. Project Tree Arrangement

We recommend you to arrange the project tree as follows.

```
home
├── TransNet  # The cloned TransNet repository
│   ├── dataset
│   ├── models
│   ├── utils
│   ├── main.py
├── COST2100  # The data folder
│   ├── trainin.pt
│   ├── valin.pt
│   ├── testin.pt
│   ├── ...
├── Experiments
│   ├── checkpoints  # The checkpoints folder
│   │     ├── 4_in.pth
│   │     ├── ...
│   ├── run.sh  # The bash script
...
```

## Train TransNet from Scratch

An example of run.sh is listed below. Simply use it with `sh run.sh`. It will start TransNet training from scratch. Select data by passing `--train_path`, `--val_path`, and `--test_path`. Change training epochs with `--epochs` and compression ratio with `--cr`.

``` bash
python /home/TransNet/main.py \
  --exp_name 'exp_1' \
  --train_path '/home/COST2100/in_train.pt' \
  --val_path '/home/COST2100/in_val.pt' \
  --test_path '/home/COST2100/in_test.pt' \
  --epochs 400 \
  --batch_size 200 \
  --workers 0 \
  --cr 4 \
  --nt 32 \
  --nc 32 \
  --scheduler const \
  --gpu 0
```

## LoRA Experiments

Use `--lora_component` to enable LoRA on any of these components:
`encoder_self_attn`, `encoder_ffn`, `decoder_self_attn`, `decoder_cross_attn`,
`decoder_ffn`, `fc_encoder`, and `fc_decoder`.

``` bash
python /home/TransNet/main.py \
  --exp_name 'lora_fc' \
  --train_path '/home/COST2100/in_train.pt' \
  --val_path '/home/COST2100/in_val.pt' \
  --test_path '/home/COST2100/in_test.pt' \
  --epochs 400 \
  --batch_size 200 \
  --workers 0 \
  --cr 4 \
  --nt 32 \
  --nc 32 \
  --lora_component encoder_self_attn encoder_ffn decoder_self_attn decoder_cross_attn decoder_ffn \
  --lora_rank 8 \
  --lora_alpha 16 \
  --scheduler const \
  --gpu 0
```

## Results and Reproduction

The main results reported in our paper are presented as follows. All the listed results can be found in Table1 of our paper. They are achieved from training TransNet with our  2 kind of training scheme (constant learning rate at 1e-4 for 400/1000 epochs).

Results of 400 epochs
Scenario | Compression Ratio | NMSE | Flops
:--: | :--: | :--: | :--:
indoor | 1/4 | -29.22 | 35.72M
indoor | 1/8 | -21.62 | 34.70M
indoor | 1/16 | -14.98 | 34.14M
indoor | 1/32 | -9.83 | 33.88M
indoor | 1/64 | -5.77 | 33.75M
outdoor | 1/4 | -13.99 | 35.72M
outdoor | 1/8 | -9.57 | 34.70M
outdoor | 1/16 | -6.90 | 34.14M
outdoor | 1/32 | -3.77 | 33.88M
outdoor | 1/64 | -2.20 | 33.75M

Results of 1000 epochs
Scenario | Compression Ratio | NMSE | Flops
:--: | :--: | :--: | :--:
indoor | 1/4 | -32.38 | 35.72M
indoor | 1/8 | -22.91 | 34.70M
indoor | 1/16 | -15.00 | 34.14M
indoor | 1/32 | -10.49 | 33.88M
indoor | 1/64 | -6.08 | 33.75M
outdoor | 1/4 | -14.86 | 35.72M
outdoor | 1/8 | -9.99 | 34.70M
outdoor | 1/16 | -7.82 | 34.14M
outdoor | 1/32 | -4.13 | 33.88M
outdoor | 1/64 | -2.62 | 33.75M

**To reproduce all these results, simplely add `--evaluate` to `scripts.sh` and pick the corresponding pre-trained model with `--pretrained`.** An example is shown as follows.

``` bash
python /home/TransNet/main.py \
  --exp_name 'eval_4_in' \
  --train_path '/home/COST2100/in_train.pt' \
  --val_path '/home/COST2100/in_val.pt' \
  --test_path '/home/COST2100/in_test.pt' \
  --pretrained './checkpoints/4_in.pth' \
  --evaluate \
  --batch_size 200 \
  --workers 0 \
  --nt 32 \
  --nc 32 \
  --cr 4\ # Note that cr should be same as  checkpoints
  --cpu

```




## Acknowledgment

Thank Chao-Kai Wen and Shi Jin group again for providing the pre-processed COST2100 dataset, you can find their related work named CsiNet in [Github-Python_CsiNet](https://github.com/sydney222/Python_CsiNet)


Thanks two open source works, CRNet and CLNet, that build on work above and advance the CSI feedback problem in DL, you can find their related work in [Github-Python-PyTorch CRNet](https://github.com/Kylin9511/CRNet) and [Github-Python-PyTorch CLNet](https://github.com/SIJIEJI/CLNet)

Thanks  the Github project members for the open source [Transformer tutorial](https://github.com/datawhalechina/Learn-NLP-with-Transformers), our base model for TransNet is based on their work.
