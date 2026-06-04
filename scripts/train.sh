#!/bin/bash

# ==============================================================================
# 1. 基础路径
# ==============================================================================
train_path=${train_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/base/train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/base/val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/base/test.pt}

# ==============================================================================
# 2. 模型结构与数据维度参数
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

# ==============================================================================
# 3. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-200}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-2}
seed=${seed:-42}
exp_name=${exp_name:-WAIRD/seed${seed}/base}


# ==============================================================================
# 4. 运行 Python 脚本
# ==============================================================================
python ./main.py \
  --exp_name "${exp_name}" \
  --train_path "${train_path}" \
  --val_path "${val_path}" \
  --test_path "${test_path}" \
  --epochs "${epochs}" \
  --d_model "${d_model}" \
  --nt "${nt}" \
  --nc "${nc}" \
  --dim_feedforward "${dim_feedforward}" \
  --batch_size "${batch_size}" \
  --workers "${workers}" \
  --cr "${cr}" \
  --scheduler "${scheduler}" \
  --lr_init "${lr_init}" \
  --weight_decay "${weight_decay}" \
  --gpu "${gpu}" \
  --seed "${seed}"