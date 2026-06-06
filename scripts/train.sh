#!/bin/bash

# scen_name=base nt=32 nc=32 dim_feedforward=2048 seed=3232 gpu=2 bash scripts/train.sh

# ==============================================================================
# 1. 基础路径
# ==============================================================================
scen_name=${scen_name:-base}
train_path=${train_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/test.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/test.pt}

# ==============================================================================
# 2. 模型结构与数据维度参数
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

# ==============================================================================
# 3. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-100}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-2}
seed=${seed:-3232}
transformer_backend=${transformer_backend:-torch}
layer_sharing=${layer_sharing:-shared}
exp_name=${exp_name:-WAIRD/seed${seed}/${scen_name}/${transformer_backend}_${layer_sharing}}



# ==============================================================================
# 4. 运行 Python 脚本
# ==============================================================================
mktouch() {
    mkdir -p "$(dirname "$1")" && touch "$1"
}

log_file="exps/${exp_name}/train.out"

mktouch "${log_file}"

echo "============================================================"
echo "Running backend=${transformer_backend}, layer_sharing=${layer_sharing}"
echo "Seed: ${seed}, GPU: ${gpu}"
echo "Experiment: ${exp_name}"
echo "============================================================"

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
  --seed "${seed}" \
  --transformer_backend "${transformer_backend}" \
  --layer_sharing "${layer_sharing}" \
  > "${log_file}" 2>&1 &
 