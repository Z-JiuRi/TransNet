#!/bin/bash

set -euo pipefail

# ==============================================================================
# 1. Data paths
# ==============================================================================
train_path=${train_path:-data/base/train.pt}
val_path=${val_path:-data/base/test.pt}
test_path=${test_path:-data/base/test.pt}

# ==============================================================================
# 2. Model and data shape
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

# ==============================================================================
# 3. Runtime settings
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-64}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-0}
gpu=${gpu:-0}
seed=${seed:-0}
exp_root=${exp_root:-WAIRD/seed${seed}/arch_no_lora_bs64}

run_one() {
  local transformer_backend=$1
  local layer_sharing=$2
  local gpu=$3
  local exp_name="${exp_root}/${transformer_backend}_${layer_sharing}"

  echo "============================================================"
  echo "Running backend=${transformer_backend}, layer_sharing=${layer_sharing}"
  echo "Experiment: exps/${exp_name}"
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
    > logs/${transformer_backend}_${layer_sharing}.log 2>&1 &
}

run_one torch shared 0
run_one torch independent 0
run_one original shared 1
run_one original independent 1  
