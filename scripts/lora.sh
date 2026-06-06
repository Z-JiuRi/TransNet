#!/bin/bash

# scen_name=scenario_2/00032 nt=32 nc=32 dim_feedforward=2048 lora_rank=8 lora_alpha=16 gpu=2 seed=3232 bash scripts/lora.sh

# ==============================================================================
# 1. 基础路径
# ==============================================================================
scen_name=${scen_name:-"scenario_2/00247"}
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
# 3. LoRA 参数
# ==============================================================================
lora_component=${lora_component:-decoder_ffn}
lora_rank=${lora_rank:-8}
lora_alpha=${lora_alpha:-16}

# ==============================================================================
# 4. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-200}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-0}
gpu=${gpu:-2}
seed=${seed:-3232}
transformer_backend=${transformer_backend:-torch}
layer_sharing=${layer_sharing:-shared}
exp_name=${exp_name:-WAIRD/seed${seed}/${scen_name}_${lora_component}_${lora_rank}_${lora_alpha}_${transformer_backend}_${layer_sharing}}
pretrained=${pretrained:-/storage/hujiacong/zxd/Huawei/TransNet/exps/WAIRD/seed${seed}/base/${transformer_backend}_${layer_sharing}/checkpoints/best_nmse.pth}

# ==============================================================================
# 5. 运行 Python 脚本
# ==============================================================================

mktouch() {
    mkdir -p "$(dirname "$1")" && touch "$1"
}

log_file="exps/${exp_name}/train.out"

mktouch "${log_file}"

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
  --pretrained "${pretrained}" \
  --lora_component ${lora_component} \
  --lora_rank "${lora_rank}" \
  --lora_alpha "${lora_alpha}" \
  --transformer_backend "${transformer_backend}" \
  --layer_sharing "${layer_sharing}" \
  > "${log_file}" 2>&1 &
