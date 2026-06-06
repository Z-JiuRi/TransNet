#!/bin/bash

# transformer_backend=torch layer_sharing=shared gpu=2 bash scripts/train_on_base_pretrained.sh

# 配置参数
scen_name=${scen_name:-"scenario_2/00032"}  # 要微调的场景
train_path=${train_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/test.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/test.pt}

d_model=${d_model:-64}
nt=${nt:-32}
nc=${nc:-32}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

epochs=${epochs:-200}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}  # 微调的学习率可以稍小一些
weight_decay=${weight_decay:-1e-3}
gpu=${gpu:-5}
seed=${seed:-3232}
transformer_backend=${transformer_backend:-original}
layer_sharing=${layer_sharing:-independent}

# 预训练模型路径
pretrained=${pretrained:-/storage/hujiacong/zxd/Huawei/TransNet/exps/WAIRD/seed${seed}/base/${transformer_backend}_${layer_sharing}/checkpoints/best_nmse.pth}

# 实验名称
exp_name=${exp_name:-WAIRD/seed${seed}/${scen_name}_${transformer_backend}_${layer_sharing}_finetune_full}

mktouch() {
    mkdir -p "$(dirname "$1")" && touch "$1"
}

log_file="exps/${exp_name}/train.out"
mktouch "${log_file}"

echo "============================================================"
echo "Fine-tuning PRETRAINED base model on ${scen_name}"
echo "NO LoRA - full fine-tuning"
echo "Pretrained from: ${pretrained}"
echo "Backend=${transformer_backend}, layer_sharing=${layer_sharing}"
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
  --pretrained "${pretrained}" \
  --transformer_backend "${transformer_backend}" \
  --layer_sharing "${layer_sharing}" \
  > "${log_file}" 2>&1 &

echo "Full fine-tuning started in background. Log: ${log_file}"
