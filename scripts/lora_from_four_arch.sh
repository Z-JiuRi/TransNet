#!/bin/bash

set -euo pipefail

# ==============================================================================
# 1. Base checkpoints and target data
# ==============================================================================
base_root=${base_root:-exps/WAIRD/seed0/arch_no_lora}
scen_name=${scen_name:-scenario_2/00032}
train_path=${train_path:-data/${scen_name}/train.pt}
val_path=${val_path:-data/${scen_name}/test.pt}
test_path=${test_path:-data/${scen_name}/test.pt}

# ==============================================================================
# 2. Model structure and data shape
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

# ==============================================================================
# 3. LoRA settings
# ==============================================================================
lora_rank=${lora_rank:-8}
lora_alpha=${lora_alpha:-16}
lora_components=${lora_components:-"decoder_ffn"}

# ==============================================================================
# 4. Training and runtime settings
# ==============================================================================
epochs=${epochs:-200}
batch_size=${batch_size:-64}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-5e-4}
weight_decay=${weight_decay:-0}
gpu=${gpu:-1}
seed=${seed:-0}
exp_root=${exp_root:-WAIRD/seed${seed}/${scen_name}/lora_from_four_arch}

run_one() {
  local transformer_backend=$1
  local layer_sharing=$2
  local lora_component=$3
  local gpu=$4
  local arch_name="${transformer_backend}_${layer_sharing}"
  local pretrained="${base_root}/${arch_name}/checkpoints/best_nmse.pth"
  local exp_name="${exp_root}/${arch_name}/${lora_component}"

  if [[ ! -f "${pretrained}" ]]; then
    echo "Missing base checkpoint: ${pretrained}" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Running base=${arch_name}, lora_component=${lora_component}"
  echo "Pretrained: ${pretrained}"
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
    --pretrained "${pretrained}" \
    --transformer_backend "${transformer_backend}" \
    --layer_sharing "${layer_sharing}" \
    --lora_component "${lora_component}" \
    --lora_rank "${lora_rank}" \
    --lora_alpha "${lora_alpha}" \
    > logs/${arch_name}_${lora_component}.log 2>&1 &
}

for lora_component in ${lora_components}; do
  run_one torch shared "${lora_component}" 1
  run_one torch independent "${lora_component}" 1
  run_one original shared "${lora_component}" 2
  run_one original independent "${lora_component}" 2

done
