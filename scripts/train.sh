#!/usr/bin/env bash
set -euo pipefail

# Base TransNet training.
#
# Example:
#   scen_name=base epochs=400 batch_size=256 seed=2002 gpu=0 bash scripts/train.sh

scen_name=${scen_name:-base}
data_root=${data_root:-/storage/hujiacong/zxd/datasets/WAIRD/data}
# data_root=${data_root:-/storage/hujiacong/zxd/datasets/deepmimo/data}
train_path=${train_path:-${data_root}/${scen_name}/train.pt}
val_path=${val_path:-${data_root}/${scen_name}/test.pt}
test_path=${test_path:-${data_root}/${scen_name}/test.pt}

d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

epochs=${epochs:-400}
batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
seed=${seed:-42}
gpu=${gpu:-0}
transformer_backend=${transformer_backend:-torch}
layer_sharing=${layer_sharing:-independent}
fc_lora=${fc_lora:-false}
fc_lora_rank=${fc_lora_rank:-512}
exp_name=${exp_name:-WAIRD/seed${seed}/${scen_name}/${transformer_backend}_${layer_sharing}}

log_file="exps/${exp_name}/train.out"
mkdir -p "$(dirname "$log_file")"

extra_args=()
# if [[ "${fc_lora}" == "true" || "${fc_lora}" == "1" ]]; then
#   extra_args+=(--fc_lora)
# fi
# if [[ -n "${fc_lora_rank}" ]]; then
#   extra_args+=(--fc_lora_rank "${fc_lora_rank}")
# fi

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
  "${extra_args[@]}" \
  > "${log_file}" 2>&1 &