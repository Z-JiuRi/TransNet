#!/usr/bin/env bash
set -euo pipefail

# Base TransNet evaluation.
#
# Example:
#   seed=797 scen_name=scenario_2/01109 bash scripts/test.sh

scen_name=${scen_name:-scenario_2/01109}
scen_tag=${scen_name//\//_}
data_root=${data_root:-/storage/hujiacong/zxd/datasets/WAIRD/data}
train_path=${train_path:-${data_root}/${scen_name}/train.pt}
val_path=${val_path:-${data_root}/${scen_name}/test.pt}
test_path=${test_path:-${data_root}/${scen_name}/test.pt}

d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-1e-3}
seed=${seed:-797}
gpu=${gpu:-0}
transformer_backend=${transformer_backend:-torch}
layer_sharing=${layer_sharing:-shared}
pretrained=${pretrained:-exps/WAIRD/seed${seed}/base/${transformer_backend}_${layer_sharing}/checkpoints/best_nmse.pth}
exp_name=${exp_name:-WAIRD/seed${seed}/test/${scen_tag}_${transformer_backend}_${layer_sharing}}

log_file="exps/${exp_name}/test.out"
mkdir -p "$(dirname "$log_file")"

python ./main.py \
  --exp_name "${exp_name}" \
  --train_path "${train_path}" \
  --val_path "${val_path}" \
  --test_path "${test_path}" \
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
  --seed "${seed}" \
  --pretrained "${pretrained}" \
  --transformer_backend "${transformer_backend}" \
  --layer_sharing "${layer_sharing}" \
  --evaluate \
  --gpu "${gpu}" \
  > "${log_file}" 2>&1
