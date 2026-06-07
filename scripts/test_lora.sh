#!/usr/bin/env bash
set -euo pipefail

# Handwritten FFN LoRA evaluation.
#
# Example:
#   seed=797 scen_name=scenario_2/01109 lora_component=decoder_ffn bash scripts/test_lora.sh

scen_name=${scen_name:-scenario_2/01109}
scen_tag=${scen_name//\//_}
data_root=${data_root:-data}
train_path=${train_path:-${data_root}/${scen_name}/train.pt}
val_path=${val_path:-${data_root}/${scen_name}/test.pt}
test_path=${test_path:-${data_root}/${scen_name}/test.pt}

d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
dim_feedforward=${dim_feedforward:-2048}
cr=${cr:-4}

lora_component=${lora_component:-decoder_ffn}
case "$lora_component" in
  encoder_ffn|decoder_ffn) ;;
  *)
    echo "lora_component must be encoder_ffn or decoder_ffn, got: ${lora_component}" >&2
    exit 2
    ;;
esac
lora_rank=${lora_rank:-8}
lora_alpha=${lora_alpha:-16}

batch_size=${batch_size:-256}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-1e-3}
weight_decay=${weight_decay:-0}
seed=${seed:-797}
gpu=${gpu:-0}
transformer_backend=${transformer_backend:-torch}
layer_sharing=${layer_sharing:-shared}
pretrained=${pretrained:-exps/WAIRD/seed${seed}/base/${transformer_backend}_${layer_sharing}/checkpoints/best_nmse.pth}
lora_pretrained=${lora_pretrained:-exps/WAIRD/seed${seed}/${scen_name}_${lora_component}_${lora_rank}_${lora_alpha}_${transformer_backend}_${layer_sharing}/checkpoints/best_nmse.pth}
exp_name=${exp_name:-WAIRD/seed${seed}/test_lora/${scen_tag}_${lora_component}_${lora_rank}_${lora_alpha}_${transformer_backend}_${layer_sharing}}

log_file="exps/${exp_name}/test.out"
mkdir -p "$(dirname "$log_file")"

cmd=(
  python ./main.py
  --exp_name "$exp_name"
  --train_path "$train_path"
  --val_path "$val_path"
  --test_path "$test_path"
  --d_model "$d_model"
  --nt "$nt"
  --nc "$nc"
  --dim_feedforward "$dim_feedforward"
  --batch_size "$batch_size"
  --workers "$workers"
  --cr "$cr"
  --scheduler "$scheduler"
  --lr_init "$lr_init"
  --weight_decay "$weight_decay"
  --seed "$seed"
  --pretrained "$pretrained"
  --lora_component "$lora_component"
  --lora_rank "$lora_rank"
  --lora_alpha "$lora_alpha"
  --lora_pretrained "$lora_pretrained"
  --transformer_backend "$transformer_backend"
  --layer_sharing "$layer_sharing"
  --evaluate
  --gpu "$gpu"
)

printf 'Running LoRA eval: %s\n' "$exp_name"
printf 'Base checkpoint: %s\n' "$pretrained"
printf 'LoRA checkpoint: %s\n' "$lora_pretrained"
printf 'Log: %s\n' "$log_file"
"${cmd[@]}" 2>&1 & | tee "$log_file"
