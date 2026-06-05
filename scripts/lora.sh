#!/bin/bash

# ==============================================================================
# 1. 基础路径
# ==============================================================================
scen_name=${scen_name:-"scenario_2/00247"}
train_path=${train_path:-data/${scen_name}/train.pt}
val_path=${val_path:-data/${scen_name}/test.pt}
test_path=${test_path:-data/${scen_name}/test.pt}


# ==============================================================================
# 2. 模型结构与数据维度参数
# ==============================================================================
d_model=${d_model:-64}
nt=${nt:-64}
nc=${nc:-64}
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
batch_size=${batch_size:-64}
workers=${workers:-0}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-2e-4}
weight_decay=${weight_decay:-0}
gpu=${gpu:-2}
seed=${seed:-0}
exp_name=${exp_name:-WAIRD/seed${seed}/${scen_name}}
pretrained=${pretrained:-exps/WAIRD/seed0/arch_no_lora/original_shared/checkpoints/best_nmse.pth}

# ==============================================================================
# 5. 运行 Python 脚本
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
  --seed "${seed}" \
  --pretrained "${pretrained}" \
  --lora_component ${lora_component} \
  --lora_rank "${lora_rank}" \
  --lora_alpha "${lora_alpha}"
