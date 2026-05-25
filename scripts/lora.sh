#!/bin/bash

# ==============================================================================
# 1. 基础路径与实验名称（环境变量传参，带默认值）
# ==============================================================================
exp_name=${exp_name:-COST2100_bs32/in/seed0/lora_fc_encoder_fc_decoder}
train_path=${train_path:-/storage/hujiacong/zxd/datasets/cost2100/in_train.pt}
val_path=${val_path:-/storage/hujiacong/zxd/datasets/cost2100/in_val.pt}
test_path=${test_path:-/storage/hujiacong/zxd/datasets/cost2100/in_test.pt}
pretrained=${pretrained:-/storage/hujiacong/zxd/Huawei/TransNet/exps/COST2100_bs512/in/seed42/base/checkpoints/best_nmse.pth}

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
lora_component=${lora_component:-"fc_encoder fc_decoder"}
lora_rank=${lora_rank:-8}
lora_alpha=${lora_alpha:-16}

# ==============================================================================
# 4. 训练超参数与硬件设置
# ==============================================================================
epochs=${epochs:-400}
batch_size=${batch_size:-32}
workers=${workers:-4}
scheduler=${scheduler:-cosine}
lr_init=${lr_init:-1e-4}
weight_decay=${weight_decay:-0}
gpu=${gpu:-2}
seed=${seed:-0}

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
