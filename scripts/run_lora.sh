#!/bin/bash

pretrained=/storage/hujiacong/zxd/Huawei/TransNet/exps/COST2100_bs512/in/seed42/base/checkpoints/best_nmse.pth

if [ ! -f "${pretrained}" ]; then
  echo "Missing pretrained checkpoint: ${pretrained}"
  echo "Run scripts/run.sh first, or set pretrained to a finished post-fix bs512 base checkpoint."
  exit 1
fi

exp_name=COST2100_lora/in/seed0/fc_both_r8_a8_lr2e-4 \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
pretrained="${pretrained}" \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
weight_decay=0 \
gpu=0 \
seed=0 \
lora_component="fc_encoder fc_decoder" \
lora_rank=8 \
lora_alpha=8 \
./scripts/lora.sh > lora_fc_both_r8_a8_lr2e-4_in_seed0.log 2>&1 &

exp_name=COST2100_lora/in/seed0/fc_both_r8_a16_lr2e-4 \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
pretrained="${pretrained}" \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
weight_decay=0 \
gpu=1 \
seed=0 \
lora_component="fc_encoder fc_decoder" \
lora_rank=8 \
lora_alpha=16 \
./scripts/lora.sh > lora_fc_both_r8_a16_lr2e-4_in_seed0.log 2>&1 &

exp_name=COST2100_lora/in/seed0/fc_both_r16_a16_lr2e-4 \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
pretrained="${pretrained}" \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
weight_decay=0 \
gpu=2 \
seed=0 \
lora_component="fc_encoder fc_decoder" \
lora_rank=16 \
lora_alpha=16 \
./scripts/lora.sh > lora_fc_both_r16_a16_lr2e-4_in_seed0.log 2>&1 &



exp_name=COST2100_lora/in/seed0/fc_both_r16_a32_lr2e-4 \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
pretrained="${pretrained}" \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
weight_decay=0 \
gpu=3 \
seed=0 \
lora_component="fc_encoder fc_decoder" \
lora_rank=16 \
lora_alpha=32 \
./scripts/lora.sh > lora_fc_both_r16_a32_lr2e-4_in_seed0.log 2>&1 &

exp_name=COST2100_lora/in/seed0/fc_both_r8_a16_lr3e-4 \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
pretrained="${pretrained}" \
batch_size=32 \
epochs=400 \
lr_init=3e-4 \
weight_decay=0 \
gpu=4 \
seed=0 \
lora_component="fc_encoder fc_decoder" \
lora_rank=8 \
lora_alpha=16 \
./scripts/lora.sh > lora_fc_both_r8_a16_lr3e-4_in_seed0.log 2>&1 &

exp_name=COST2100_lora/in/seed0/fc_both_r16_a32_lr3e-4 \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
pretrained="${pretrained}" \
batch_size=32 \
epochs=400 \
lr_init=3e-4 \
weight_decay=0 \
gpu=5 \
seed=0 \
lora_component="fc_encoder fc_decoder" \
lora_rank=16 \
lora_alpha=32 \
./scripts/lora.sh > lora_fc_both_r16_a32_lr3e-4_in_seed0.log 2>&1 &
