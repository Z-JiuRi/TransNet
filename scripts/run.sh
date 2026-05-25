exp_name=COST2100_bs32/out/seed3407/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/out_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/out_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/out_test.pt \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
gpu=0 \
seed=3407 \
./scripts/train.sh > out_3407.log 2>&1 &

exp_name=COST2100_bs32/out/seed42/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/out_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/out_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/out_test.pt \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
gpu=1 \
seed=42 \
./scripts/train.sh > out_42.log 2>&1 &

exp_name=COST2100_bs32/out/seed2026/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/out_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/out_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/out_test.pt \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
gpu=2 \
seed=2026 \
./scripts/train.sh > out_2026.log 2>&1 &

exp_name=COST2100_bs32/in/seed42/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
gpu=3 \
seed=42 \
./scripts/train.sh > in_42.log 2>&1 &

exp_name=COST2100_bs32/in/seed3407/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
gpu=4 \
seed=3407 \
./scripts/train.sh > in_3407.log 2>&1 &

exp_name=COST2100_bs32/in/seed2026/base \
train_path=/storage/hujiacong/zxd/datasets/cost2100/in_train.pt \
val_path=/storage/hujiacong/zxd/datasets/cost2100/in_val.pt \
test_path=/storage/hujiacong/zxd/datasets/cost2100/in_test.pt \
batch_size=32 \
epochs=400 \
lr_init=2e-4 \
gpu=5 \
seed=2026 \
./scripts/train.sh > in_2026.log 2>&1 &








