scen_name=scenario_2/01109

python ./main.py \
  --exp_name WAIRD/seed797/test\
  --train_path /storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/train.pt \
  --val_path /storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/test.pt \
  --test_path /storage/hujiacong/zxd/datasets/WAIRD/data/${scen_name}/test.pt \
  --d_model 64 \
  --nt 64 \
  --nc 64 \
  --dim_feedforward 2048 \
  --batch_size 32 \
  --workers 0 \
  --cr 4 \
  --scheduler cosine \
  --lr_init 2e-4 \
  --weight_decay 1e-3 \
  --gpu 0 \
  --seed 797 \
  --pretrained /storage/hujiacong/zxd/Huawei/TransNet/exps/WAIRD/seed797/base/torch_shared/checkpoints/best_nmse.pth \
  --evaluate