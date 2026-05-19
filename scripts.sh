python ./main.py \
  --train-path /home/z-jiuri/workspace/Huawei/TransNet/COST2100/in_train.pt \
  --val-path /home/z-jiuri/workspace/Huawei/TransNet/COST2100/in_val.pt \
  --test-path /home/z-jiuri/workspace/Huawei/TransNet/COST2100/in_test.pt \
  --epochs 400 \
  --d_model 64 \
  --nt 32 \
  --nc 32 \
  --dim-feedforward 2048 \
  --batch-size 200 \
  --workers 4 \
  --cr 4 \
  --scheduler cosine \
  --gpu 0 \
  2>&1 | tee log.out
