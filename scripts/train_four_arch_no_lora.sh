#!/bin/bash

scen_name=base \
epochs=400 \
nt=64 \
nc=64 \
dim_feedforward=2048 \
seed=797 \
gpu=3 \
transformer_backend=torch \
layer_sharing=shared \
bash scripts/train.sh

scen_name=base \
epochs=400 \
nt=64 \
nc=64 \
dim_feedforward=2048 \
seed=797 \
gpu=3 \
transformer_backend=torch \
layer_sharing=independent \
bash scripts/train.sh

scen_name=base \
epochs=400 \
nt=64 \
nc=64 \
dim_feedforward=2048 \
seed=797 \
gpu=3 \
transformer_backend=original \
layer_sharing=shared \
bash scripts/train.sh

scen_name=base \
epochs=400 \
nt=64 \
nc=64 \
dim_feedforward=2048 \
seed=797 \
gpu=3 \
transformer_backend=original \
layer_sharing=independent \
bash scripts/train.sh