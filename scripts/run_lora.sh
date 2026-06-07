#!/bin/bash

lora_components=(encoder_ffn decoder_ffn)

for lora_component in ${lora_components[@]}; do
    scen_name=scenario_2/01109 \
    lora_component=${lora_component} \
    lora_rank=8 \
    lora_alpha=16 \
    epochs=400 \
    nt=32 \
    nc=32 \
    dim_feedforward=2048 \
    seed=796 \
    gpu=0 \
    lr_init=1e-3 \
    transformer_backend=torch \
    layer_sharing=shared \
    bash scripts/lora.sh

    scen_name=scenario_2/01109 \
    lora_component=${lora_component} \
    lora_rank=8 \
    lora_alpha=16 \
    epochs=400 \
    nt=32 \
    nc=32 \
    dim_feedforward=2048 \
    seed=796 \
    gpu=3 \
    lr_init=1e-3 \
    transformer_backend=torch \
    layer_sharing=independent \
    bash scripts/lora.sh

    scen_name=scenario_2/01109 \
    lora_component=${lora_component} \
    lora_rank=8 \
    lora_alpha=16 \
    epochs=400 \
    nt=32 \
    nc=32 \
    dim_feedforward=2048 \
    seed=796 \
    gpu=4 \
    lr_init=1e-3 \
    transformer_backend=original \
    layer_sharing=shared \
    bash scripts/lora.sh

    scen_name=scenario_2/01109 \
    lora_component=${lora_component} \
    lora_rank=8 \
    lora_alpha=16 \
    epochs=400 \
    nt=32 \
    nc=32 \
    dim_feedforward=2048 \
    seed=796 \
    gpu=5 \
    lr_init=1e-3 \
    transformer_backend=original \
    layer_sharing=independent \
    bash scripts/lora.sh

    echo "lora_component: ${lora_component}, waiting 1min"
    sleep 60
done
