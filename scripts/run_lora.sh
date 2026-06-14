# #!/bin/bash

# lora_components=(fc_encoder)

# for lora_component in ${lora_components[@]}; do
#     seed=797 \
#     gpu=0 \
#     scen_name=scenario_2/01105 \
#     lora_component=${lora_component} \
#     lora_rank=256 \
#     lora_alpha=1024 \
#     epochs=200 \
#     batch_size=32 \
#     transformer_backend=torch \
#     layer_sharing=shared \
#     bash scripts/train_lora.sh

#     seed=797 \
#     gpu=3 \
#     scen_name=scenario_2/01105 \
#     lora_component=${lora_component} \
#     lora_rank=256 \
#     lora_alpha=1024 \
#     epochs=200 \
#     batch_size=32 \
#     transformer_backend=torch \
#     layer_sharing=independent \
#     bash scripts/train_lora.sh

#     seed=797 \
#     gpu=4 \
#     scen_name=scenario_2/01105 \
#     lora_component=${lora_component} \
#     lora_rank=256 \
#     lora_alpha=1024 \
#     epochs=200 \
#     batch_size=32 \
#     transformer_backend=original \
#     layer_sharing=shared \
#     bash scripts/train_lora.sh

#     seed=797 \
#     gpu=5 \
#     scen_name=scenario_2/01105 \
#     lora_component=${lora_component} \
#     lora_rank=256 \
#     lora_alpha=1024 \
#     epochs=200 \
#     batch_size=32 \
#     transformer_backend=original \
#     layer_sharing=independent \
#     bash scripts/train_lora.sh

#     echo "lora_component: ${lora_component}, waiting 1min"
#     sleep 60
# done


scen_name=city_76_houston_3p5_part_14 lora_component=fc_decoder lora_rank=8 lora_alpha=16 gpu=5 seed=42 bash scripts/lora.sh
scen_name=city_76_houston_3p5_part_14 lora_component=fc_encoder lora_rank=8 lora_alpha=16 gpu=4 seed=42 bash scripts/lora.sh
scen_name=city_76_houston_3p5_part_14 lora_component=decoder_ffn lora_rank=8 lora_alpha=16 gpu=4 seed=42 bash scripts/lora.sh
scen_name=city_76_houston_3p5_part_14 lora_component=encoder_ffn lora_rank=8 lora_alpha=16 gpu=5 seed=42 bash scripts/lora.sh