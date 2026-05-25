import os
import random
import thop
import torch

from models import transnet
from utils import logger, line_seg

__all__ = ["seed_everything", "init_device", "init_model",
           "freeze_component", "lora_component", "show_parameter"]


def seed_everything(seed):
    logger.info(f"Random seed set to {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_component(model, components):
    if not components:
        return

    component_map = {
        "encoder_self_attn": [model.encoder.layer.self_attn],
        "encoder_ffn": [model.encoder.layer.linear1, model.encoder.layer.linear2],
        "decoder_self_attn": [model.decoder.layer.self_attn],
        "decoder_cross_attn": [model.decoder.layer.multihead_attn],
        "decoder_ffn": [model.decoder.layer.linear1, model.decoder.layer.linear2],
        "fc_encoder": [model.fc_encoder],
        "fc_decoder": [model.fc_decoder],
    }

    for component in components:
        if component not in component_map:
            raise ValueError(
                f"Unknown freeze component '{component}'. "
                f"Valid choices: {list(component_map.keys())}"
            )
        for module in component_map[component]:
            for param in module.parameters():
                param.requires_grad = False

    # Verify: count frozen vs trainable
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    logger.info(
        f"=> Frozen components: {', '.join(components)} "
        f"({frozen_count}/{total_count} params frozen)"
    )


def lora_component(model, components, rank, alpha):
    if not components:
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "peft is required for LoRA. "
            "Install peft (and compatible torch/transformers) to use --lora_component."
        ) from exc

    component_map = {
        "encoder_self_attn": {
            "modules": ["encoder.layer.self_attn.out_proj"],
            "parameters": ["encoder.layer.self_attn.in_proj_weight"],
        },
        "encoder_ffn": {
            "modules": ["encoder.layer.linear1", "encoder.layer.linear2"],
            "parameters": [],
        },
        "decoder_self_attn": {
            "modules": ["decoder.layer.self_attn.out_proj"],
            "parameters": ["decoder.layer.self_attn.in_proj_weight"],
        },
        "decoder_cross_attn": {
            "modules": ["decoder.layer.multihead_attn.out_proj"],
            "parameters": ["decoder.layer.multihead_attn.in_proj_weight"],
        },
        "decoder_ffn": {
            "modules": ["decoder.layer.linear1", "decoder.layer.linear2"],
            "parameters": [],
        },
        "fc_encoder": {
            "modules": ["fc_encoder"],
            "parameters": [],
        },
        "fc_decoder": {
            "modules": ["fc_decoder"],
            "parameters": [],
        },
    }
    target_modules = []
    target_parameters = []
    for component in components:
        if component not in component_map:
            raise ValueError(
                f"Unknown LoRA component '{component}'. "
                f"Valid choices: {list(component_map.keys())}"
            )
        target_modules.extend(component_map[component]["modules"])
        target_parameters.extend(component_map[component]["parameters"])

    target_modules = list(dict.fromkeys(target_modules))
    target_parameters = list(dict.fromkeys(target_parameters))

    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        target_parameters=target_parameters,
    )
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    logger.info(
        f"=> LoRA enabled on components: {', '.join(components)}; "
        f"target_modules={target_modules}; "
        f"target_parameters={target_parameters} "
        f"(rank={rank}, alpha={alpha}); "
        f"{trainable_count}/{total_count} params trainable"
    )
    return model


def show_parameter(model):
    fmt_str = "{:<65} {:<8} {}"
    lines = []
    
    # 收集所有参数信息
    for name, param in model.named_parameters():
        lines.append(fmt_str.format(name, str(param.requires_grad), str(tuple(param.shape))))
    
    # 加上结尾的分隔线
    lines.append(line_seg)
    
    # 用换行符拼接成一个完整的字符串，只调用一次 logger.info
    logger.info("\n" + "\n".join(lines))


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        seed_everything(seed)

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        pin_memory = True
        logger.info("Running on GPU %d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    model = transnet(reduction=args.cr,
                     d_model=args.d_model,
                     channel=args.channel,
                     nt=args.nt,
                     nc=args.nc,
                     dim_feedforward=args.dim_feedforward)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained, weights_only=False, map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict,strict=False)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    H_a = torch.randn([1, args.channel, args.nt, args.nc])
    flops, params = thop.profile(model, inputs=(H_a,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.4e")

    # Model info logging
    logger.info(f'=> Model Name: TransNet [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}; '
                f'input shape=({args.channel}, {args.nt}, {args.nc}); '
                f'input dim={args.channel * args.nt * args.nc}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'\n{line_seg}\n{model}\n{line_seg}\n')

    if args.lora_component:
        if args.freeze_components:
            logger.warning(
                "freeze_components is ignored because LoRA is enabled."
            )
        model = lora_component(model, args.lora_component,
                               args.lora_rank, args.lora_alpha)
    elif args.freeze_components:
        freeze_component(model, args.freeze_components)
    
    show_parameter(model)

    return model
