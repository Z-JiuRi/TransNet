import os
import random
import numpy as np
import thop
import torch
import torch.nn as nn

from models import transnet
from utils import logger, line_seg

__all__ = ["seed_everything", "init_device", "init_model",
           "freeze_component", "LoRALinear", "lora_component",
           "show_parameter"]


def seed_everything(seed):
    logger.info(f"Random seed set to {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    elif hasattr(torch, "set_deterministic"):
        torch.set_deterministic(True)


def freeze_component(model, components):
    if not components:
        return

    encoder_layers = list(model.encoder.layers)
    decoder_layers = list(model.decoder.layers)
    component_map = {
        "encoder_self_attn": [layer.self_attn for layer in encoder_layers],
        "encoder_ffn": [module for layer in encoder_layers
                        for module in (layer.linear1, layer.linear2)],
        "decoder_self_attn": [layer.self_attn for layer in decoder_layers],
        "decoder_cross_attn": [layer.multihead_attn for layer in decoder_layers],
        "decoder_ffn": [module for layer in decoder_layers
                        for module in (layer.linear1, layer.linear2)],
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


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank, alpha):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"LoRALinear only supports nn.Linear, got {type(base_layer)}"
            )
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)

        for param in self.base_layer.parameters():
            param.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * self.scaling


def _replace_module(root, name, new_module):
    parent = root
    parts = name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def lora_component(model, components, rank, alpha):
    if not components:
        return model

    def modules_with(prefix, suffixes):
        return [
            name for name, _ in model.named_modules()
            if name.startswith(prefix) and name.endswith(suffixes)
        ]

    def modules_with_any(prefixes, suffixes):
        return [
            name for prefix in prefixes
            for name in modules_with(prefix, suffixes)
        ]

    component_map = {
        "encoder_ffn": {
            "modules": modules_with_any(
                ("encoder.layers.", "encoder.layer."),
                (".linear1", ".linear2")
            ),
        },
        "decoder_ffn": {
            "modules": modules_with_any(
                ("decoder.layers.", "decoder.layer."),
                (".linear1", ".linear2")
            ),
        },
    }
    target_modules = []
    for component in components:
        if component not in component_map:
            raise ValueError(
                f"Unknown LoRA component '{component}'. "
                f"Valid choices: {list(component_map.keys())}"
            )
        target_modules.extend(component_map[component]["modules"])

    target_modules = list(dict.fromkeys(target_modules))
    if not target_modules:
        raise ValueError(
            f"No FFN Linear modules matched for LoRA components: {components}"
        )

    for param in model.parameters():
        param.requires_grad = False

    for name in target_modules:
        module = dict(model.named_modules())[name]
        _replace_module(model, name, LoRALinear(module, rank, alpha))

    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)

    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"=> LoRA enabled on components: {', '.join(components)}; "
        f"target_modules={target_modules}; "
        f"(rank={rank}, alpha={alpha}); "
        f"{trainable_count}/{total_count} scalar params trainable"
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
                     dim_feedforward=args.dim_feedforward,
                     shared_layers=args.layer_sharing == 'shared',
                     transformer_backend=args.transformer_backend)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained, weights_only=False, map_location=torch.device('cpu'))['state_dict']        
        load_result = model.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys:
            logger.warning(
                "Missing keys when loading pretrained weights ({}): {}".format(
                    len(load_result.missing_keys),
                    load_result.missing_keys
                )
            )
        if load_result.unexpected_keys:
            logger.warning(
                "Unexpected keys in pretrained weights ({}): {}".format(
                    len(load_result.unexpected_keys),
                    load_result.unexpected_keys
                )
            )
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    if args.lora_component:
        if args.freeze_components:
            logger.warning(
                "freeze_components is ignored because LoRA is enabled."
            )
        model = lora_component(model, args.lora_component,
                               args.lora_rank, args.lora_alpha)
        if args.lora_pretrained is not None:
            assert os.path.isfile(args.lora_pretrained)
            state_dict = torch.load(
                args.lora_pretrained,
                weights_only=False,
                map_location=torch.device('cpu')
            )['state_dict']
            model.load_state_dict(state_dict, strict=True)
            logger.info("LoRA weights loaded from {}".format(args.lora_pretrained))
    elif args.lora_pretrained is not None:
        raise ValueError("--lora_pretrained requires --lora_component")
    elif args.freeze_components:
        freeze_component(model, args.freeze_components)
    
    # Model flops and params counting
    H_a = torch.randn([1, args.channel, args.nt, args.nc])
    flops, params = thop.profile(model, inputs=(H_a,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.4e")

    # Model info logging
    logger.info(f'=> Model Name: TransNet [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}; '
                f'input shape=({args.channel}, {args.nt}, {args.nc}); '
                f'input dim={args.channel * args.nt * args.nc}; '
                f'layer sharing={args.layer_sharing}; '
                f'transformer backend={args.transformer_backend}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'\n{line_seg}\n{model}\n{line_seg}\n')
    
    show_parameter(model)

    return model
