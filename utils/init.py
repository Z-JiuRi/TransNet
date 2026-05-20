import os
import random
import thop
import torch

from models import transnet
from utils import logger, line_seg

__all__ = ["seed_everything", "init_device", "init_model",
           "freeze_component", "show_parameter"]


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


def show_parameter(model):
    logger.info(f'\n{line_seg}\n=> Parameter trainable status\n{line_seg}')
    logger.info(f"name\ttrainable\tshape")
    for name, param in model.named_parameters():
        logger.info(f"{name}\t{param.requires_grad}\t{tuple(param.shape)}")
    logger.info(line_seg)


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
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict,strict=False)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting
    H_a = torch.randn([1, args.channel, args.nt, args.nc])
    flops, params = thop.profile(model, inputs=(H_a,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    logger.info(f'=> Model Name: TransNet [pretrained: {args.pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{args.cr}; '
                f'input shape=({args.channel}, {args.nt}, {args.nc}); '
                f'input dim={args.channel * args.nt * args.nc}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'\n{line_seg}\n{model}\n{line_seg}\n')

    if args.freeze_components:
        freeze_component(model, args.freeze_components)
    
    show_parameter(model)

    return model