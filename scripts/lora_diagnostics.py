#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import transnet
from utils.init import lora_component
from utils.scheduler import WarmUpCosineAnnealingLR


def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint
    return {"state_dict": checkpoint}


def make_model(args, device):
    model = transnet(
        reduction=args.cr,
        d_model=args.d_model,
        channel=args.channel,
        nt=args.nt,
        nc=args.nc,
        dim_feedforward=args.dim_feedforward,
        shared_layers=args.layer_sharing == "shared",
        transformer_backend=args.transformer_backend,
    )
    return model.to(device)


def load_base(args, device):
    model = make_model(args, device)
    checkpoint = load_checkpoint(args.base_checkpoint)
    result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"base load mismatch: {result}")
    return model


def load_lora_from_base(args, device, lora_checkpoint=None):
    model = load_base(args, device)
    model = lora_component(model, args.lora_component, args.lora_rank, args.lora_alpha)
    model.to(device)
    if lora_checkpoint is not None:
        checkpoint = load_checkpoint(lora_checkpoint)
        result = model.load_state_dict(checkpoint["state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"lora load mismatch: {result}")
    return model


def nmse_per_sample(pred, gt):
    power = gt[:, 0].square() + gt[:, 1].square()
    diff = gt - pred
    mse = diff[:, 0].square() + diff[:, 1].square()
    return 10 * torch.log10(mse.sum(dim=(1, 2)) / power.sum(dim=(1, 2)))


def evaluate(model, loader, device, base_model=None):
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_elems = 0
    nmse_values = []
    ratio_sum = 0.0
    ratio_count = 0
    delta_num = 0.0
    delta_den = 0.0

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device, dtype=torch.float32)
            pred = model(x)
            total_loss += criterion(pred, x).item()
            total_elems += x.numel()
            power = x[:, 0].square() + x[:, 1].square()
            diff = x - pred
            mse = diff[:, 0].square() + diff[:, 1].square()
            ratios = mse.sum(dim=(1, 2)) / power.sum(dim=(1, 2))
            ratio_sum += ratios.sum().item()
            ratio_count += ratios.numel()
            nmse_values.append(10 * torch.log10(ratios).cpu())
            if base_model is not None:
                base_pred = base_model(x)
                delta_num += (pred - base_pred).square().sum().item()
                delta_den += base_pred.square().sum().item()

    nmse = torch.cat(nmse_values)
    out = {
        "mse": total_loss / total_elems,
        "nmse_db_global": torch.log10(torch.tensor(ratio_sum / ratio_count)).mul(10).item(),
        "nmse_db_mean": nmse.mean().item(),
        "nmse_db_median": nmse.median().item(),
        "nmse_db_p05": nmse.quantile(0.05).item(),
        "nmse_db_p95": nmse.quantile(0.95).item(),
    }
    if base_model is not None:
        out["pred_delta_rel_l2"] = (delta_num / delta_den) ** 0.5 if delta_den else 0.0
    return out


def lora_stats(model):
    trainable_norm_sq = 0.0
    lora_norm_sq = 0.0
    lora_abs_max = 0.0
    trainable_scalars = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_norm_sq += param.detach().float().square().sum().item()
            trainable_scalars += param.numel()
        if "lora_" in name:
            data = param.detach().float()
            lora_norm_sq += data.square().sum().item()
            lora_abs_max = max(lora_abs_max, data.abs().max().item())
    return {
        "trainable_scalars": trainable_scalars,
        "trainable_l2": trainable_norm_sq ** 0.5,
        "lora_l2": lora_norm_sq ** 0.5,
        "lora_abs_max": lora_abs_max,
    }


def print_metrics(label, metrics, stats=None):
    parts = [
        label,
        f"mse={metrics['mse']:.6e}",
        f"nmse_global={metrics['nmse_db_global']:.4f}dB",
        f"nmse_mean={metrics['nmse_db_mean']:.4f}dB",
        f"nmse_median={metrics['nmse_db_median']:.4f}dB",
        f"p05={metrics['nmse_db_p05']:.4f}dB",
        f"p95={metrics['nmse_db_p95']:.4f}dB",
    ]
    if "pred_delta_rel_l2" in metrics:
        parts.append(f"delta_vs_base={metrics['pred_delta_rel_l2']:.6e}")
    if stats:
        parts.extend([
            f"trainable={stats['trainable_scalars']}",
            f"lora_l2={stats['lora_l2']:.6e}",
            f"lora_abs_max={stats['lora_abs_max']:.6e}",
        ])
    print(" | ".join(parts), flush=True)


def make_loader(data, batch_size, shuffle=False):
    return DataLoader(
        TensorDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_eval(args, device, test_loader):
    base = load_base(args, device)
    base_metrics = evaluate(base, test_loader, device)
    print_metrics("base", base_metrics)

    lora_epoch0 = load_lora_from_base(args, device)
    epoch0_metrics = evaluate(lora_epoch0, test_loader, device, base_model=base)
    print_metrics("lora_epoch0", epoch0_metrics, lora_stats(lora_epoch0))

    if args.lora_checkpoint:
        lora_trained = load_lora_from_base(args, device, args.lora_checkpoint)
        trained_metrics = evaluate(lora_trained, test_loader, device, base_model=base)
        print_metrics("lora_checkpoint", trained_metrics, lora_stats(lora_trained))


def run_train_probe(args, device, train_data, test_loader):
    base = load_base(args, device)
    base.eval()
    model = load_lora_from_base(args, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_init,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(train_data, args.batch_size, shuffle=True)
    scheduler = WarmUpCosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.scheduler_epochs * len(train_loader),
        T_warmup=0.1 * args.scheduler_epochs * len(train_loader),
        eta_min=args.eta_min,
    )
    criterion = nn.MSELoss()

    print_metrics(
        "probe_epoch=0",
        evaluate(model, test_loader, device, base_model=base),
        lora_stats(model),
    )

    for epoch in range(1, args.probe_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (x,) in train_loader:
            x = x.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, x)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model, test_loader, device, base_model=base)
        stats = lora_stats(model)
        lr = optimizer.param_groups[0]["lr"]
        label = f"probe_epoch={epoch} train_mse={total_loss / n_batches:.6e} lr={lr:.6e}"
        print_metrics(label, metrics, stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose TransNet LoRA checkpoints")
    parser.add_argument("--train-data", default="data/scenario_2/01109/train.pt")
    parser.add_argument("--test-data", default="data/scenario_2/01109/test.pt")
    parser.add_argument("--base-checkpoint", default="exps/WAIRD/seed797/base/torch_shared/checkpoints/best_nmse.pth")
    parser.add_argument("--lora-checkpoint", default=None,
                        help="optional checkpoint trained with the handwritten LoRA implementation")
    parser.add_argument("--lora-component", nargs="+", default=["encoder_ffn"],
                        choices=["encoder_ffn", "decoder_ffn"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--channel", type=int, default=2)
    parser.add_argument("--nt", type=int, default=64)
    parser.add_argument("--nc", type=int, default=64)
    parser.add_argument("--cr", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--layer-sharing", choices=["shared", "independent"], default="shared")
    parser.add_argument("--transformer-backend", choices=["torch", "original"], default="torch")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--disable-fastpath", action="store_true",
                        help="disable torch MHA/Transformer fast path; needed for torch encoder_ffn LoRA eval")
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--scheduler-epochs", type=int, default=400)
    parser.add_argument("--lr-init", type=float, default=1e-3)
    parser.add_argument("--eta-min", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.disable_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    torch.manual_seed(797)

    train_data = load_tensor(args.train_data)
    test_data = load_tensor(args.test_data)
    print("config:", json.dumps(vars(args), sort_keys=True))
    print(f"train_shape={tuple(train_data.shape)} test_shape={tuple(test_data.shape)} device={device}")
    expected_tail = (args.channel, args.nt, args.nc)
    if tuple(train_data.shape[1:]) != expected_tail or tuple(test_data.shape[1:]) != expected_tail:
        raise ValueError(
            f"data shape mismatch: expected tail {expected_tail}, "
            f"got train {tuple(train_data.shape[1:])}, test {tuple(test_data.shape[1:])}"
        )

    test_loader = make_loader(test_data, args.batch_size, shuffle=False)
    run_eval(args, device, test_loader)
    if not args.eval_only:
        run_train_probe(args, device, train_data, test_loader)


if __name__ == "__main__":
    main()
