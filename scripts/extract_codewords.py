import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.TransNet import transnet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract TransNet compressed codewords for CSI tensors."
    )
    parser.add_argument("--data", type=str, required=True,
                        help="path to input .pt data with shape (N, channel, nt, nc)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to trained checkpoint")
    parser.add_argument("--output", type=str, required=True,
                        help="path to save codewords .pt tensor")
    parser.add_argument("-b", "--batch_size", type=int, default=200,
                        help="mini-batch size for encoding")
    parser.add_argument("--cpu", action="store_true",
                        help="force CPU inference")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id to use")

    parser.add_argument("--cr", type=int, default=4,
                        help="compression ratio denominator used by checkpoint")
    parser.add_argument("--channel", type=int, default=2,
                        help="number of input channels")
    parser.add_argument("--nt", type=int, default=32,
                        help="number of antennas in the CSI tensor")
    parser.add_argument("--nc", type=int, default=32,
                        help="number of delay/frequency bins in the CSI tensor")
    parser.add_argument("-d", "--d_model", type=int, default=64,
                        help="Transformer feature dimension")
    parser.add_argument("--dim_feedforward", type=int, default=2048,
                        help="hidden dimension of Transformer feed-forward layers")

    return parser.parse_args()


def load_data(path, channel, nt, nc):
    data = torch.as_tensor(
        torch.load(path, map_location="cpu", weights_only=False),
        dtype=torch.float32,
    )
    expected_tail = (channel, nt, nc)
    if data.dim() != 4 or tuple(data.shape[1:]) != expected_tail:
        raise ValueError(
            f"Input data must have shape (N, {channel}, {nt}, {nc}), "
            f"but got {tuple(data.shape)}."
        )
    return data


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def main():
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    data = load_data(args.data, args.channel, args.nt, args.nc)
    loader = DataLoader(
        TensorDataset(data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = transnet(
        reduction=args.cr,
        d_model=args.d_model,
        channel=args.channel,
        nt=args.nt,
        nc=args.nc,
        dim_feedforward=args.dim_feedforward,
    )
    model.load_state_dict(load_checkpoint(args.checkpoint), strict=False)
    model.to(device)
    model.eval()

    codewords = []
    with torch.no_grad():
        for (sparse_gt,) in loader:
            sparse_gt = sparse_gt.to(device)
            codewords.append(model.encode(sparse_gt).cpu())

    codewords = torch.cat(codewords, dim=0)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(codewords, args.output)

    print(f"Saved codewords to {args.output}")
    print(f"Input shape: {tuple(data.shape)}")
    print(f"Codewords shape: {tuple(codewords.shape)}")


if __name__ == "__main__":
    main()
