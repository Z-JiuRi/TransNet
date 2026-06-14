#!/usr/bin/env python3
import argparse
import os
import signal
import shutil
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


signal.signal(signal.SIGPIPE, signal.SIG_DFL)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DeepMIMO .npy files to float32 .pt files while preserving directory structure."
    )
    parser.add_argument(
        "--src_root",
        type=Path,
        default=Path("/storage/hujiacong/zxd/datasets/deepmimo/data"),
        help="Root directory containing .npy files.",
    )
    parser.add_argument(
        "--dst_root",
        type=Path,
        default=Path("/home/hujiacong/zxd/Huawei/TransNet/data/DeepMIMO"),
        help="Root directory where .pt files will be saved.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of concurrent conversion workers. Keep this small for large files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files.",
    )
    parser.add_argument(
        "--validate_existing",
        choices=["size", "full", "none"],
        default="size",
        help=(
            "How to validate existing .pt files before skipping. "
            "'size' checks whether file size can hold the float32 tensor; "
            "'full' additionally torch.load()s the file and checks shape/dtype."
        ),
    )
    parser.add_argument(
        "--zip_serialization",
        action="store_true",
        help=(
            "Use PyTorch's default zip serialization. By default the script uses "
            "legacy serialization to avoid inline_container write-position errors "
            "seen on some filesystems with large tensors."
        ),
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep stale temporary files from previous interrupted runs.",
    )
    parser.add_argument(
        "--min_free_ratio",
        type=float,
        default=1.05,
        help=(
            "Require this multiple of the tensor byte size as free space before "
            "writing each temporary .pt file. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned conversions.",
    )
    return parser.parse_args()


def discover_files(src_root, dst_root):
    npy_files = sorted(src_root.rglob("*.npy"))
    return [
        (src_path, dst_root / src_path.relative_to(src_root).with_suffix(".pt"))
        for src_path in npy_files
    ]


def load_npy_as_float32_tensor(src_path):
    arr = np.load(src_path, mmap_mode="r")
    if arr.dtype != np.float32:
        arr = np.asarray(arr, dtype=np.float32)
    elif not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The given NumPy array is not writable",
            category=UserWarning,
        )
        tensor = torch.from_numpy(arr)
    return tensor


def expected_float32_nbytes(src_path):
    arr = np.load(src_path, mmap_mode="r")
    return int(np.prod(arr.shape)) * np.dtype(np.float32).itemsize, tuple(arr.shape)


def existing_file_status(src_path, dst_path, validate_existing):
    if not dst_path.exists():
        return False, "missing"
    if validate_existing == "none":
        return True, "exists"

    expected_nbytes, expected_shape = expected_float32_nbytes(src_path)
    actual_size = dst_path.stat().st_size
    if actual_size < expected_nbytes:
        return False, f"incomplete size={actual_size} < tensor_bytes={expected_nbytes}"
    if validate_existing == "size":
        return True, f"exists size={actual_size}"

    try:
        obj = torch.load(dst_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return False, f"load failed: {exc}"
    if not isinstance(obj, torch.Tensor):
        return False, f"not a tensor: {type(obj)}"
    if tuple(obj.shape) != expected_shape:
        return False, f"shape mismatch {tuple(obj.shape)} != {expected_shape}"
    if obj.dtype != torch.float32:
        return False, f"dtype mismatch {obj.dtype} != torch.float32"
    return True, f"valid shape={expected_shape} dtype={obj.dtype}"


def cleanup_stale_tmp(dst_root):
    removed = 0
    for tmp_path in dst_root.rglob(".*.pt.*.tmp"):
        tmp_path.unlink(missing_ok=True)
        removed += 1
    return removed


def check_free_space(dst_dir, required_bytes, min_free_ratio):
    if min_free_ratio <= 0:
        return
    free_bytes = shutil.disk_usage(dst_dir).free
    min_bytes = int(required_bytes * min_free_ratio)
    if free_bytes < min_bytes:
        raise OSError(
            f"not enough free space in {dst_dir}: "
            f"free={free_bytes} required>={min_bytes}"
        )


def convert_one(
    src_path,
    dst_path,
    overwrite=False,
    dry_run=False,
    validate_existing="size",
    zip_serialization=False,
    min_free_ratio=1.05,
):
    if dst_path.exists() and not overwrite:
        is_valid, detail = existing_file_status(src_path, dst_path, validate_existing)
        if is_valid:
            return "skip", src_path, dst_path, detail
        if dry_run:
            return "reconvert", src_path, dst_path, detail

    if dry_run:
        return "dry-run", src_path, dst_path, ""

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    tensor = load_npy_as_float32_tensor(src_path)
    check_free_space(dst_path.parent, tensor.numel() * tensor.element_size(),
                     min_free_ratio)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dst_path.name}.",
        suffix=".tmp",
        dir=str(dst_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(
            tensor.float(),
            tmp_path,
            _use_new_zipfile_serialization=zip_serialization,
        )
        os.replace(tmp_path, dst_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    elapsed = time.time() - started
    detail = f"shape={tuple(tensor.shape)} dtype={tensor.dtype} time={elapsed:.1f}s"
    return "done", src_path, dst_path, detail


def main():
    args = parse_args()
    if not args.src_root.is_dir():
        raise FileNotFoundError(f"src_root does not exist: {args.src_root}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    tasks = discover_files(args.src_root, args.dst_root)
    print(f"Found {len(tasks)} .npy files under {args.src_root}")
    print(f"Saving .pt files under {args.dst_root}")
    print(f"validate_existing={args.validate_existing}")
    print(f"serialization={'zip' if args.zip_serialization else 'legacy'}")
    if not args.keep_tmp and args.dst_root.exists() and not args.dry_run:
        removed = cleanup_stale_tmp(args.dst_root)
        print(f"Removed {removed} stale temporary files")

    counts = {"done": 0, "skip": 0, "dry-run": 0, "reconvert": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                convert_one,
                src_path,
                dst_path,
                args.overwrite,
                args.dry_run,
                args.validate_existing,
                args.zip_serialization,
                args.min_free_ratio,
            )
            for src_path, dst_path in tasks
        ]
        for future in as_completed(futures):
            try:
                status, src_path, dst_path, detail = future.result()
                counts[status] += 1
                rel_src = src_path.relative_to(args.src_root)
                rel_dst = dst_path.relative_to(args.dst_root)
                suffix = f" ({detail})" if detail else ""
                print(f"[{status}] {rel_src} -> {rel_dst}{suffix}", flush=True)
            except Exception as exc:
                counts["error"] += 1
                print(f"[error] {exc}", flush=True)

    print(
        "Summary: "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
    if counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
