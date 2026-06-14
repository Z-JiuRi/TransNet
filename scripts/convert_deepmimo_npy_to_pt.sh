#!/usr/bin/env bash
set -euo pipefail

python_bin=${python_bin:-/home/hujiacong/zxd/.envs/miniconda3/envs/torch/bin/python}
src_root=${src_root:-/storage/hujiacong/zxd/datasets/deepmimo/data}
dst_root=${dst_root:-/home/hujiacong/zxd/Huawei/TransNet/data/DeepMIMO}
workers=${workers:-2}
overwrite=${overwrite:-0}
dry_run=${dry_run:-0}
validate_existing=${validate_existing:-size}
zip_serialization=${zip_serialization:-0}
keep_tmp=${keep_tmp:-0}
min_free_ratio=${min_free_ratio:-1.05}

args=(
  --src_root "${src_root}"
  --dst_root "${dst_root}"
  --workers "${workers}"
  --validate_existing "${validate_existing}"
  --min_free_ratio "${min_free_ratio}"
)

if [[ "${overwrite}" == "1" ]]; then
  args+=(--overwrite)
fi

if [[ "${dry_run}" == "1" ]]; then
  args+=(--dry_run)
fi

if [[ "${zip_serialization}" == "1" ]]; then
  args+=(--zip_serialization)
fi

if [[ "${keep_tmp}" == "1" ]]; then
  args+=(--keep_tmp)
fi

"${python_bin}" scripts/convert_deepmimo_npy_to_pt.py "${args[@]}"
