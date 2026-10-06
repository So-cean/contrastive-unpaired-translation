#!/usr/bin/env bash
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to your NIfTI dataset directory}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec python inference.py \
  --dataroot "$DATA_ROOT" --name "${EXPERIMENT_NAME:-medical_cut}" \
  --model cut --dataset_mode monai --input_nc 1 --output_nc 1 \
  --netG resnet_9blocks --pixel_dim 1 1 -1 --phase test "$@"
