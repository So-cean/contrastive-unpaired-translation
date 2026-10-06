#!/usr/bin/env bash
# Portable single-process CUT recipe. Trailing arguments override defaults.
set -euo pipefail
: "${DATA_ROOT:?Set DATA_ROOT to a directory containing trainA/ and trainB/}"
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
exec python train.py \
  --dataroot "$DATA_ROOT" --name "${EXPERIMENT_NAME:-medical_cut}" \
  --model cut --dataset_mode monai --direction AtoB \
  --input_nc 1 --output_nc 1 --netG resnet_9blocks --netD basic \
  --pixel_dim 1 1 -1 --batch_size 1 --num_threads 0 \
  --display_id 0 --no_html --no_flip \
  --lambda_SSIM 0 --lambda_canny 0 --lambda_elastic 0 --lambda_perceptual 0 \
  "$@"
