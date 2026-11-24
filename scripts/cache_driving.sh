#!/usr/bin/env bash
# All collected road-driving frames, one GPU, resumable final caches.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_VISIBLE_DEVICES:?Set exactly one physical GPU, for example 0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "Only one GPU is permitted" >&2
  exit 1
fi
DGW_PYTHON="${DGW_PYTHON:-python}"
DGW_RAW="${DGW_RAW:-data/kitti_driving/raw}"
DGW_STEREO="${DGW_STEREO:-data/processed/kitti_driving_stereo_full}"
DGW_CACHE="${DGW_CACHE:-data/processed/kitti_driving_dggt_full}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONNOUSERSITE=1
"$DGW_PYTHON" -m drive_gaussian_world prepare --raw "$DGW_RAW" \
  --output "$DGW_STEREO" --all-collected --stride 1 --resume --threads 2
"$DGW_PYTHON" -m drive_gaussian_world dggt-cache --dataset "$DGW_STEREO" \
  --raw "$DGW_RAW" --output "$DGW_CACHE" --width "${DGW_WIDTH:-280}" \
  --context "${DGW_CONTEXT:-4}" --precision bf16 --device cuda:0 \
  --memory-fraction "${DGW_MEMORY_FRACTION:-0.35}" --resume
"$DGW_PYTHON" -m drive_gaussian_world validate --dataset "$DGW_CACHE"
