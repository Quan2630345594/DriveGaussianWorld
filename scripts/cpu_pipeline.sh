#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
DGW_PYTHON="${DGW_PYTHON:-python}"
DGW_RUN="${1:-runs/cpu_$(date +%Y%m%d_%H%M%S)}"
# Existing output directories are deliberately rejected; new run keeps all evidence.
"$DGW_PYTHON" -m drive_gaussian_world collect --root data/kitti --frames 128 --stride 4
"$DGW_PYTHON" -m drive_gaussian_world prepare --output "$DGW_RUN/stereo" --frames 128 --stride 4
"$DGW_PYTHON" -m drive_gaussian_world dggt-cache --dataset "$DGW_RUN/stereo" --output "$DGW_RUN/dggt" --width 140 --context 2
"$DGW_PYTHON" -m drive_gaussian_world audit --root data/kitti --dataset "$DGW_RUN/dggt" --output "$DGW_RUN/data_audit.json"
"$DGW_PYTHON" -m drive_gaussian_world experiments --dataset "$DGW_RUN/dggt" --stereo-dataset "$DGW_RUN/stereo" --output "$DGW_RUN/experiments"
