#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DGW_PYTHON="${DGW_PYTHON:-python}"
DGW_DEVICE="${DGW_DEVICE:-cuda}"
DGW_RUN="${1:-runs/drive_$(date +%Y%m%d_%H%M%S)}"
DGW_CONFIG="${DGW_CONFIG:-configs/default.json}"
DGW_WIDTH="${DGW_WIDTH:-518}"
DGW_CONTEXT="${DGW_CONTEXT:-4}"
# The same configuration controls device and precision for every model stage.
"$DGW_PYTHON" -m drive_gaussian_world collect --root data/kitti --frames 128 --stride 4
"$DGW_PYTHON" -m drive_gaussian_world prepare --output "$DGW_RUN/stereo" --frames 128 --stride 4
"$DGW_PYTHON" -m drive_gaussian_world dggt-cache --dataset "$DGW_RUN/stereo" --output "$DGW_RUN/dggt" --device "$DGW_DEVICE" --width "$DGW_WIDTH" --context "$DGW_CONTEXT"
"$DGW_PYTHON" -m drive_gaussian_world audit --dataset "$DGW_RUN/dggt" --output "$DGW_RUN/data_audit.json"
"$DGW_PYTHON" -m drive_gaussian_world experiments --dataset "$DGW_RUN/dggt" --stereo-dataset "$DGW_RUN/stereo" --config "$DGW_CONFIG" --device "$DGW_DEVICE" --output "$DGW_RUN/experiments"
