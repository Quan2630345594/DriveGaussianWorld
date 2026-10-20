#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DGW_DGGT_COMMIT=a3276d2bbe4cbb03bcc117830b1836110a27adeb
if [[ -f third_party/dggt/dggt/models/vggt.py ]]; then
  echo 'DGGT source already exists; leave it unchanged.'
  exit 0
fi
if [[ -e third_party/dggt ]]; then
  echo 'Destination exists but is incomplete. Choose a clean checkout.' >&2
  exit 1
fi
mkdir -p third_party
git clone --no-checkout https://github.com/xiaomi-research/dggt.git third_party/dggt
git -C third_party/dggt checkout --detach "$DGW_DGGT_COMMIT"
echo 'Source prepared. Follow third_party/dggt/README.md for dependencies and pretrained weights.'
