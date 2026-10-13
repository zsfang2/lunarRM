#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python train.py \
  --data-root /home/DataDisk/zsfang/lunarRM \
  --output-dir runs/unet_b16_both \
  --epochs 20 \
  --batch-size 16 \
  --base-channels 16 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 4 \
  --seed 2027
