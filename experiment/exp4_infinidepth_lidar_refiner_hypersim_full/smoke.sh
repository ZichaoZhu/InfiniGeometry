#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="${1:?usage: smoke.sh OUTPUT_DIRECTORY [DEVICE]}"
DEVICE="${2:-cuda:0}"
CONFIG="$ROOT/experiment/exp4_infinidepth_lidar_refiner_hypersim_full/config.json"
PRIMARY_ROOT="/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp4_lidar_refiner"
BACKUP_ROOT="/nas1/home/zhuzichao/projects/InfiniGeometry/experiments/exp4_lidar_refiner"

if [[ "$OUTPUT" != "$PRIMARY_ROOT/"* ]]; then
  echo "Smoke output must stay below $PRIMARY_ROOT" >&2
  exit 2
fi
BACKUP_OUTPUT="$BACKUP_ROOT/${OUTPUT#"$PRIMARY_ROOT/"}"

cd "$ROOT"
mkdir -p "$OUTPUT/logs"
python -m training.disparity_refiner.train_lidar \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  --smoke \
  --smoke-max-steps 1 2>&1 | tee -a "$OUTPUT/logs/train.log"
python -m training.disparity_refiner.train_lidar \
  --config "$CONFIG" \
  --output "$OUTPUT" \
  --device "$DEVICE" \
  --resume "$BACKUP_OUTPUT/checkpoints/step_000000001" \
  --smoke \
  --smoke-max-steps 2 2>&1 | tee -a "$OUTPUT/logs/train.log"
