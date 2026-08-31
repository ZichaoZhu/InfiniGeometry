#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENVIRONMENT="/mnt/data/home/zhuzichao/envs/infinidepth_disparity_ssr"
PYTHON="$ENVIRONMENT/bin/python"
RUN_ID="${RUN_ID:-main}"
OUTPUT="/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp4_lidar_refiner/runs/${RUN_ID}"

if [[ "${START_EXP4_TRAINING:-}" != "YES" ]]; then
  echo "Set START_EXP4_TRAINING=YES only after formal training is approved." >&2
  exit 2
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "Environment Python is missing: $PYTHON" >&2
  exit 2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr/xdg"
export TORCH_HOME="/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr/torch"
export TMPDIR="/mnt/data/home/zhuzichao/tmp/infinidepth_disparity_ssr"
cd "$ROOT"
mkdir -p "$OUTPUT/logs"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME")
fi

"$PYTHON" -m training.disparity_refiner.train_lidar \
  --config "$ROOT/experiment/exp4_infinidepth_lidar_refiner_hypersim_full/config.json" \
  --run-id "$RUN_ID" \
  --output "$OUTPUT" \
  --device "${DEVICE:-cuda:0}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "$OUTPUT/logs/train.log"
