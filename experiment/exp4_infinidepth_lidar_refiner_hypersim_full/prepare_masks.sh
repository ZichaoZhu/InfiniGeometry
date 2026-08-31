#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENVIRONMENT="/mnt/data/home/zhuzichao/envs/infinidepth_disparity_ssr"
CONFIG="$ROOT/experiment/exp4_infinidepth_lidar_refiner_hypersim_full/config.json"

source "$ENVIRONMENT/bin/activate"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr/xdg"
export TORCH_HOME="/mnt/data/home/zhuzichao/cache/infinidepth_disparity_ssr/torch"
export TMPDIR="/mnt/data/home/zhuzichao/tmp/infinidepth_disparity_ssr"
cd "$ROOT"

python -m training.disparity_refiner.prepare_local_masks \
  --config "$CONFIG" \
  --device "${DEVICE:-cuda:0}" \
  "$@"
