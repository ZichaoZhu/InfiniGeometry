#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiment/exp3_infinidepth_disparity_ssr_hypersim_full"
source "$ROOT/experiment/server_environment_guard.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python experiment/prepare_exp3_cache.py --experiment "$EXPERIMENT"
mkdir -p "$EXPERIMENT/runs/main"
echo "$$" > "$EXPERIMENT/runs/main/training.pid"
exec python -m training.disparity_refiner.train \
  --config "$EXPERIMENT/config.json" \
  --run-id main \
  --output "$EXPERIMENT/runs/main" \
  --device cuda:0
