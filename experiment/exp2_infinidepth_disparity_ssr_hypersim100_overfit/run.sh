#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/experiment/exp2_infinidepth_disparity_ssr_hypersim100_overfit/config.json"
OUTPUT="$ROOT/experiment/exp2_infinidepth_disparity_ssr_hypersim100_overfit/runs/main"
source "$ROOT/experiment/server_environment_guard.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/monitor"
python experiment/monitor_exp2.py \
  --experiment "$(dirname "$(dirname "$OUTPUT")")" \
  --training-pid "$$" \
  --interval-seconds 300 \
  --stale-seconds 7200 \
  --daemon >>"$OUTPUT/monitor/daemon.log" 2>&1 &
ARGS=(
  --config "$CONFIG"
  --run-id main
  --output "$OUTPUT"
  --device "${INFINIDEPTH_DEVICE:-cuda:0}"
)
if [[ -f "$OUTPUT/checkpoints/last.pt" ]]; then
  ARGS+=(--resume "$OUTPUT/checkpoints/last.pt")
fi
exec python -m training.disparity_refiner.train "${ARGS[@]}"
