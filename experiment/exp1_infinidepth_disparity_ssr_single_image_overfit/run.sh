#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/experiment/exp1_infinidepth_disparity_ssr_single_image_overfit/config.json"
RUN_ID="${1:?用法: run.sh <01|02|03|04|05>}"
source "$ROOT/experiment/server_environment_guard.sh"

RUN_DIRECTORY="$(python - "$CONFIG" "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
matches = [run for run in config["runs"] if run["id"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(f"未知运行 ID: {sys.argv[2]}")
print(matches[0]["directory"])
PY
)"

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m training.disparity_refiner.train \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --output "$ROOT/experiment/exp1_infinidepth_disparity_ssr_single_image_overfit/$RUN_DIRECTORY"
