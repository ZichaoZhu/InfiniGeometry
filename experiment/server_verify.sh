#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/experiment/server_environment_guard.sh"
cd "$ROOT"
python -m pytest tests/disparity_refiner
python experiment/validate_experiment.py --check-git

cd "$ROOT/experiment/viewer"
npm ci
npm run typecheck
npm test
npm run build

if [[ -f "$ROOT/experiment/viewer/public/data/experiments.json" ]]; then
  npm run test:e2e
else
  echo "尚未导出查看器数据，跳过 Playwright；训练资产导出后必须重新执行。"
fi
