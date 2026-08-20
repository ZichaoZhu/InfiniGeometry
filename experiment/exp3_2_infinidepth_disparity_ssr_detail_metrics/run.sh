#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT="$ROOT/experiment/exp3_2_infinidepth_disparity_ssr_detail_metrics"
PERSONAL_ROOT="/mnt/data/home/zhuzichao"
export CONDA_PREFIX="$PERSONAL_ROOT/envs/infinidepth_disparity_ssr"
export PATH="$CONDA_PREFIX/bin:$PATH"
export TMPDIR="$PERSONAL_ROOT/tmp/infinidepth_disparity_ssr"
export XDG_CACHE_HOME="$PERSONAL_ROOT/cache/infinidepth_disparity_ssr"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export CONDA_PKGS_DIRS="$XDG_CACHE_HOME/conda_pkgs"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export npm_config_cache="$XDG_CACHE_HOME/npm"
export PLAYWRIGHT_BROWSERS_PATH="$XDG_CACHE_HOME/ms-playwright"
export INFINIDEPTH_CHECKPOINT="$ROOT/checkpoints/depth/infinidepth.ckpt"
export INFINIDEPTH_TEST_TMP_ROOT="$TMPDIR/tests"
source "$ROOT/experiment/server_environment_guard.sh"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m training.disparity_refiner.evaluate_details \
  --experiment "$EXPERIMENT" \
  --device cuda:0
