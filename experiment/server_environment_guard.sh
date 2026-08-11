#!/usr/bin/env bash

# This file is sourced by server-only entry points.
SERVER_ENV_EXPECTED_ROOT="/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/InfiniDepth"
SERVER_ENV_SAFE_ROOT="/mnt/data/home/zhuzichao"
SERVER_ENV_EXPECTED_ENV="$SERVER_ENV_SAFE_ROOT/envs/infinidepth_disparity_ssr"
SERVER_ENV_EXPECTED_CACHE="$SERVER_ENV_SAFE_ROOT/cache/infinidepth_disparity_ssr"
SERVER_ENV_ACTUAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_ENV_ACTIVE_PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"

if [[ "$SERVER_ENV_ACTUAL_ROOT" != "$SERVER_ENV_EXPECTED_ROOT" ]]; then
  echo "拒绝执行：仓库路径不是 $SERVER_ENV_EXPECTED_ROOT" >&2
  exit 2
fi
if [[ "$SERVER_ENV_ACTIVE_PREFIX" != "$SERVER_ENV_EXPECTED_ENV" ]]; then
  echo "拒绝执行：必须激活个人隔离环境 $SERVER_ENV_EXPECTED_ENV" >&2
  exit 2
fi
if [[ "${TMPDIR:-}" != "$SERVER_ENV_SAFE_ROOT/tmp/infinidepth_disparity_ssr" ]]; then
  echo "拒绝执行：TMPDIR 必须位于个人实验临时目录" >&2
  exit 2
fi
if [[ "${XDG_CACHE_HOME:-}" != "$SERVER_ENV_EXPECTED_CACHE" ]]; then
  echo "拒绝执行：XDG_CACHE_HOME 必须是 $SERVER_ENV_EXPECTED_CACHE" >&2
  exit 2
fi
if [[ "${PIP_CACHE_DIR:-}" != "$SERVER_ENV_EXPECTED_CACHE/pip" ]]; then
  echo "拒绝执行：PIP_CACHE_DIR 必须位于个人缓存目录" >&2
  exit 2
fi
if [[ "${CONDA_PKGS_DIRS:-}" != "$SERVER_ENV_EXPECTED_CACHE/conda_pkgs" ]]; then
  echo "拒绝执行：CONDA_PKGS_DIRS 必须位于个人缓存目录" >&2
  exit 2
fi
if [[ "${TORCH_HOME:-}" != "$SERVER_ENV_EXPECTED_CACHE/torch" ]]; then
  echo "拒绝执行：TORCH_HOME 必须位于个人缓存目录" >&2
  exit 2
fi
if [[ "${HF_HOME:-}" != "$SERVER_ENV_EXPECTED_CACHE/huggingface" ]]; then
  echo "拒绝执行：HF_HOME 必须位于个人缓存目录" >&2
  exit 2
fi
if [[ "${MPLCONFIGDIR:-}" != "$SERVER_ENV_EXPECTED_CACHE/matplotlib" ]]; then
  echo "拒绝执行：MPLCONFIGDIR 必须位于个人缓存目录" >&2
  exit 2
fi
if [[ "${npm_config_cache:-}" != "$SERVER_ENV_EXPECTED_CACHE/npm" ]]; then
  echo "拒绝执行：npm_config_cache 必须位于个人缓存目录" >&2
  exit 2
fi
if [[ "${PLAYWRIGHT_BROWSERS_PATH:-}" != "$SERVER_ENV_EXPECTED_CACHE/ms-playwright" ]]; then
  echo "拒绝执行：PLAYWRIGHT_BROWSERS_PATH 必须位于个人缓存目录" >&2
  exit 2
fi
if [[ "${INFINIDEPTH_TEST_TMP_ROOT:-}" != "$TMPDIR/tests" ]]; then
  echo "拒绝执行：INFINIDEPTH_TEST_TMP_ROOT 必须是 $TMPDIR/tests" >&2
  exit 2
fi
if [[ "${INFINIDEPTH_CHECKPOINT:-}" != "$SERVER_ENV_ACTUAL_ROOT/checkpoints/depth/infinidepth.ckpt" ]]; then
  echo "拒绝执行：INFINIDEPTH_CHECKPOINT 未指向当前个人仓库" >&2
  exit 2
fi

mkdir -p \
  "$TMPDIR" \
  "$INFINIDEPTH_TEST_TMP_ROOT" \
  "$PIP_CACHE_DIR" \
  "$CONDA_PKGS_DIRS" \
  "$TORCH_HOME" \
  "$HF_HOME" \
  "$MPLCONFIGDIR" \
  "$npm_config_cache" \
  "$PLAYWRIGHT_BROWSERS_PATH"

unset SERVER_ENV_EXPECTED_ROOT SERVER_ENV_SAFE_ROOT SERVER_ENV_EXPECTED_ENV
unset SERVER_ENV_EXPECTED_CACHE SERVER_ENV_ACTUAL_ROOT SERVER_ENV_ACTIVE_PREFIX
