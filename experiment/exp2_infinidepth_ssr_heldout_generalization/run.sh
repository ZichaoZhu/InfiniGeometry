#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SAFE_ROOT="/mnt/data/home/zhuzichao"
PYTHON="${SAFE_ROOT}/2026_TPAMI_InfiniGeometry/envs/infinidepth_ssr/bin/python"

case "${PROJECT_ROOT}" in
  "${SAFE_ROOT}"/*) ;;
  *) echo "Refusing to run outside ${SAFE_ROOT}: ${PROJECT_ROOT}" >&2; exit 2 ;;
esac

export TMPDIR="${SAFE_ROOT}/2026_TPAMI_InfiniGeometry/tmp/infinidepth_ssr"
export HF_HOME="${SAFE_ROOT}/2026_TPAMI_InfiniGeometry/cache/infinidepth_ssr/huggingface"
export TORCH_HOME="${SAFE_ROOT}/2026_TPAMI_InfiniGeometry/cache/infinidepth_ssr/torch"
export PIP_CACHE_DIR="${SAFE_ROOT}/2026_TPAMI_InfiniGeometry/cache/infinidepth_ssr/pip"
export XDG_CACHE_HOME="${SAFE_ROOT}/2026_TPAMI_InfiniGeometry/cache/infinidepth_ssr/xdg"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${XDG_CACHE_HOME}"
test -x "${PYTHON}" || { echo "Missing experiment Python: ${PYTHON}" >&2; exit 2; }

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m InfiniDepth.ssr_generalization \
  --config "${SCRIPT_DIR}/config.json" \
  --safe-root "${SAFE_ROOT}" \
  --log "${SCRIPT_DIR}/artifacts/logs/run.log"
