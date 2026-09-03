#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_USED_MIB="${MAX_USED_MIB:-22000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-5}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_CHECKS=0

case "${MODE}" in
  smoke|formal|all) ;;
  *)
    echo "usage: $0 [smoke|formal|all]" >&2
    exit 2
    ;;
esac

while true; do
  IFS=, read -r USED_MIB UTILIZATION < <(
    nvidia-smi -i "${GPU_INDEX}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  USED_MIB="${USED_MIB//[[:space:]]/}"
  UTILIZATION="${UTILIZATION//[[:space:]]/}"
  date --iso-8601=seconds
  echo "gpu=${GPU_INDEX} used_mib=${USED_MIB} utilization=${UTILIZATION} idle_checks=${IDLE_CHECKS}"

  if (( USED_MIB <= MAX_USED_MIB && UTILIZATION <= MAX_UTILIZATION )); then
    ((IDLE_CHECKS += 1))
  else
    IDLE_CHECKS=0
  fi
  if (( IDLE_CHECKS >= REQUIRED_IDLE_CHECKS )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "${MODE}" == "all" ]]; then
  env GPU_INDEX="${GPU_INDEX}" "${SCRIPT_DIR}/run.sh" smoke
  exec env GPU_INDEX="${GPU_INDEX}" "${SCRIPT_DIR}/run.sh" formal
fi
exec env GPU_INDEX="${GPU_INDEX}" "${SCRIPT_DIR}/run.sh" "${MODE}"
