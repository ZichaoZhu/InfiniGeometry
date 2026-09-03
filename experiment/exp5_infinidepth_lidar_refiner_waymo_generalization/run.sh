#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
GPU_INDEX="${GPU_INDEX:-0}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/experiment/exp5_infinidepth_lidar_refiner_waymo_generalization/config.json"
PYTHON="/mnt/data/home/zhuzichao/envs/infinidepth_disparity_ssr/bin/python"
OUTPUT_ROOT="/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp5_waymo_generalization"

case "${MODE}" in
  smoke)
    OUTPUT="${OUTPUT_ROOT}/smoke/waymo_val5_front_step22500"
    LIMIT=(--max-files 5)
    ;;
  formal)
    OUTPUT="${OUTPUT_ROOT}/runs/waymo_val202_front_step22500"
    LIMIT=()
    ;;
  *)
    echo "usage: $0 [smoke|formal]" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON}" -m training.disparity_refiner.evaluate_waymo \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  "${LIMIT[@]}"
