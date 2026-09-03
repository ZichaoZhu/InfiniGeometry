#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
GPU_INDEX="${GPU_INDEX:-1}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${PROJECT_ROOT}/experiment/exp5_infinidepth_lidar_refiner_eth3d_generalization/config.json"
PYTHON="/mnt/data/home/zhuzichao/envs/infinidepth_disparity_ssr/bin/python"
OUTPUT_ROOT="/mnt/data/home/zhuzichao/projects/InfiniGeometry/experiments/exp5_ETH3D"
INPUT_MANIFEST="${OUTPUT_ROOT}/inputs/eth3d_input_manifest.json"

case "${MODE}" in
  preprocess)
    exec "${PYTHON}" -m training.disparity_refiner.prepare_eth3d_inputs \
      --extracted-root /mnt/data/home/zhuzichao/datasets/ETH3D/high_res_training_exp5/extracted_20260903 \
      --output "${INPUT_MANIFEST}" \
      --expected-scenes 13 \
      --expected-samples 454
    ;;
  smoke)
    OUTPUT="${OUTPUT_ROOT}/smoke/eth3d_train5_step22500"
    MIN_FREE_MIB=8000
    ARGS=(--max-samples 5 --no-local-masks)
    ;;
  masks)
    OUTPUT="${OUTPUT_ROOT}/masks/moge3_v2_sam2_1_small_eth3d_density010_v1"
    MIN_FREE_MIB=12000
    ARGS=()
    ;;
  formal)
    OUTPUT="${OUTPUT_ROOT}/runs/eth3d_highres_train_step22500"
    MIN_FREE_MIB=8000
    ARGS=()
    ;;
  *)
    echo "usage: $0 [preprocess|smoke|masks|formal]" >&2
    exit 2
    ;;
esac

FREE_MIB="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d '[:space:]')"
if (( FREE_MIB < MIN_FREE_MIB )); then
  echo "GPU ${GPU_INDEX} has only ${FREE_MIB} MiB free; need ${MIN_FREE_MIB} MiB" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"
if [[ "${MODE}" == "masks" ]]; then
  exec env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON}" -m training.disparity_refiner.prepare_eth3d_masks \
    --config "${CONFIG}" \
    --input-manifest "${INPUT_MANIFEST}" \
    --output "${OUTPUT}" \
    --device cuda:0 \
    "${ARGS[@]}"
fi
exec env CUDA_VISIBLE_DEVICES="${GPU_INDEX}" "${PYTHON}" -m training.disparity_refiner.evaluate_eth3d \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  "${ARGS[@]}"
