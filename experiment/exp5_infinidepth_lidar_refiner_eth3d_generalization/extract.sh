#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCHIVE_ROOT="${ETH3D_ARCHIVE_ROOT:-/mnt/data/home/zhuzichao/datasets/ETH3D/high_res_training_exp5/downloads}"
OUTPUT="${ETH3D_EXTRACTED_ROOT:-/mnt/data/home/zhuzichao/datasets/ETH3D/high_res_training_exp5/extracted_20260903}"
SEVEN_ZIP="${SEVEN_ZIP:-${PROJECT_ROOT}/third_party/7zip-25.01-linux-x64/7zz}"

if [[ ! -x "${SEVEN_ZIP}" ]]; then
  echo "7-Zip executable is unavailable: ${SEVEN_ZIP}" >&2
  exit 1
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to reuse or overwrite extraction output: ${OUTPUT}" >&2
  exit 1
fi
if [[ ! -f "${ARCHIVE_ROOT}/SHA256SUMS" ]]; then
  echo "Missing downloaded ETH3D SHA256SUMS: ${ARCHIVE_ROOT}" >&2
  exit 1
fi

mapfile -t ARCHIVES < <(awk '{print $2}' "${ARCHIVE_ROOT}/SHA256SUMS")
if [[ "${#ARCHIVES[@]}" -ne 14 ]]; then
  echo "Expected 14 ETH3D archives, found ${#ARCHIVES[@]}" >&2
  exit 1
fi
(
  cd "${ARCHIVE_ROOT}"
  sha256sum --check --status SHA256SUMS
)

TEMPORARY="${OUTPUT}.extracting.$$"
mkdir -p "${TEMPORARY}"
for archive in "${ARCHIVES[@]}"; do
  "${SEVEN_ZIP}" x -y "${ARCHIVE_ROOT}/${archive}" "-o${TEMPORARY}" >/dev/null
done

SCENE_COUNT="$(find "${TEMPORARY}" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')"
RAW_COUNT="$(find "${TEMPORARY}" -path '*/ground_truth_depth/dslr_images/*' -type f | wc -l | tr -d ' ')"
JPG_COUNT="$(find "${TEMPORARY}" -path '*/images/dslr_images/*' -type f -iname '*.jpg' | wc -l | tr -d ' ')"
CAMERA_COUNT="$(find "${TEMPORARY}" -path '*/dslr_calibration_jpg/cameras.txt' -type f | wc -l | tr -d ' ')"
if [[ "${SCENE_COUNT}" -ne 13 || "${RAW_COUNT}" -ne 454 || "${JPG_COUNT}" -ne 454 || "${CAMERA_COUNT}" -ne 13 ]]; then
  echo "Unexpected ETH3D extraction layout: scenes=${SCENE_COUNT}, raw=${RAW_COUNT}, jpg=${JPG_COUNT}, cameras=${CAMERA_COUNT}" >&2
  exit 1
fi
mv "${TEMPORARY}" "${OUTPUT}"
printf 'scenes=%s raw_depths=%s jpgs=%s cameras=%s\n' "${SCENE_COUNT}" "${RAW_COUNT}" "${JPG_COUNT}" "${CAMERA_COUNT}"
