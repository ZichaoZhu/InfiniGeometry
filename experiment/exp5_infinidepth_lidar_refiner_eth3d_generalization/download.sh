#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${ETH3D_DOWNLOAD_ROOT:-/mnt/data/home/zhuzichao/datasets/ETH3D/high_res_training_exp5/downloads}"
BASE_URL="https://www.eth3d.net/data"
ARCHIVES=(
  pipes_dslr_depth.7z
  multi_view_training_dslr_jpg.7z
  courtyard_dslr_depth.7z
  delivery_area_dslr_depth.7z
  electro_dslr_depth.7z
  facade_dslr_depth.7z
  kicker_dslr_depth.7z
  meadow_dslr_depth.7z
  office_dslr_depth.7z
  playground_dslr_depth.7z
  relief_dslr_depth.7z
  relief_2_dslr_depth.7z
  terrace_dslr_depth.7z
  terrains_dslr_depth.7z
)

mkdir -p "${OUTPUT}"
for archive in "${ARCHIVES[@]}"; do
  wget --continue --https-only --timeout=30 --tries=0 \
    --directory-prefix="${OUTPUT}" "${BASE_URL}/${archive}"
done

(
  cd "${OUTPUT}"
  sha256sum "${ARCHIVES[@]}" > SHA256SUMS
  date --iso-8601=seconds > DOWNLOAD_COMPLETE
)
