#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    printf 'usage: %s /path/to/patched/DiskANN\n' "$0" >&2
    exit 2
fi

artifact_root=$(cd "$(dirname "$0")/.." && pwd)
patched_root=$(cd "$1" && pwd)

files=(
    apps/utils/gpu_vamana_memory_index.cpp
    apps/utils/gpu_stitched_vamana_index.cpp
    include/filtered_label_coverage.cuh
    include/gpu_merge.h
    include/gpu_vamana_builder.h
    include/gpu_vamana_config.h
    include/gpu_vamana_filtered.h
    src/gpu_merge.cu
    src/gpu_vamana_filtered.cu
    src/gpu_vamana_uint8.cu
    src/gpu_vamana_vnew2.cu
    src/gpu_vamana_vnew2_handoff.cu
)

for file in "${files[@]}"; do
    cmp "$artifact_root/$file" "$patched_root/$file"
done

printf 'standalone sources match the applied integration patch\n'
