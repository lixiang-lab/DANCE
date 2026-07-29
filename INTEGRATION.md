# DiskANN integration

## Supported base

The tested upstream dependency is:

```text
repository: https://github.com/microsoft/DiskANN
branch: cpp_main
commit: 78256bbab4685e1774e78d331e081a153be26823
```

DiskANN is not bundled in this repository. Clone it separately and retain its license and notices.

## Apply DANCE

```bash
git clone https://github.com/microsoft/DiskANN.git
cd DiskANN
git checkout 78256bbab4685e1774e78d331e081a153be26823
git apply /path/to/DANCE/integrations/dance-diskann-78256bba.patch
```

The patch installs the DANCE sources, CMake targets, memory builders, filtered builder, shard builder, optional GPU merge, and DiskANN-compatible serialization.

## Build profiles

```bash
cmake -S . -B build_gpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GPU_VAMANA=ON \
  -DDANCE_PROFILE=QUALITY \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build_gpu -j"$(nproc)"
```

| Profile | Purpose |
|---|---|
| `FAST_SMALL` | Fast standard 1M and 10M construction |
| `QUALITY` | FP32 standard construction |
| `PRO6000_100M` | RTX PRO 6000 100M construction |
| `FILTERED_PAPER` | Released GPU-FilteredVamana |

Use architecture 89 for RTX 4090 and 120 for RTX PRO 6000.

## Invocation

Standard in-memory construction uses `gpu_vamana_memory_index`.

Standard sharded construction uses:

```bash
export DISKANN_USE_GPU_SHARD_BUILD=1
export DISKANN_SHARD_BUILD_PARALLELISM=1
export DISKANN_SHARD_BUILD_DEVICES=0
build_gpu/apps/build_disk_index <DiskANN arguments>
```

Filtered in-memory construction uses:

```bash
build_gpu/apps/build_memory_index <DiskANN arguments> --gpu_filtered
```

Filtered sharded construction uses:

```bash
build_gpu/apps/build_disk_index <DiskANN arguments> --gpu_filtered
```

The release does not expose ablation variants. Search uses the ordinary DiskANN `search_memory_index` or `search_disk_index`.

## Storage placement

Construction and search locations are independent. The paper's SSD experiments construct and retain partition, PQ, materialized-shard, and intermediate index files on the capacity SSD. The completed searchable index and required sidecars are copied to the search SSD before invoking `search_disk_index`.

## Numeric representation

`FAST_SMALL`, `PRO6000_100M`, and `FILTERED_PAPER` convert float input to FP16 for GPU construction and accumulate distances in FP32. `QUALITY` stores float vectors in FP32 and accumulates in FP32. The uint8 builder consumes uint8 source values and uses FP32 distance accumulation. Serialized full vectors retain the DiskANN input representation.
