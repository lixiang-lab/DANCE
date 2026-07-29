# DANCE

This is the artifact for:

> **DANCE: GPU-Native Disk-Based Graph Index Construction for Billion-Scale Approximate Nearest Neighbor Search**

The manuscript is being prepared for submission to **PVLDB / VLDB 2027**. This repository contains the latest standard DANCE Vamana and GPU-FilteredVamana implementations used by the paper.

## Scope

This is intentionally a minimal repository. It contains:

- DANCE-authored CUDA and C++ source files;
- a patch for integrating DANCE into a fixed Microsoft DiskANN revision;
- a deterministic filtered-workload generator;
- build, invocation, dataset, and validation documentation.

It does not contain experimental logs, generated results, figures, indexes, datasets, competitor source code, retired implementations, or ablation controls. It also does not mirror the Microsoft DiskANN repository.

GPU-FilteredVamana uses the complete released path: per-label search, union-label search, label-aware pruning, coverage repair, and reverse-edge repair are enabled. Label balancing is disabled.

## Install into DiskANN

```bash
git clone https://github.com/microsoft/DiskANN.git
cd DiskANN
git checkout 78256bbab4685e1774e78d331e081a153be26823
git apply /path/to/DANCE/integrations/dance-diskann-78256bba.patch
```

Build the standard 1M/10M profile:

```bash
cmake -S . -B build_fast_small \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GPU_VAMANA=ON \
  -DDANCE_PROFILE=FAST_SMALL \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build_fast_small -j"$(nproc)"
```

Build GPU-FilteredVamana:

```bash
cmake -S . -B build_filtered \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_GPU_VAMANA=ON \
  -DDANCE_PROFILE=FILTERED_PAPER \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build_filtered -j"$(nproc)"
```

Use CUDA architecture 89 for RTX 4090 and 120 for RTX PRO 6000.

## Standard Vamana

```bash
build_fast_small/apps/utils/gpu_vamana_memory_index \
  --data_path /data/deep10M_base.fbin \
  --data_type float \
  --index_prefix /index/deep10m_dance \
  --R 64 --L 100 --C 80 --STEPS 64 --builder vnew2
```

The resulting graph is searched using DiskANN's ordinary `search_memory_index`. Large datasets use the patched `build_disk_index` shard pipeline and ordinary `search_disk_index`.

## GPU-FilteredVamana

```bash
build_filtered/apps/build_memory_index \
  --data_type float --dist_fn l2 \
  --data_path /data/deep1M_base.fbin \
  --index_path_prefix /index/deep1m_filtered \
  --label_file /data/deep1m_labels.txt \
  --max_degree 40 --Lbuild 100 --FilteredLbuild 100 \
  --alpha 1.2 --num_threads 16 --gpu_filtered
```

Filtered search uses DiskANN's ordinary `search_memory_index` or `search_disk_index`.

## Reproduction documentation

- [Artifact contents and entry points](ARTIFACT.md)
- [DiskANN integration and invocation](INTEGRATION.md)
- [Dataset provenance and workload generation](DATASETS.md)
- [Validated 1M/10M results and acceptance rules](REPRODUCIBILITY.md)
- [Exact baseline versions](BASELINES.md)
- [Third-party ownership and baseline repositories](THIRD_PARTY_NOTICES.md)

## Compared systems

The paper compares against or reports results from the following official repositories. Their source code is not redistributed here.

- [Microsoft DiskANN / CPU Vamana / FilteredVamana](https://github.com/microsoft/DiskANN)
- [Tagore](https://github.com/ZJU-DAILY/Tagore)
- [Jasper](https://github.com/saltsystemslab/Jasper)
- [NVIDIA cuVS](https://github.com/NVIDIA/cuvs)

Exact release tags, branches, and full commit identifiers are recorded in [BASELINES.md](BASELINES.md).

## License and ownership

DANCE-authored files are released under the MIT License in [LICENSE](LICENSE). Microsoft DiskANN and all compared systems remain the property of their respective authors and are governed by their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
