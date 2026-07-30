# Reproducibility

## Reference platform

The paper's small-scale builds use one NVIDIA RTX 4090. CUDA architecture 89 is used on RTX 4090 and architecture 120 on RTX PRO 6000.

## Input format

Vector files use DiskANN binary layout:

```text
uint32 number_of_vectors
uint32 dimension
packed row-major vector payload
```

Filtered labels are comma-separated unsigned integer labels, one vector per line. Query filters contain one label per query.

Canonical public DEEP inputs can be obtained from the sources in [DATASETS.md](DATASETS.md). The repository does not redistribute benchmark vectors.

## Standard configuration

Profile: `FAST_SMALL`.

Parameters:

```text
R=64
L=100
C=80
STEPS=64
iterations=4
input=float32
GPU storage=FP16
distance accumulation=FP32
```

Clean-checkout DEEP1M validation on RTX 4090:

| GPU core construction | Command wall time | Invalid neighbors | Self-loops |
|---:|---:|---:|---:|
| 2.768 s | 3.64 s | 0 | 0 |

| L | Recall@10 |
|---:|---:|
| 10 | 84.70 |
| 20 | 92.53 |
| 50 | 97.76 |
| 100 | 99.21 |
| 200 | 99.77 |
| 500 | 99.97 |
| 1000 | 100.00 |

## Filtered configuration

Profile: `FILTERED_PAPER`.

The fixed release path starts from a deterministic random graph with initial degree 16 and seed 42. It enables per-label search, union-label search, deterministic label-quota compaction, label-aware pruning, and exact CSR reverse-edge repair. All three refinement rounds process the complete vertex and vertex-label task sets. The active-frontier implementation remains available only through `DISKANN_GPU_FILTERED_ENABLE_ACTIVE_FRONTIER=1` and is disabled by default. Label coverage repair and the separate prune-shortlist balancing control remain disabled.

Parameters:

```text
R=64
L=100
FilteredL=100
per-label pool=32
per-label output K_F=8
union-label pool L_U=64
alpha=1.2
threads=16
input=float32
GPU storage=FP16
distance accumulation=FP32
initialization=deterministic random
initial degree=16
initialization seed=42
label refinement rounds=3
round scheduling=full task set in every round
candidate aggregation=fixed-order materialization, then label-quota compaction
label quota=max(K_F, ceil(R / number of row labels))
exact CSR reverse repair=enabled
active frontier=disabled
prune-shortlist balancing=disabled
label coverage repair=disabled
```

Clean-checkout DEEP1M-MultiLabel validation on RTX 4090 reported
`per_label_pool=32`, `per_label_keep=8`, `ordinary_pool=64`,
`active_frontier=0`, `label_quota=1`, `exact_csr_reverse=1`, three
`kind=full` rounds, and `label_coverage_repair disabled=1`.

| GPU construction | Indexing time | Command wall time | Invalid neighbors | Self-loops |
|---:|---:|---:|---:|---:|
| 12.073 s | 13.355 s | 14.14 s | 0 | 0 |

| L | P50 Recall@10 | P1 Recall@10 |
|---:|---:|---:|
| 20 | 75.85 | 69.52 |
| 40 | 86.41 | 80.96 |
| 80 | 93.50 | 88.96 |
| 160 | 97.47 | 94.78 |
| 320 | 99.14 | 97.68 |
| 650 | 99.73 | 99.28 |
| 1000 | 99.90 | 99.61 |

## GPU-StitchedVamana release target

The release includes and compiles `gpu_stitched_vamana_index` under the `FILTERED_PAPER` profile. Its fixed path performs label-local GPU Vamana construction, global-ID remapping, edge stitching, sidecar generation, and conditional global pruning. It shares the same FP16 GPU vector representation and FP32 distance accumulation as the filtered profile.

## Acceptance rules

A release is accepted only if:

1. every required target compiles from a clean checkout;
2. the emitted standard and filtered indexes contain no invalid neighbors or self-loops;
3. the executable startup diagnostics report the canonical parameters listed above;
4. no build directory, dataset, generated index, result binary, developer path, credential, or third-party implementation is tracked;
5. the public repository can be cloned without authentication.

Historical measurements obtained with superseded parameter combinations are intentionally omitted. QPS is machine-load sensitive and is not used as a graph-correctness equality test.
