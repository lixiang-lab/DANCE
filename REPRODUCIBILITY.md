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

The fixed release path starts from a deterministic random graph with initial degree 16 and seed 42. It enables per-label search, union-label search, label-aware pruning, and reverse-edge repair. The label coverage repair implementation remains in the source but is disabled, and label balancing is zero. No environment variable or CLI option selects an ablation.

Parameters:

```text
R=64
L=100
FilteredL=100
per-label pool=100
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
candidate aggregation=fixed-order first-come
quota balancing=disabled
label coverage repair=disabled
```

Clean-checkout DEEP1M-MultiLabel validation on RTX 4090 used 1,000,000
vectors and 3,811,856 vertex-label tasks in every round:

| GPU construction | Indexing time | Command wall time | Invalid neighbors | Self-loops |
|---:|---:|---:|---:|---:|
| 23.823 s | 25.196 s | 25.99 s | 0 | 0 |

| L | P50 Recall@10 | P1 Recall@10 |
|---:|---:|---:|
| 20 | 76.71 | 75.47 |
| 40 | 87.12 | 86.58 |
| 80 | 93.63 | 93.08 |
| 160 | 97.49 | 96.95 |
| 320 | 99.02 | 98.84 |
| 650 | 99.71 | 99.60 |
| 1000 | 99.80 | 99.78 |

The build log reports `per_label_pool=100`, `per_label_keep=8`,
`ordinary_pool=64`, three `kind=full` rounds, and
`label_coverage_repair disabled=1`.

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
