# Reproducibility

## Validation platform

The release gate ran on one NVIDIA RTX 4090 with CUDA 13.1.115, GCC 13.3, and CMake 3.28.3. CUDA architecture 89 was used.

## Input format

Vector files use DiskANN binary layout:

```text
uint32 number_of_vectors
uint32 dimension
packed row-major vector payload
```

Filtered labels are comma-separated unsigned integer labels, one vector per line. Query filters contain one label per query.

Canonical public DEEP inputs can be obtained from the sources in [DATASETS.md](DATASETS.md). The repository does not redistribute benchmark vectors.

## Standard release gate

Profile: `FAST_SMALL`.

Parameters:

```text
R=64
L=100
C=80
STEPS=64
iterations=5
input=float32
GPU storage=FP16
distance accumulation=FP32
```

Measured results:

| Dataset | Vectors | GPU core construction | Command wall time |
|---|---:|---:|---:|
| DEEP1M | 1,000,000 | 3.144069 s | 4.46 s |
| DEEP10M | 10,000,000 | 36.990106 s | 49.43 s |

DEEP10M Recall@10:

| L | Release | Retained paper result |
|---:|---:|---:|
| 10 | 77.58 | 77.58 |
| 20 | 87.30 | 87.30 |
| 50 | 95.13 | 95.13 |
| 100 | 97.94 | 97.94 |
| 200 | 99.24 | 99.24 |
| 500 | 99.81 | 99.81 |
| 1000 | 99.94 | 99.94 |

The complete release curve is graph-equivalent to the retained best result. QPS is machine-load sensitive and is not used as a graph-correctness equality test.

## Filtered release gate

Profile: `FILTERED_PAPER`.

The fixed release path enables per-label search, union-label search, label-aware pruning, label coverage repair, and reverse-edge repair. Label balancing is zero. No environment variable or CLI option can select an ablation.

Parameters:

```text
R=40
L=100
FilteredL=100
alpha=1.2
threads=16
input=float32
GPU storage=FP16
distance accumulation=FP32
```

Measured results:

| Dataset | Vectors | Indexing time | Command wall time | Peak GPU memory |
|---|---:|---:|---:|---:|
| DEEP1M-FD12 | 1,000,000 | 8.73741 s | 9.43 s | 3.288 GiB |
| DEEP10M-FD12 | 10,000,000 | 120.912 s | 127.12 s | 20.887 GiB |

DEEP10M-FD12 Recall@10:

| L | Recall@10 |
|---:|---:|
| 10 | 79.15 |
| 20 | 88.84 |
| 40 | 94.72 |
| 80 | 97.87 |
| 160 | 99.21 |
| 320 | 99.75 |
| 650 | 99.91 |
| 1000 | 99.95 |

Both builds report zero invalid neighbors and zero self-loops.

## Acceptance rules

A release is accepted only if:

1. every required target compiles from a clean checkout;
2. standard DEEP10M Recall@10 matches the retained curve at every listed L;
3. filtered builds report zero invalid neighbors and self-loops;
4. filtered Recall@10 is no lower than the table above at the corresponding L, allowing 0.02 percentage points for platform nondeterminism;
5. no build directory, dataset, generated index, result binary, developer path, credential, or third-party implementation is tracked;
6. the public repository can be cloned without authentication.

Raw local release-gate logs are intentionally not committed because they contain machine paths. The tables above are the portable summary.
