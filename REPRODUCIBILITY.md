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

The release compiles `gpu_stitched_vamana_index` under the `QUALITY` profile
used for the Figure 10 GPU-StitchedVamana results. Its fixed path performs
label-local GPU Vamana construction, global-ID remapping, edge stitching,
sidecar generation, and conditional global pruning. The label-local float
builders use FP32 vectors and FP32 distance accumulation
(`ENABLE_FLOAT16_BUILD=0`).

### DEEP1M-MultiLabel release gate

The lightweight GPU-StitchedVamana gate uses the same overlapping 47-label
DEEP1M workload and parameters as Figure 10:

```text
Rsmall=32
Lsmall=100
Rstitched=64
C=96
steps=64
alpha=1.2
```

Build the index:

```bash
/usr/bin/time -f 'wall_seconds=%e\nmax_rss_kb=%M' -o gate/build.time \
  build_quality/apps/utils/gpu_stitched_vamana_index \
  --data_type float \
  --data_path /data/deep1M_base.fbin \
  --label_file /data/deep1M_multilabel/labels.txt \
  --index_path_prefix gate/index \
  --Rsmall 32 --Lsmall 100 --Rstitched 64 \
  --C 96 --steps 64 --alpha 1.2 --num_threads 32 \
  > gate/build.log 2>&1
```

Check the graph, overlap semantics, label medoids, all search sidecars, and the
global-prune decision:

```bash
python3 tools/validate_gpu_stitched_gate.py \
  --index-prefix gate/index \
  --build-log gate/build.log \
  --expected-nodes 1000000 \
  --expected-labels 47 \
  --max-degree 64 \
  --output gate/validation.json
```

Search P50 and P1 with the ordinary DiskANN memory-search executable:

```bash
for bucket in P50 P1; do
  build_quality/apps/search_memory_index \
    --data_type float --dist_fn l2 \
    --index_path_prefix gate/index \
    --result_path gate/${bucket}_result \
    --query_file /data/deep1M_multilabel/${bucket}_queries.fbin \
    --query_filters_file /data/deep1M_multilabel/${bucket}_query_filters.txt \
    --gt_file /data/deep1M_multilabel/${bucket}_groundtruth.bin \
    --recall_at 10 --search_list 20 --num_threads 16
done
```

A clean release build on RTX 4090 produced:

| Check | Fresh release gate | Figure 10 retained result |
|---|---:|---:|
| Builder total | 12.120 s | 12.53 s |
| Full command wall time | 12.32 s | — |
| P50 Recall@10, L=20 | 76.19% | 76.19% |
| P1 Recall@10, L=20 | 73.90% | 73.90% |

The clean-checkout gate reproduces both retained Figure 10 recall values
exactly. Builder time and full command wall time are reported separately.

The gate processed all 3,811,856 memberships across 47 overlapping labels.
The final graph has 1,000,000 nodes, maximum degree 64, and zero invalid IDs,
self-loops, duplicate edges, degree overflows, or label-incompatible edges.
All vector, label, formatted-label, label-map, and label-to-medoid sidecars
exist; both label tables contain one million rows; all 47 label-map and medoid
entries agree; and every medoid carries its assigned label.

Global pruning is required for this overlapping workload. The stitched union
has maximum degree 178, so the builder executes the global label-aware prune
for 1.828 s and emits a degree-64 graph. The machine-readable retained gate is
[`validation/gpu_stitched_deep1m_gate.json`](validation/gpu_stitched_deep1m_gate.json).

## Acceptance rules

A release is accepted only if:

1. every required target compiles from a clean checkout;
2. the emitted standard and filtered indexes contain no invalid neighbors or self-loops;
3. the executable startup diagnostics report the canonical parameters listed above;
4. no build directory, dataset, generated index, result binary, developer path, credential, or third-party implementation is tracked;
5. the public repository can be cloned without authentication.

Historical measurements obtained with superseded parameter combinations are intentionally omitted. QPS is machine-load sensitive and is not used as a graph-correctness equality test.
