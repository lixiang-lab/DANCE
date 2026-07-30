# Datasets

## Public vectors

DEEP1B is distributed by Yandex Research. The release gate uses prefixes of the 96-dimensional float32 base file together with the public 10,000-query file:

- https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search
- https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/

SIFT1B/BIGANN is the 128-dimensional uint8 benchmark. Its benchmark description and public files are available from:

- https://big-ann-benchmarks.com/neurips21.html
- https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/

The repository does not redistribute these vectors. Record the source URL, byte length, and SHA-256 digest of every downloaded input in each reproduction manifest.

## Standard subsets

DEEP1M and DEEP10M are the first 1,000,000 and 10,000,000 rows of the DEEP1B base file. Preserve the DiskANN binary header and replace its vector count with the selected prefix size. Queries are not truncated.

## Geometry-aware 47-label workloads

The DEEP, COLOR, and SIFT filtered workloads are generated with the same
deterministic geometry-aware procedure, not the same trained mapping. For each
vector dataset, train a 256-centroid mapping with 47 overlapping labels and
then materialize the selected dataset:

```bash
python3 tools/prepare_multilabel_workload.py train-mapping \
  --training-base /data/deep1M_base.fbin \
  --data-type float \
  --output-dir /data/deep_mapping \
  --seed 20260719 \
  --training-sample-size 200000 \
  --centroids 256 \
  --labels 47 \
  --target-mean-labels 4 \
  --max-labels 8 \
  --max-region-centroids 48 \
  --zipf-alpha 1.15

python3 tools/prepare_multilabel_workload.py materialize \
  --dataset DEEP1M-MultiLabel \
  --data-type float \
  --base /data/deep1M_base.fbin \
  --queries /data/deep_query.fbin \
  --mapping-dir /data/deep_mapping \
  --output-dir /data/deep1M_multilabel \
  --ground-truth-executable /path/to/DiskANN/build/apps/utils/compute_groundtruth_for_filters \
  --queries-per-label 200
```

Run `train-mapping` separately for COLOR and SIFT using their own vectors;
for SIFT pass `--data-type uint8`. The script fixes the seed and all mapping
parameters, writes `labels.txt`, P100/P75/P50/P25/P1 query/filter/ground-truth
files, dataset statistics, and a manifest containing SHA-256 checksums.
Ground truth is exhaustive top-10 squared-L2 search within each selected label
posting list.

Dependencies are Python 3, NumPy, and scikit-learn. The mapping is
deterministic for the recorded package versions and input-file checksum; keep
the generated `mapping_manifest.json` with every result.

## YFCC1M-Real

The real workload is the official NeurIPS BigANN YFCC supplemental
`single-low` workload: 192-dimensional uint8 CLIP descriptors, real
year/month/camera/country metadata, single-field equality predicates, and an
official match-rate range of 0.01%--0.1%. The authoritative description is:

- https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/yfcc_filtered_dataset.md

Download the exact five inputs used by the paper:

```bash
mkdir -p /data/yfcc1m_single_low
cd /data/yfcc1m_single_low
wget https://comp21storage.z5.web.core.windows.net/yfcc/base.1M.u8bin
wget https://comp21storage.z5.web.core.windows.net/yfcc/base.1M.jsonl
wget https://comp21storage.z5.web.core.windows.net/yfcc/single-low/query_10k.u8bin
wget https://comp21storage.z5.web.core.windows.net/yfcc/single-low/query_filters.jsonl
wget https://comp21storage.z5.web.core.windows.net/yfcc/single-low/GT_1M.bin
sha256sum -c /path/to/DANCE/datasets/yfcc1m_real.sha256
```

The checked-in checksum file contains:

```text
3ea51347cb66416abf75c670b69997e41c01713e10fc54fabba36795c746f90f  base.1M.u8bin
9a7444346ea5572245edc1fe96c9fbf7a64d0882fc251dc4f7321c3fe1151ab0  base.1M.jsonl
97d66d0e5ae9d7d9e7841991ed9c71af86a1644ff7c61f50fc9ca25029d682ed  query_10k.u8bin
4d1cb64d3b48e7971c0920cbca69d866c695c8a2ae86210e6bdf2cceaa12b3a5  query_filters.jsonl
793177d0ac3d9cb908c44547ee5ac786e4daed9bb42400571252d487f4782a3b  GT_1M.bin
```

Convert the official metadata and range ground truth to the exact DiskANN
inputs used by Figure 10:

```bash
python3 tools/prepare_yfcc1m_real.py \
  --input-dir /data/yfcc1m_single_low \
  --output-dir /data/yfcc1m_real_diskann
```

The converter retains the 9,898 queries having at least ten matches in the 1M
slice, takes the official range ground truth's first ten identifiers, and
writes labels, query filters, query vectors, top-10 ground truth, label
mapping, and a manifest with source and output SHA-256 checksums.

## File formats

Vector matrices contain two little-endian `uint32` values, number of rows and dimension, followed by packed row-major values. Exact ground truth contains `uint32` query count and `uint32` neighbor count, then row-major `uint32` identifiers followed by row-major `float32` distances.
