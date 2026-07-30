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

## Filtered workloads

The paper evaluates a 47-label geometry-aware overlapping-label procedure on
DEEP, COLOR, and SIFT, and a real-metadata workload on YFCC1M. The prepared
labels, queries, exact ground truth, and input manifests belong to the paper's
data artifact; they are inputs to DANCE and are not alternative construction
implementations. Exact filtered ground truth is computed by the upstream
DiskANN `compute_groundtruth_for_filters` utility using FP32 squared-L2
distance within each selected label posting list.

## File formats

Vector matrices contain two little-endian `uint32` values, number of rows and dimension, followed by packed row-major values. Exact ground truth contains `uint32` query count and `uint32` neighbor count, then row-major `uint32` identifiers followed by row-major `float32` distances.
