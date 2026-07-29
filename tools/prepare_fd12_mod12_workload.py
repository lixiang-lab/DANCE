#!/usr/bin/env python3
# Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np


LABELS = tuple(f"NHQ_A{a}_B{b}_C{c}" for a in range(3) for b in range(2) for c in range(2))
DTYPES = {"float": np.float32, "float32": np.float32, "uint8": np.uint8, "int8": np.int8}


def matrix(path: Path, dtype: np.dtype) -> tuple[int, int, np.memmap]:
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"short matrix header: {path}")
    n, dim = struct.unpack("<II", header)
    expected = 8 + n * dim * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"invalid matrix size for {path}: expected {expected}, got {path.stat().st_size}")
    return n, dim, np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, dim))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_queries(source: Path, output: Path, dtype: np.dtype, limit: int) -> tuple[int, int]:
    n, dim, values = matrix(source, dtype)
    count = min(n, limit) if limit else n
    with output.open("wb") as stream:
        stream.write(struct.pack("<II", count, dim))
        np.asarray(values[:count], dtype=dtype).tofile(stream)
    return count, dim


def write_labels(path: Path, points: int) -> list[int]:
    counts = [0] * len(LABELS)
    with path.open("w", encoding="utf-8") as stream:
        for point in range(points):
            label = point % len(LABELS)
            counts[label] += 1
            stream.write(LABELS[label] + "\n")
    return counts


def write_query_filters(path: Path, queries: int, seed: int) -> list[int]:
    repetitions = math.ceil(queries / len(LABELS))
    labels = np.asarray((list(range(len(LABELS))) * repetitions)[:queries], dtype=np.int32)
    np.random.default_rng(seed + 101).shuffle(labels)
    path.write_text("".join(f"{LABELS[int(label)]}\n" for label in labels), encoding="utf-8")
    return labels.tolist()


def exact_ground_truth(
    base_path: Path,
    query_path: Path,
    dtype: np.dtype,
    query_labels: list[int],
    output: Path,
    k: int,
    candidate_chunk: int,
    query_chunk: int,
) -> None:
    points, dim, base = matrix(base_path, dtype)
    queries, query_dim, query = matrix(query_path, dtype)
    if query_dim != dim or queries != len(query_labels):
        raise ValueError("query shape/filter count does not match the base")
    if min((points + len(LABELS) - 1 - label) // len(LABELS) for label in range(len(LABELS))) < k:
        raise ValueError("a label posting list contains fewer than k points")

    result_ids = np.full((queries, k), np.iinfo(np.uint32).max, dtype=np.uint32)
    result_distances = np.full((queries, k), np.inf, dtype=np.float32)
    grouped = {label: [] for label in range(len(LABELS))}
    for query_id, label in enumerate(query_labels):
        grouped[label].append(query_id)

    for label, query_ids in grouped.items():
        candidates = np.arange(label, points, len(LABELS), dtype=np.uint32)
        query_ids_array = np.asarray(query_ids, dtype=np.int64)
        for begin in range(0, len(query_ids), query_chunk):
            selected = query_ids_array[begin : begin + query_chunk]
            query_block = np.asarray(query[selected], dtype=np.float32)
            query_norm = np.sum(query_block * query_block, axis=1, keepdims=True)
            best_ids = np.full((len(selected), k), np.iinfo(np.uint32).max, dtype=np.uint32)
            best_distances = np.full((len(selected), k), np.inf, dtype=np.float32)
            for candidate_begin in range(0, len(candidates), candidate_chunk):
                candidate_ids = candidates[candidate_begin : candidate_begin + candidate_chunk]
                candidate_block = np.asarray(base[candidate_ids], dtype=np.float32)
                distances = (
                    query_norm
                    + np.sum(candidate_block * candidate_block, axis=1)[None, :]
                    - 2.0 * (query_block @ candidate_block.T)
                )
                np.maximum(distances, 0.0, out=distances)
                merged_distances = np.concatenate((best_distances, distances.astype(np.float32)), axis=1)
                merged_ids = np.concatenate(
                    (best_ids, np.broadcast_to(candidate_ids.astype(np.uint32), distances.shape)), axis=1
                )
                take = np.argpartition(merged_distances, k - 1, axis=1)[:, :k]
                rows = np.arange(len(selected))[:, None]
                best_distances = merged_distances[rows, take]
                best_ids = merged_ids[rows, take]
                order = np.argsort(best_distances, axis=1)
                best_distances = np.take_along_axis(best_distances, order, axis=1)
                best_ids = np.take_along_axis(best_ids, order, axis=1)
            result_ids[selected] = best_ids
            result_distances[selected] = best_distances
        print(f"[fd12] exact GT complete: {LABELS[label]} ({len(query_ids)} queries)", flush=True)

    with output.open("wb") as stream:
        stream.write(struct.pack("<II", queries, k))
        result_ids.tofile(stream)
        result_distances.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--data-type", choices=tuple(DTYPES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--max-queries", type=int, default=240)
    parser.add_argument("--gt-k", type=int, default=100)
    parser.add_argument("--compute-gt", action="store_true")
    parser.add_argument("--candidate-chunk", type=int, default=200000)
    parser.add_argument("--query-chunk", type=int, default=4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output_dir}")
    if min(args.max_queries, args.gt_k, args.candidate_chunk, args.query_chunk) <= 0:
        raise SystemExit("query, k and chunk parameters must be positive")

    dtype = DTYPES[args.data_type]
    points, dim, _ = matrix(args.base_file, dtype)
    args.output_dir.mkdir(parents=True)
    labels_path = args.output_dir / "labels.txt"
    queries_path = args.output_dir / "queries.bin"
    filters_path = args.output_dir / "query_filters.txt"
    truth_path = args.output_dir / "groundtruth.bin"
    query_count, query_dim = write_queries(args.query_file, queries_path, dtype, args.max_queries)
    if query_dim != dim:
        raise ValueError(f"query dimension {query_dim} != base dimension {dim}")
    counts = write_labels(labels_path, points)
    query_labels = write_query_filters(filters_path, query_count, args.seed)
    if args.compute_gt:
        exact_ground_truth(
            args.base_file, queries_path, dtype, query_labels, truth_path,
            args.gt_k, args.candidate_chunk, args.query_chunk,
        )

    outputs = (labels_path, queries_path, filters_path) + ((truth_path,) if args.compute_gt else ())
    manifest = {
        "schema": "dance-filtered-fd12-v1",
        "base_file": str(args.base_file.resolve()),
        "base_size": args.base_file.stat().st_size,
        "base_sha256": sha256(args.base_file),
        "query_source": str(args.query_file.resolve()),
        "query_source_size": args.query_file.stat().st_size,
        "query_source_sha256": sha256(args.query_file),
        "data_type": args.data_type,
        "points": points,
        "dimensions": dim,
        "queries": query_count,
        "seed": args.seed,
        "assignment": "point_id modulo 12",
        "query_assignment": "balanced labels shuffled by numpy default_rng(seed + 101)",
        "labels": list(LABELS),
        "label_counts": counts,
        "ground_truth": "exhaustive float32 squared L2 inside the selected label posting list"
        if args.compute_gt else "not generated",
        "ground_truth_k": args.gt_k if args.compute_gt else 0,
        "outputs": {
            path.name: {"size": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[fd12] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
