#!/usr/bin/env python3
# Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.
"""Build one reusable geometry-correlated DEEP-MultiLabel workload.

The shared artifact contains centroids and a centroid-to-label mapping.  A
scale-specific materialization only assigns every vector to its nearest shared
centroid; it never retrains or changes the mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np


PERCENTILES = (("P100", 100.0), ("P75", 75.0), ("P50", 50.0),
               ("P25", 25.0), ("P1", 1.0))


DTYPES = {"float": np.float32, "uint8": np.uint8, "int8": np.int8}


def vector_bin(path: Path, data_type: str = "float") -> np.memmap:
    dtype = np.dtype(DTYPES[data_type])
    with path.open("rb") as source:
        n, dim = struct.unpack("<II", source.read(8))
    if path.stat().st_size != 8 + n * dim * dtype.itemsize:
        raise ValueError(f"invalid {data_type} vector bin: {path}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, dim))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_vector_bin(path: Path, values: np.ndarray, data_type: str = "float") -> None:
    values = np.asarray(values, dtype=DTYPES[data_type])
    with path.open("wb") as output:
        output.write(struct.pack("<II", *values.shape))
        values.tofile(output)


def read_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as source:
        n, k = struct.unpack("<II", source.read(8))
        ids = np.fromfile(source, dtype=np.uint32, count=n * k).reshape(n, k)
        distances = np.fromfile(source, dtype=np.float32, count=n * k).reshape(n, k)
    return ids, distances


def write_truth(path: Path, ids: np.ndarray, distances: np.ndarray) -> None:
    with path.open("wb") as output:
        output.write(struct.pack("<II", *ids.shape))
        np.asarray(ids, dtype=np.uint32).tofile(output)
        np.asarray(distances, dtype=np.float32).tofile(output)


def build_ground_truths(base: Path, output_dir: Path, selected: list[dict], queries: np.ndarray,
                        queries_per_label: int, executable: Path, data_type: str = "float",
                        query_batch: int = 200) -> None:
    """Run DiskANN's independent filtered exhaustive ground truth."""
    for bucket, _ in PERCENTILES:
        rows = [row for row in selected if row["bucket"] == bucket]
        bucket_queries = np.concatenate([queries] * len(rows))
        query_path = output_dir / f"{bucket}_queries.fbin"
        filter_path = output_dir / f"{bucket}_query_filters.txt"
        truth_path = output_dir / f"{bucket}_groundtruth.bin"
        write_vector_bin(query_path, bucket_queries, data_type)
        filter_path.write_text("\n".join(row["label"] for row in rows
                                          for _ in range(queries_per_label)) + "\n")
        subprocess.run([
            str(executable), "--data_type", data_type, "--dist_fn", "l2",
            "--base_file", str(base.resolve()), "--query_file", str(query_path.resolve()),
            "--label_file", str((output_dir / "labels.txt").resolve()),
            "--filter_label_file", str(filter_path.resolve()),
            "--gt_file", str(truth_path.resolve()), "--K", "10",
        ], check=True)
        ids, distances = read_truth(truth_path)
        expected = queries_per_label * len(rows)
        if ids.shape != (expected, 10) or not np.isfinite(distances).all():
            raise RuntimeError(f"invalid exact ground truth produced for {bucket}: {ids.shape}")


def nearest_centroids(values: np.ndarray, centers: np.ndarray, batch: int):
    center_norm = np.einsum("ij,ij->i", centers, centers)
    for start in range(0, len(values), batch):
        block = np.asarray(values[start:start + batch], dtype=np.float32)
        dist = (np.einsum("ij,ij->i", block, block)[:, None]
                + center_norm[None, :] - 2.0 * block @ centers.T)
        yield start, np.argmin(dist, axis=1).astype(np.uint16)


def integer_zipf_targets(total: int, count: int, alpha: float, maximum: int) -> np.ndarray:
    weights = np.arange(1, count + 1, dtype=np.float64) ** (-alpha)
    raw = weights / weights.sum() * total
    target = np.clip(np.floor(raw).astype(np.int64), 1, maximum)
    while target.sum() < total:
        eligible = np.where(target < maximum)[0]
        score = raw[eligible] - target[eligible]
        target[eligible[np.argmax(score)]] += 1
    while target.sum() > total:
        eligible = np.where(target > 1)[0]
        score = target[eligible] - raw[eligible]
        target[eligible[np.argmax(score)]] -= 1
    return target


def farthest_seeds(centers: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(centers)))]
    best = np.full(len(centers), np.inf, dtype=np.float64)
    for _ in range(1, count):
        delta = centers - centers[chosen[-1]]
        best = np.minimum(best, np.einsum("ij,ij->i", delta, delta))
        best[chosen] = -1
        chosen.append(int(np.argmax(best)))
    return np.asarray(chosen, dtype=np.int32)


def build_mapping(centers: np.ndarray, labels: int, mean: int, cap: int,
                  max_region_centroids: int, alpha: float,
                  seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = len(centers) * mean
    targets = integer_zipf_targets(total, labels, alpha, max_region_centroids)
    seeds = farthest_seeds(centers, labels, seed)
    orders = []
    for center_id in seeds:
        delta = centers - centers[center_id]
        orders.append(np.argsort(np.einsum("ij,ij->i", delta, delta), kind="stable"))
    mapping = np.zeros((len(centers), labels), dtype=bool)
    degree = np.zeros(len(centers), dtype=np.int16)
    # Place broad regions first.  They spread capacity over most centroids and
    # leave the smaller, local regions enough nearby choices under the cap.
    for label in np.argsort(-targets, kind="stable"):
        available = [int(c) for c in orders[int(label)] if degree[c] < cap]
        need = int(targets[label])
        if len(available) < need:
            raise RuntimeError("cannot satisfy label targets under max-label cap")
        chosen = np.asarray(available[:need], dtype=np.int32)
        mapping[chosen, label] = True
        degree[chosen] += 1
    # Repair uncovered centroids without changing any label cardinality or total memberships.
    for centroid in np.where(degree == 0)[0]:
        label_order = np.argsort([
            np.dot(centers[centroid] - centers[s], centers[centroid] - centers[s])
            for s in seeds])
        repaired = False
        for label in label_order:
            donors = [int(c) for c in orders[int(label)][::-1]
                      if mapping[c, label] and degree[c] > 1]
            if donors:
                donor = donors[0]
                mapping[donor, label] = False
                degree[donor] -= 1
                mapping[centroid, label] = True
                degree[centroid] += 1
                repaired = True
                break
        if not repaired:
            raise RuntimeError("unable to repair uncovered centroid")
    if mapping.sum() != total or degree.min() < 1 or degree.max() > cap:
        raise AssertionError("invalid centroid mapping")
    return mapping, targets, seeds


def train(args) -> None:
    from sklearn.cluster import MiniBatchKMeans
    base = vector_bin(args.training_base, args.data_type)
    rng = np.random.default_rng(args.seed)
    sample_ids = np.sort(rng.choice(len(base), args.training_sample_size, replace=False))
    training = np.asarray(base[sample_ids], dtype=np.float32)
    model = MiniBatchKMeans(n_clusters=args.centroids, random_state=args.seed,
                            batch_size=args.kmeans_batch_size, max_iter=args.kmeans_iterations,
                            n_init=3, reassignment_ratio=0.01, verbose=0)
    model.fit(training)
    centers = model.cluster_centers_.astype(np.float32)
    mapping, targets, seeds = build_mapping(
        centers, args.labels, args.target_mean_labels, args.max_labels,
        args.max_region_centroids,
        args.zipf_alpha, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "centroids.npy", centers)
    np.save(args.output_dir / "centroid_to_label.npy", mapping)
    np.save(args.output_dir / "training_sample_ids.npy", sample_ids)
    with (args.output_dir / "centroid_to_label.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["centroid_id", "labels"])
        for centroid, row in enumerate(mapping):
            writer.writerow([centroid, ",".join(f"L{x:02d}" for x in np.where(row)[0])])
    metadata = {
        "definition": "nearest shared coarse centroid inherits its overlapping geometry-local labels",
        "seed": args.seed, "dimension": int(base.shape[1]),
        "centroids": args.centroids, "labels": args.labels,
        "target_centroid_memberships_per_centroid": args.target_mean_labels,
        "max_labels_per_centroid": args.max_labels, "zipf_alpha": args.zipf_alpha,
        "max_centroids_per_label": args.max_region_centroids,
        "training_base": str(args.training_base.resolve()),
        "training_base_sha256": sha256(args.training_base),
        "training_sample_size": args.training_sample_size,
        "kmeans_inertia": float(model.inertia_),
        "target_centroid_counts": targets.tolist(), "label_seed_centroids": seeds.tolist(),
    }
    (args.output_dir / "mapping_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def materialize(args) -> None:
    base = vector_bin(args.base, args.data_type)
    centers = np.load(args.mapping_dir / "centroids.npy")
    mapping = np.load(args.mapping_dir / "centroid_to_label.npy")
    meta = json.loads((args.mapping_dir / "mapping_manifest.json").read_text())
    if base.shape[1] != centers.shape[1]:
        raise ValueError("base and shared centroid dimensions differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.mapping_dir / "centroid_to_label.csv", args.output_dir / "centroid_to_label.csv")
    assignment_path = args.output_dir / "centroid_assignment.u16"
    assignment = np.memmap(assignment_path, dtype=np.uint16, mode="w+", shape=(len(base),))
    counts = np.zeros(mapping.shape[1], dtype=np.int64)
    memberships = 0
    maximum = 0
    with (args.output_dir / "labels.txt").open("w", buffering=8 << 20) as labels_out:
        for start, nearest in nearest_centroids(base, centers, args.assignment_batch_size):
            assignment[start:start + len(nearest)] = nearest
            for centroid in nearest:
                label_ids = np.flatnonzero(mapping[int(centroid)])
                labels_out.write(",".join(f"L{x:02d}" for x in label_ids) + "\n")
                counts[label_ids] += 1
                memberships += len(label_ids)
                maximum = max(maximum, len(label_ids))
    assignment.flush()
    cardinalities = sorted(counts.tolist())
    percentile_values = {name: float(np.percentile(counts, percentile))
                         for name, percentile in PERCENTILES}
    selected = []
    for name, percentile in PERCENTILES:
        target = percentile_values[name]
        ranking = sorted(range(len(counts)), key=lambda label: (abs(counts[label] - target), label))[:5]
        for rank, label in enumerate(ranking, 1):
            selected.append({"bucket": name, "percentile": percentile, "rank": rank,
                             "label": f"L{label:02d}", "cardinality": int(counts[label]),
                             "selectivity": float(counts[label] / len(base)),
                             "target_cardinality": target, "queries": args.queries_per_label})
    with (args.output_dir / "selected_labels.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=selected[0].keys())
        writer.writeheader(); writer.writerows(selected)
    with (args.output_dir / "dataset_stats.csv").open("w", newline="") as output:
        fields = ["N", "labels", "memberships", "M_over_N", "mean_labels_per_vector",
                  "max_labels_per_vector"] + [f"{name}_cardinality" for name, _ in PERCENTILES] \
                 + [f"{name}_selectivity" for name, _ in PERCENTILES]
        row = {"N": len(base), "labels": len(counts), "memberships": memberships,
               "M_over_N": memberships / len(base), "mean_labels_per_vector": memberships / len(base),
               "max_labels_per_vector": maximum}
        for name, _ in PERCENTILES:
            row[f"{name}_cardinality"] = percentile_values[name]
            row[f"{name}_selectivity"] = percentile_values[name] / len(base)
        writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader(); writer.writerow(row)
    with (args.output_dir / "label_cardinality.csv").open("w", newline="") as output:
        writer = csv.writer(output); writer.writerow(["label", "cardinality", "selectivity"])
        for label, count in enumerate(counts):
            writer.writerow([f"L{label:02d}", int(count), count / len(base)])
    query_source = vector_bin(args.queries, args.data_type)
    queries = np.asarray(query_source[:args.queries_per_label], dtype=DTYPES[args.data_type])
    build_ground_truths(args.base, args.output_dir, selected, queries, args.queries_per_label,
                        args.ground_truth_executable.resolve(), args.data_type, args.gt_query_batch)
    artifacts = ["dataset_stats.csv", "label_cardinality.csv", "selected_labels.csv", "labels.txt"]
    for bucket, _ in PERCENTILES:
        artifacts += [f"{bucket}_queries.fbin", f"{bucket}_query_filters.txt", f"{bucket}_groundtruth.bin"]
    manifest = {
        "dataset": args.dataset, "N": len(base), "dimension": int(base.shape[1]),
        "data_type": args.data_type,
        "base": str(args.base.resolve()), "base_sha256": sha256(args.base),
        "shared_mapping_dir": str(args.mapping_dir.resolve()),
        "shared_mapping_manifest_sha256": sha256(args.mapping_dir / "mapping_manifest.json"),
        "seed": meta["seed"], "labels": len(counts), "memberships": memberships,
        "M_over_N": memberships / len(base), "mean_labels_per_vector": memberships / len(base),
        "max_labels_per_vector": maximum, "queries_per_label": args.queries_per_label,
        "labels_per_percentile": 5, "equal_weighted_labels": True,
        "ground_truth": f"independent exhaustive {args.data_type} L2 top-10 within each selected label posting list",
        "files": {name: sha256(args.output_dir / name) for name in artifacts},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def ground_truth_only(args) -> None:
    with (args.output_dir / "selected_labels.csv").open(newline="") as source:
        selected = list(csv.DictReader(source))
    queries = np.asarray(vector_bin(args.queries, args.data_type)[:args.queries_per_label],
                         dtype=DTYPES[args.data_type])
    build_ground_truths(args.base, args.output_dir, selected, queries, args.queries_per_label,
                        args.ground_truth_executable.resolve(), args.data_type, args.gt_query_batch)
    with (args.output_dir / "dataset_stats.csv").open(newline="") as source:
        stats = next(csv.DictReader(source))
    artifacts = ["dataset_stats.csv", "label_cardinality.csv", "selected_labels.csv", "labels.txt"]
    for bucket, _ in PERCENTILES:
        artifacts += [f"{bucket}_queries.fbin", f"{bucket}_query_filters.txt", f"{bucket}_groundtruth.bin"]
    mapping_manifest = json.loads((args.mapping_dir / "mapping_manifest.json").read_text())
    manifest = {
        "dataset": args.dataset, "N": int(stats["N"]),
        "dimension": int(vector_bin(args.base, args.data_type).shape[1]),
        "data_type": args.data_type,
        "base": str(args.base.resolve()), "base_sha256": sha256(args.base),
        "shared_mapping_dir": str(args.mapping_dir.resolve()),
        "shared_mapping_manifest_sha256": sha256(args.mapping_dir / "mapping_manifest.json"),
        "seed": mapping_manifest["seed"], "labels": int(stats["labels"]),
        "memberships": int(stats["memberships"]), "M_over_N": float(stats["M_over_N"]),
        "mean_labels_per_vector": float(stats["mean_labels_per_vector"]),
        "max_labels_per_vector": int(stats["max_labels_per_vector"]),
        "queries_per_label": args.queries_per_label, "labels_per_percentile": 5,
        "equal_weighted_labels": True,
        "ground_truth": f"independent exhaustive {args.data_type} L2 top-10 within each selected label posting list",
        "files": {name: sha256(args.output_dir / name) for name in artifacts},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train-mapping")
    train_parser.add_argument("--training-base", type=Path, required=True)
    train_parser.add_argument("--data-type", choices=tuple(DTYPES), default="float")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=20260719)
    train_parser.add_argument("--training-sample-size", type=int, default=200000)
    train_parser.add_argument("--centroids", type=int, default=256)
    train_parser.add_argument("--labels", type=int, default=47)
    train_parser.add_argument("--target-mean-labels", type=int, default=4)
    train_parser.add_argument("--max-labels", type=int, default=8)
    train_parser.add_argument("--max-region-centroids", type=int, default=48)
    train_parser.add_argument("--zipf-alpha", type=float, default=1.15)
    train_parser.add_argument("--kmeans-batch-size", type=int, default=8192)
    train_parser.add_argument("--kmeans-iterations", type=int, default=100)
    train_parser.set_defaults(run=train)
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("--dataset", required=True)
    materialize_parser.add_argument("--data-type", choices=tuple(DTYPES), default="float")
    materialize_parser.add_argument("--base", type=Path, required=True)
    materialize_parser.add_argument("--queries", type=Path, required=True)
    materialize_parser.add_argument("--mapping-dir", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument("--ground-truth-executable", type=Path, required=True)
    materialize_parser.add_argument("--queries-per-label", type=int, default=200)
    materialize_parser.add_argument("--assignment-batch-size", type=int, default=32768)
    materialize_parser.add_argument("--gt-query-batch", type=int, default=200)
    materialize_parser.set_defaults(run=materialize)
    gt_parser = sub.add_parser("ground-truth-only")
    gt_parser.add_argument("--dataset", required=True)
    gt_parser.add_argument("--data-type", choices=tuple(DTYPES), default="float")
    gt_parser.add_argument("--base", type=Path, required=True)
    gt_parser.add_argument("--queries", type=Path, required=True)
    gt_parser.add_argument("--mapping-dir", type=Path, required=True)
    gt_parser.add_argument("--output-dir", type=Path, required=True)
    gt_parser.add_argument("--ground-truth-executable", type=Path, required=True)
    gt_parser.add_argument("--queries-per-label", type=int, default=200)
    gt_parser.add_argument("--gt-query-batch", type=int, default=25)
    gt_parser.set_defaults(run=ground_truth_only)
    args = parser.parse_args(); args.run(args)


if __name__ == "__main__":
    main()
