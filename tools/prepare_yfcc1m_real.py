#!/usr/bin/env python3
# Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.
"""Convert the official YFCC-1M single-low equality workload to DiskANN files."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.input_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    metadata_path = source / "base.1M.jsonl"
    query_filter_path = source / "query_filters.jsonl"
    source_gt_path = source / "GT_1M.bin"

    token_to_id: dict[tuple[str, str], int] = {}
    counts: Counter[int] = Counter()
    point_label_counts: list[int] = []
    label_rows: list[list[int]] = []

    with metadata_path.open(encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            row = json.loads(line)
            if row.pop("doc_id") != expected_id:
                raise ValueError(f"non-contiguous doc_id at row {expected_id}")
            labels: list[int] = []
            for field, value in sorted(row.items()):
                key = (str(field), str(value))
                label_id = token_to_id.setdefault(key, len(token_to_id))
                labels.append(label_id)
                counts[label_id] += 1
            if not labels:
                raise ValueError(f"empty metadata at row {expected_id}")
            labels.sort()
            label_rows.append(labels)
            point_label_counts.append(len(labels))

    query_label_ids: list[int | None] = []
    with query_filter_path.open(encoding="utf-8") as handle:
        for expected_id, line in enumerate(handle):
            row = json.loads(line)
            if row["query_id"] != expected_id:
                raise ValueError(f"non-contiguous query_id at row {expected_id}")
            predicate = row["filter"]
            if len(predicate) != 1:
                raise ValueError(f"query {expected_id} is not a single predicate")
            field, operation = next(iter(predicate.items()))
            if set(operation) != {"$eq"}:
                raise ValueError(f"query {expected_id} is not equality")
            key = (str(field), str(operation["$eq"]))
            query_label_ids.append(token_to_id.get(key))

    labels_out = output / "labels.txt"
    with labels_out.open("w", encoding="utf-8") as handle:
        for labels in label_rows:
            handle.write(",".join(map(str, labels)) + "\n")

    mapping_out = output / "label_mapping.json"
    mapping = [
        {"label_id": label_id, "field": field, "value": value, "count": counts[label_id]}
        for (field, value), label_id in sorted(token_to_id.items(), key=lambda item: item[1])
    ]
    with mapping_out.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    raw = source_gt_path.read_bytes()
    nq, total = struct.unpack_from("<II", raw, 0)
    if nq != len(query_label_ids):
        raise ValueError(f"GT nq={nq}, filters={len(query_label_ids)}")
    counts_offset = 8
    result_counts = struct.unpack_from(f"<{nq}I", raw, counts_offset)
    ids_offset = counts_offset + 4 * nq
    if ids_offset + 4 * total != len(raw):
        raise ValueError("unexpected range-ground-truth size")
    all_ids = struct.unpack_from(f"<{total}I", raw, ids_offset)
    k = 10
    result_offsets: list[int] = []
    cursor = 0
    for result_count in result_counts:
        result_offsets.append(cursor)
        cursor += result_count
    if cursor != total:
        raise ValueError("range-ground-truth result count mismatch")

    selected_query_ids = [
        query_id
        for query_id, (label_id, result_count) in enumerate(zip(query_label_ids, result_counts))
        if label_id is not None and result_count >= k
    ]
    fixed_ids: list[int] = []
    for query_id in selected_query_ids:
        offset = result_offsets[query_id]
        fixed_ids.extend(all_ids[offset : offset + k])

    filters_out = output / "query_filters.txt"
    with filters_out.open("w", encoding="utf-8") as handle:
        for query_id in selected_query_ids:
            handle.write(f"{query_label_ids[query_id]}\n")

    query_source = source / "query_10k.u8bin"
    with query_source.open("rb") as handle:
        query_n, query_dim = struct.unpack("<II", handle.read(8))
        query_bytes = handle.read()
    if query_n != nq or len(query_bytes) != query_n * query_dim:
        raise ValueError("unexpected uint8 query-vector size")
    query_out = output / "queries.u8bin"
    with query_out.open("wb") as handle:
        handle.write(struct.pack("<II", len(selected_query_ids), query_dim))
        for query_id in selected_query_ids:
            start = query_id * query_dim
            handle.write(query_bytes[start : start + query_dim])
    with (output / "query_original_ids.txt").open("w", encoding="utf-8") as handle:
        for query_id in selected_query_ids:
            handle.write(f"{query_id}\n")

    gt_out = output / "groundtruth_top10.bin"
    with gt_out.open("wb") as handle:
        selected_nq = len(selected_query_ids)
        handle.write(struct.pack("<II", selected_nq, k))
        handle.write(struct.pack(f"<{selected_nq * k}I", *fixed_ids))
        handle.write(struct.pack(f"<{selected_nq * k}f", *([0.0] * (selected_nq * k))))

    candidate_counts = [counts[query_label_ids[query_id]] for query_id in selected_query_ids]
    sorted_candidates = sorted(candidate_counts)
    sorted_memberships = sorted(point_label_counts)

    def percentile(values: list[int], fraction: float) -> int:
        return values[round((len(values) - 1) * fraction)]

    manifest = {
        "dataset": "official YFCC-1M single-low",
        "source": "NeurIPS BigANN YFCC supplemental filtered workload",
        "source_documentation": "https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/yfcc_filtered_dataset.md",
        "filter_semantics": "single-field equality",
        "official_match_rate_range": [0.0001, 0.001],
        "points": len(label_rows),
        "original_queries": len(query_label_ids),
        "queries": len(selected_query_ids),
        "excluded_queries_with_fewer_than_10_matches": len(query_label_ids) - len(selected_query_ids),
        "dimensions": 192,
        "data_type": "uint8",
        "unique_labels": len(token_to_id),
        "total_memberships": sum(point_label_counts),
        "labels_per_point": {
            "mean": sum(point_label_counts) / len(point_label_counts),
            "p50": percentile(sorted_memberships, 0.5),
            "p90": percentile(sorted_memberships, 0.9),
            "max": max(point_label_counts),
        },
        "candidate_count": {
            "min": min(candidate_counts),
            "mean": sum(candidate_counts) / len(candidate_counts),
            "p50": percentile(sorted_candidates, 0.5),
            "p90": percentile(sorted_candidates, 0.9),
            "max": max(candidate_counts),
        },
        "selectivity": {
            "min": min(candidate_counts) / len(label_rows),
            "mean": sum(candidate_counts) / len(candidate_counts) / len(label_rows),
            "p50": percentile(sorted_candidates, 0.5) / len(label_rows),
            "p90": percentile(sorted_candidates, 0.9) / len(label_rows),
            "max": max(candidate_counts) / len(label_rows),
        },
        "range_groundtruth": {
            "queries": nq,
            "total_results": total,
            "min_results": min(result_counts),
            "max_results": max(result_counts),
        },
        "files": {
            "base": str(source / "base.1M.u8bin"),
            "queries": str(query_out),
            "labels": str(labels_out),
            "query_filters": str(filters_out),
            "groundtruth_top10": str(gt_out),
            "label_mapping": str(mapping_out),
        },
        "source_sha256": {
            "base.1M.u8bin": sha256(source / "base.1M.u8bin"),
            "base.1M.jsonl": sha256(metadata_path),
            "query_10k.u8bin": sha256(query_source),
            "query_filters.jsonl": sha256(query_filter_path),
            "GT_1M.bin": sha256(source_gt_path),
        },
        "output_sha256": {
            "queries.u8bin": sha256(query_out),
            "labels.txt": sha256(labels_out),
            "query_filters.txt": sha256(filters_out),
            "groundtruth_top10.bin": sha256(gt_out),
            "label_mapping.json": sha256(mapping_out),
        },
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
