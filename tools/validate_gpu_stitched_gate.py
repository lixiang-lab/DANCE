#!/usr/bin/env python3
# Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.

import argparse
import json
import re
import struct
from pathlib import Path


def parse_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield tuple(sorted(int(value) for value in line.strip().split(",") if value))


def parse_map(path: Path):
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            source, numeric = line.rstrip().split("\t")
            result[source] = int(numeric)
    return result


def audit_graph(path: Path, labels, expected_degree: int):
    size = path.stat().st_size
    invalid = self_loops = duplicates = overflow = edges = rows = 0
    incompatible = 0
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise ValueError("truncated graph header")
        declared_size, header_degree, start, frozen = struct.unpack("<QIIQ", header)
        if declared_size != size:
            raise ValueError(f"graph size {size} != header {declared_size}")
        while handle.tell() < size:
            degree_raw = handle.read(4)
            if len(degree_raw) != 4:
                raise ValueError("truncated graph degree")
            degree = struct.unpack("<I", degree_raw)[0]
            raw = handle.read(4 * degree)
            if len(raw) != 4 * degree:
                raise ValueError("truncated graph row")
            neighbors = struct.unpack(f"<{degree}I", raw) if degree else ()
            overflow += degree > expected_degree
            invalid += sum(neighbor >= len(labels) for neighbor in neighbors)
            self_loops += sum(neighbor == rows for neighbor in neighbors)
            duplicates += degree - len(set(neighbors))
            source_labels = set(labels[rows])
            incompatible += sum(
                neighbor < len(labels) and not source_labels.intersection(labels[neighbor])
                for neighbor in neighbors
            )
            edges += degree
            rows += 1
    return {
        "nodes": rows,
        "edges": edges,
        "average_degree": edges / rows,
        "header_max_degree": header_degree,
        "start_node": start,
        "frozen_points": frozen,
        "invalid_ids": invalid,
        "self_loops": self_loops,
        "duplicate_edges": duplicates,
        "degree_overflow_rows": overflow,
        "incompatible_label_edges": incompatible,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-prefix", type=Path, required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=int, required=True)
    parser.add_argument("--expected-labels", type=int, default=47)
    parser.add_argument("--max-degree", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefix = args.index_prefix
    required = {
        "graph": prefix,
        "data": Path(f"{prefix}.data"),
        "labels": Path(f"{prefix}_labels.txt"),
        "formatted_labels": Path(f"{prefix}_label_formatted.txt"),
        "label_map": Path(f"{prefix}_labels_map.txt"),
        "label_medoids": Path(f"{prefix}_labels_to_medoids.txt"),
    }
    missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("missing or empty sidecars: " + ", ".join(missing))

    labels = list(parse_rows(required["labels"]))
    formatted = list(parse_rows(required["formatted_labels"]))
    if len(labels) != args.expected_nodes or len(formatted) != args.expected_nodes:
        raise ValueError("label sidecar row count does not match expected nodes")
    if any(set(left) != set(right) for left, right in zip(labels, formatted)):
        raise ValueError("formatted and search label sidecars differ")

    label_map = parse_map(required["label_map"])
    medoids = {}
    with required["label_medoids"].open(encoding="utf-8") as handle:
        for line in handle:
            label, medoid = (int(value.strip()) for value in line.split(","))
            medoids[label] = medoid
    if len(label_map) != args.expected_labels or len(medoids) != args.expected_labels:
        raise ValueError("label map or medoid sidecar has the wrong label count")
    if set(medoids) != set(label_map.values()):
        raise ValueError("medoid labels do not match label map")
    bad_medoids = [
        label for label, medoid in medoids.items()
        if medoid >= len(labels) or label not in labels[medoid]
    ]
    if bad_medoids:
        raise ValueError("invalid label medoids: " + ",".join(map(str, bad_medoids)))

    graph = audit_graph(required["graph"], labels, args.max_degree)
    log = args.build_log.read_text(encoding="utf-8")
    timing_match = re.search(r"\[GPU stitched timing\].*?final_prune_seconds=([0-9.]+).*?total_seconds=([0-9.]+)", log)
    union_match = re.search(r"\[GPU stitched\] wrote graph .*?max_degree=(\d+)", log)
    if not timing_match or not union_match:
        raise ValueError("build log lacks GPU-Stitched timing or union-degree diagnostics")
    final_prune_seconds, total_seconds = map(float, timing_match.groups())
    union_max_degree = int(union_match.group(1))
    global_prune_executed = union_max_degree > args.max_degree and final_prune_seconds > 0

    failures = {
        key: value for key, value in graph.items()
        if key in {"invalid_ids", "self_loops", "duplicate_edges",
                   "degree_overflow_rows", "incompatible_label_edges"} and value != 0
    }
    if graph["nodes"] != args.expected_nodes:
        failures["nodes"] = graph["nodes"]
    if graph["header_max_degree"] != args.max_degree:
        failures["header_max_degree"] = graph["header_max_degree"]
    if not global_prune_executed:
        failures["global_prune_executed"] = False

    result = {
        "release_gate_pass": not failures,
        "graph": graph,
        "sidecars": {
            "all_present": True,
            "label_rows": len(labels),
            "formatted_label_rows": len(formatted),
            "label_map_entries": len(label_map),
            "label_medoid_entries": len(medoids),
            "valid_label_medoids": not bad_medoids,
        },
        "global_prune": {
            "executed": global_prune_executed,
            "pre_prune_max_degree": union_max_degree,
            "final_prune_seconds": final_prune_seconds,
        },
        "builder_total_seconds": total_seconds,
        "failures": failures,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
