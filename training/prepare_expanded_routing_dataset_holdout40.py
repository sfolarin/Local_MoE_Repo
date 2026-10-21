#!/usr/bin/env python3
"""Adapt the frozen expanded expert run for dense-gate training.

The expert-preparation run is authoritative for the seven classical-force
arrays.  This adapter creates only a small manifest and a symlink to those
shards; it does not copy, relabel, resplit, relax, or modify any structure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

EXPECTED = {"train": 369, "val": 38, "test": 105}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True,
                        help="dataset root, e.g. data/expanded_holdout40_v1")
    parser.add_argument("--expert-run", type=Path, required=True,
                        help="completed expert run, e.g. data/expanded_holdout40_v1_experts_v2")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    expert = args.expert_run.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output is non-empty; choose a new directory: {out}")

    source_rows = read_csv(dataset / "manifest.csv")
    completed_rows = read_csv(expert / "completed_expert_manifest.csv")
    if len(source_rows) != 512 or len(completed_rows) != 512:
        raise ValueError(f"Expected 512 rows, found dataset={len(source_rows)} expert={len(completed_rows)}")

    source_by_id = {row["record_id"]: row for row in source_rows}
    expert_by_id = {row["record_id"]: row for row in completed_rows}
    if set(source_by_id) != set(expert_by_id):
        raise ValueError("Dataset and completed expert manifests have different record IDs")

    counts = Counter(row["split"] for row in source_rows)
    if dict(counts) != EXPECTED:
        raise ValueError(f"Frozen split changed: {dict(counts)}; expected {EXPECTED}")
    if any(row.get("status", "PASS") != "PASS" for row in completed_rows):
        raise ValueError("Completed expert manifest contains a non-PASS row")

    for key in ("phase_pair_group", "split_group", "geometry_sha256"):
        groups = defaultdict(set)
        for row in source_rows:
            groups[row[key]].add(row["split"])
        bad = {group: sorted(splits) for group, splits in groups.items() if len(splits) != 1}
        if bad:
            raise ValueError(f"Cross-split conflict in {key}: {list(bad)[:3]}")

    out.mkdir(parents=True)
    shard_link = out / "shards"
    os.symlink(expert / "shards", shard_link, target_is_directory=True)

    rows = []
    for source in sorted(source_rows, key=lambda r: (r["split"], r["record_id"])):
        rid = source["record_id"]
        expert_row = expert_by_id[rid]
        # The legacy trainer only uses this field for reporting/grouping.  The
        # frozen split is authoritative and is enforced by the expanded copy.
        rows.append({
            "record_id": rid,
            "source_file": f"{source['source']}/{source['source_record']}",
            "output_file": f"{rid}.npz",
            "phase_pair": source["phase_pair_group"],
            "duplicate_group": source["split_group"],
            "n_atoms": source["n_atoms"],
            "best_expert_all": expert_row.get("best_expert_all", ""),
            "split": source["split"],
            "geometry_sha256": source["geometry_sha256"],
            "expert_file": expert_row["expert_file"],
            "expert_sha256": expert_row["expert_sha256"],
        })
    write_csv(out / "dataset_manifest.csv", rows)

    contract = {
        "stage": "expanded_dense_gate_dataset_v1",
        "dataset_root": str(dataset),
        "expert_run": str(expert),
        "dataset_manifest_sha256": sha256(dataset / "manifest.csv"),
        "expert_manifest_sha256": sha256(expert / "completed_expert_manifest.csv"),
        "counts": dict(EXPECTED),
        "shards": "symlink_to_completed_expert_run",
        "split_policy": "preserve_expanded_manifest_exactly; no regrouping",
        "source_counts": dict(Counter(row["source"] for row in source_rows)),
    }
    (out / "routing_dataset_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"Prepared: {out}")
    print(f"Shard link: {shard_link} -> {expert / 'shards'}")
    print(f"Frozen counts: {dict(counts)}")
    print(f"Source counts: {dict(Counter(row['source'] for row in source_rows))}")
    print("No structures copied or modified; no splits regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
