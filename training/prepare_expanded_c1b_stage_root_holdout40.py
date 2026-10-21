#!/usr/bin/env python3
"""Prepare a manifest compatible with the legacy C1b conservative trainer.

The expanded dataset manifest uses ``data_file`` and the feature cache uses
``feature_file``.  C1b expects ``output_file`` containing geometry/region
arrays and a separate ``source_shard`` containing MATPes energy/forces.  This
adapter creates geometry-only NPZs and leaves the original MATPes shards
untouched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output is non-empty; choose a new directory: {out}")

    rows = read_csv(dataset / "manifest.csv")
    if len(rows) != 512:
        raise ValueError(f"Expected 512 expanded records, found {len(rows)}")
    counts = Counter(row["split"] for row in rows)
    if dict(counts) != EXPECTED:
        raise ValueError(f"Frozen split changed: {dict(counts)}; expected {EXPECTED}")

    for key in ("phase_pair_group", "split_group", "geometry_sha256"):
        groups = defaultdict(set)
        for row in rows:
            groups[row[key]].add(row["split"])
        if any(len(splits) != 1 for splits in groups.values()):
            raise ValueError(f"Cross-split conflict in {key}")

    out.mkdir(parents=True)
    (out / "geometry").mkdir()
    manifest: List[Dict[str, str]] = []

    for row in sorted(rows, key=lambda r: (r["split"], r["record_id"])):
        rid = row["record_id"]
        source = (dataset / row["data_file"]).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as data:
            required = {"atomic_numbers", "positions", "cell"}
            missing = required - set(data.files)
            if missing:
                raise ValueError(f"{source}: missing geometry keys {sorted(missing)}")
            z = np.asarray(data["atomic_numbers"], dtype=np.int16).reshape(-1)
            positions = np.asarray(data["positions"], dtype=np.float64)
            cell = np.asarray(data["cell"], dtype=np.float64)
            n = len(z)
            phase = np.asarray(
                data["phase"] if "phase" in data.files else np.full(n, -1),
                dtype=np.int16,
            ).reshape(-1)
            center = np.asarray(
                data["interface_center"]
                if "interface_center" in data.files
                else np.full(n, -1),
                dtype=np.int16,
            ).reshape(-1)

        if n != int(row["n_atoms"]) or positions.shape != (n, 3) or cell.shape != (3, 3):
            raise ValueError(f"{rid}: geometry shape mismatch")
        if phase.shape != (n,) or center.shape != (n,):
            raise ValueError(f"{rid}: region-array shape mismatch")
        if not np.isfinite(positions).all() or not np.isfinite(cell).all():
            raise ValueError(f"{rid}: non-finite geometry")

        relative = Path("geometry") / f"{rid}.npz"
        target = out / relative
        np.savez_compressed(
            target,
            atomic_numbers=z,
            positions=positions,
            cell=cell,
            phase=phase,
            interface_center=center,
            record_id=np.asarray(rid),
            split=np.asarray(row["split"]),
            geometry_sha256=np.asarray(row["geometry_sha256"]),
        )
        manifest.append({
            "source_file": rid,
            "phase_pair": row["phase_pair_group"],
            "split": row["split"],
            "n_atoms": row["n_atoms"],
            "output_file": str(relative),
            "source_shard": str(source),
            "geometry_sha256": row["geometry_sha256"],
            "original_source": row["source_record"],
            "dataset_source": row["source"],
        })

    write_csv(out / "manifest.csv", manifest)
    contract = {
        "stage": "expanded_c1b_stage_root_v1",
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256(dataset / "manifest.csv"),
        "counts": dict(EXPECTED),
        "reference_pes": "MATPes-r2SCAN",
        "source_shard_policy": "frozen expanded MATPes geometry/label shards",
        "region_policy": "preserve phase/interface arrays; use -1 for unavailable external masks",
    }
    (out / "stage_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"Prepared: {out}")
    print(f"Frozen counts: {dict(counts)}")
    print("Geometry files: 512")
    print("MATPes source shards: 512")
    print("No labels, geometries, or splits were modified in the source dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
