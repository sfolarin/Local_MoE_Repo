#!/usr/bin/env python3
"""Join cached TorchSOAP features with frozen dense-gate teacher weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

EXPECTED = {"train": 369, "val": 38, "test": 105}
EXPERTS = [
    "ML-BKS", "Original_BKS", "Chik_BKS", "PSO1_BKS",
    "PSO2_BKS", "Tersoff", "Vashishta",
]


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--routing-dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    features_root = args.features.expanduser().resolve()
    teachers_root = args.teachers.expanduser().resolve()
    routing_root = args.routing_dataset.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output is non-empty; choose a new directory: {out}")

    feature_rows = read_csv(features_root / "manifest.csv")
    teacher_rows = read_csv(teachers_root / "teacher_manifest.csv")
    routing_rows = read_csv(routing_root / "dataset_manifest.csv")
    if len(feature_rows) != 512 or len(teacher_rows) != 512 or len(routing_rows) != 512:
        raise ValueError(
            f"Expected 512 rows: features={len(feature_rows)}, "
            f"teachers={len(teacher_rows)}, routing={len(routing_rows)}"
        )

    feature_by_id = {row["record_id"]: row for row in feature_rows}
    route_by_id = {row["record_id"]: row for row in routing_rows}
    teacher_by_id: Dict[str, Dict[str, str]] = {}
    for row in teacher_rows:
        # The teacher manifest inherits shard_file=shards/<record_id>.npz.
        rid = Path(row["shard_file"]).stem
        if rid in teacher_by_id:
            raise ValueError(f"Duplicate teacher record ID: {rid}")
        teacher_by_id[rid] = row

    ids = set(feature_by_id) & set(route_by_id) & set(teacher_by_id)
    if ids != set(feature_by_id) or ids != set(route_by_id) or ids != set(teacher_by_id):
        raise ValueError("Feature, teacher, and routing manifests have different record IDs")

    counts = Counter(row["split"] for row in feature_rows)
    if dict(counts) != EXPECTED:
        raise ValueError(f"Frozen split changed: {dict(counts)}; expected {EXPECTED}")

    out.mkdir(parents=True)
    (out / "features").mkdir()
    manifest: List[Dict[str, str]] = []

    for rid in sorted(ids, key=lambda x: (feature_by_id[x]["split"], x)):
        frow = feature_by_id[rid]
        trow = teacher_by_id[rid]
        rrow = route_by_id[rid]
        if frow["split"] != trow["split"] or frow["split"] != rrow["split"]:
            raise ValueError(f"Split mismatch for {rid}")
        feature_path = resolve(features_root, frow["feature_file"])
        teacher_path = resolve(teachers_root, trow["teacher_file"])
        expert_path = resolve(routing_root, rrow["expert_file"])
        target = out / "features" / f"{rid}.npz"

        with np.load(feature_path, allow_pickle=False) as fdata:
            x = np.asarray(fdata["X_local_torchsoap"], dtype=np.float32)
        with np.load(teacher_path, allow_pickle=False) as tdata:
            teacher = np.asarray(tdata["teacher_weights"], dtype=np.float32)
            phase = np.asarray(
                tdata["phase"] if "phase" in tdata.files else np.full(len(x), -1),
                dtype=np.int16,
            ).reshape(-1)
            center = np.asarray(
                tdata["interface_center"]
                if "interface_center" in tdata.files
                else np.full(len(x), -1),
                dtype=np.int16,
            ).reshape(-1)
            names = [str(v) for v in tdata["expert_names"].tolist()]

        n_atoms = int(rrow["n_atoms"])
        if x.shape != (n_atoms, 392):
            raise ValueError(f"{rid}: invalid feature shape {x.shape}")
        if teacher.shape != (n_atoms, 7):
            raise ValueError(f"{rid}: invalid teacher shape {teacher.shape}")
        if phase.shape != (n_atoms,) or center.shape != (n_atoms,):
            raise ValueError(f"{rid}: invalid region-mask shape")
        if names != EXPERTS:
            raise ValueError(f"{rid}: expert order mismatch: {names}")
        if not np.isfinite(x).all() or not np.isfinite(teacher).all():
            raise ValueError(f"{rid}: non-finite feature or teacher values")
        if np.max(np.abs(teacher.sum(axis=1) - 1.0)) > 5e-6:
            raise ValueError(f"{rid}: teacher weights are not simplex-normalized")
        if not expert_path.is_file():
            raise FileNotFoundError(expert_path)

        np.savez_compressed(
            target,
            X_local_torchsoap=x,
            teacher_weights=teacher,
            phase=phase,
            interface_center=center,
            expert_names=np.asarray(EXPERTS),
            record_id=np.asarray(rid),
            split=np.asarray(frow["split"]),
        )
        manifest.append({
            # Use the unique record ID as source_file.  Original source names
            # such as Oishi's ``relaxed`` are not unique.
            "source_file": rid,
            "original_source_file": trow["source_file"],
            "record_id": rid,
            "phase_pair": rrow["phase_pair"],
            "split": frow["split"],
            "n_atoms": str(n_atoms),
            "output_file": str(target.relative_to(out)),
            "source_shard": str(expert_path),
            "feature_file": str(feature_path),
            "teacher_file": str(teacher_path),
        })

    write_csv(out / "manifest.csv", manifest)
    contract = {
        "stage": "expanded_torchsoap_distillation_dataset_v1",
        "features_root": str(features_root),
        "teachers_root": str(teachers_root),
        "routing_dataset": str(routing_root),
        "counts": dict(EXPECTED),
        "source_counts": dict(Counter(r["original_source_file"].split("/", 1)[0] for r in manifest)),
        "feature_manifest_sha256": digest(features_root / "manifest.csv"),
        "teacher_manifest_sha256": digest(teachers_root / "teacher_manifest.csv"),
        "routing_manifest_sha256": digest(routing_root / "dataset_manifest.csv"),
        "source_file_policy": "unique_record_id; original source retained in original_source_file",
    }
    (out / "distillation_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"Prepared: {out}")
    print(f"Frozen counts: {dict(counts)}")
    print(f"Feature shards: {len(manifest)}")
    print("No training performed; teacher weights and cached features were only joined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
