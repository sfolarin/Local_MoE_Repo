#!/usr/bin/env python3
"""Prepare, label and summarize the expanded Si/O Local-MoE configurations.

prepare: CPU geometry QC and provenance; never loads MACE or assigns splits.
label:   fixed-geometry MACE-MATPES-r2SCAN energy/force single points on a GPU.
summary: verify completed label files, retaining failed/pending/QC-held cases.
recover-parser-holds: recheck Oishi names with cut/registry suffixes, preserving
                      existing labels and backing up the old manifests first.

Requires NumPy and ASE; label additionally requires the existing MACE/PyTorch
environment. Run label inside a GPU allocation. Rank/world default to Slurm.
All input directories are read-only. An output run must be new at preparation.
The NPZ label files are authoritative float64 geometries, energies and forces.
Historical EXTXYZ energies/forces are never used as new training targets.
Exact geometry hashes preserve atom order and do not establish symmetry-based
equivalence. Phase-pair groups are conservative hints for a later split audit.
This stage does not calculate gamma, relax structures, or retrain Local-MoE.
"""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import socket
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TEACHER = "mace-matpes-r2scan-0"
EXTERNAL = {"Oishi_all179", "Oishi_relaxed52", "Siddharth"}
GEOMETRY_KEYS = ("atomic_numbers", "positions", "cell", "pbc")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def atomic_write(path, writer):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        writer(temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda p: p.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))

    def save(p):
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    atomic_write(path, save)


def write_npz(path, data):
    def save(p):
        with p.open("wb") as f:
            np.savez_compressed(f, **data)
    atomic_write(path, save)


def snapshot(src, dst):
    before = sha256(src)
    shutil.copy2(src, dst)
    if sha256(dst) != before or sha256(src) != before:
        raise RuntimeError("Input changed while copying: " + str(src))
    return before


def geometry(data):
    """Only geometry: no old calculator, energy, force, constraint or metadata."""
    return dict(atomic_numbers=np.asarray(data["atomic_numbers"], dtype="<i8").reshape(-1),
                positions=np.asarray(data["positions"], dtype="<f8"),
                cell=np.asarray(data["cell"], dtype="<f8"),
                pbc=np.asarray(data["pbc"], dtype=np.bool_).reshape(3))


def from_npz(path, legacy_periodic=False):
    with np.load(path, allow_pickle=False) as d:
        data = {k: d[k] for k in GEOMETRY_KEYS if k in d}
        if "pbc" not in data and legacy_periodic:
            # Matches the existing periodic TorchSOAP trainer for the old B1 data.
            data["pbc"] = np.ones(3, dtype=bool)
        return geometry(data)


def geometry_hash(g):
    h = hashlib.sha256(b"ordered-geometry-f64-v1\0")
    for key in GEOMETRY_KEYS:
        a = np.ascontiguousarray(g[key])
        h.update(key.encode() + b"\0" + str(a.dtype).encode() + b"\0")
        h.update(json.dumps(a.shape).encode() + b"\0" + a.tobytes())
    return h.hexdigest()


def basic_qc(g, expected_n, max_atoms):
    z, r, c, pbc = (g[k] for k in GEOMETRY_KEYS)
    n = len(z)
    if n != expected_n or not 0 < n <= max_atoms:
        raise ValueError("Atom count mismatch or outside size limit: " + str(n))
    if r.shape != (n, 3) or c.shape != (3, 3):
        raise ValueError("Invalid positions/cell shape")
    if not np.isfinite(r).all() or not np.isfinite(c).all():
        raise ValueError("Nonfinite coordinates/cell")
    if not pbc.all() or abs(np.linalg.det(c)) < 1e-6:
        raise ValueError("Requires a nonsingular, fully periodic cell")
    n_si, n_o = int((z == 14).sum()), int((z == 8).sum())
    if n_si == 0 or n_si + n_o != n or n_o != 2 * n_si:
        raise ValueError("Requires Si/O only, with exact SiO2 stoichiometry")


def as_atoms(g):
    from ase import Atoms
    return Atoms(numbers=g["atomic_numbers"], positions=g["positions"],
                 cell=g["cell"], pbc=g["pbc"])


def neighbor_qc(g, overlap, bond_cutoff):
    from ase.neighborlist import neighbor_list
    z = g["atomic_numbers"]
    i, j, distances = neighbor_list("ijd", as_atoms(g),
                                    max(3.0, bond_cutoff), self_interaction=False)
    min_dist = float(distances.min()) if len(distances) else None
    if min_dist is not None and min_dist < overlap:
        raise ValueError("Gross periodic overlap: minimum distance {:.6f} A".format(min_dist))
    unlike = (z[i] != z[j]) & (distances < bond_cutoff)
    cn = np.bincount(i[unlike], minlength=len(z))
    si4 = float(np.mean(cn[z == 14] == 4))
    o2 = float(np.mean(cn[z == 8] == 2))
    return dict(min_distance_within_search_A=min_dist, Si4_fraction=si4,
                O2_fraction=o2, coordination_review=si4 < 1.0 or o2 < 1.0)


def pair_key(source, family):
    if source.startswith("Oishi"):
        # Both forms occur in the supplied Oishi export. The longer form stores
        # a cut shift after each Miller plane and a final registry suffix:
        # NAT_1_0_0_0.1916085__THO_0_1_0_0.3867520000000001__0_0.
        # Preserve that full source name elsewhere; only extract the two phases.
        plane = r"([^_]+)_[+-]?\d+_[+-]?\d+_[+-]?\d+"
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        match = re.fullmatch(plane + "_" + plane, family)
        if match is None:
            match = re.fullmatch(plane + "_" + number + "__" + plane + "_" + number
                                 + "(?:__" + number + "_" + number + ")?", family)
        if match is None:
            raise ValueError("Unrecognized Oishi case: " + family)
        phases = list(match.groups())
    elif source == "Siddharth":
        phases = family.split("_")
    else:
        phases = [re.sub(r"^\d+_", "", p) for p in family.split("__")]
    if len(phases) != 2:
        raise ValueError("Unrecognized phase pair: " + family)
    phases = [p.upper().removesuffix("_F") if p.upper() == "MFI_F"
              else p.upper() for p in phases]
    return "__".join(sorted(phases))


def write_group_reports(run, own_rows, records):
    groups, hashes = defaultdict(list), defaultdict(list)
    for row in own_rows + records:
        if row.get("phase_pair_group"):
            groups[row["phase_pair_group"]].append(row)
        if row.get("geometry_sha256"):
            hashes[row["geometry_sha256"]].append(row)
    group_rows = []
    for group, rows in sorted(groups.items()):
        counts = Counter(r["source"] for r in rows)
        splits = sorted({r["existing_split"] for r in rows if r["source"] == "Own_augmented"})
        group_rows.append(dict(phase_pair_group=group, counts_by_source=json.dumps(dict(counts)),
                               existing_own_splits=";".join(splits),
                               touches_old_holdout=any(s in {"val", "test"} for s in splits),
                               related_record_ids=";".join(r["record_id"] for r in rows)))
    write_csv(run / "phase_pair_overlap.csv", group_rows)
    write_csv(run / "exact_geometry_overlaps.csv", [
        dict(geometry_sha256=h, record_ids=";".join(r["record_id"] for r in rows),
             sources=";".join(r["source"] for r in rows))
        for h, rows in sorted(hashes.items()) if len(rows) > 1],
        ["geometry_sha256", "record_ids", "sources"])


def prepare(args):
    from ase.io import read
    prep, run = args.prep.resolve(), args.run.resolve()
    previous = json.loads((prep / "preparation_summary.json").read_text())
    if previous.get("issues"):
        raise RuntimeError("Resolve preparation issues: " + str(previous["issues"]))
    cap = int(previous["size_limit_atoms"])
    selected = read_csv(prep / "manifests/size_eligible.csv")
    external = [r for r in selected if r["source"] in EXTERNAL]
    if not external or any(int(r["n_atoms"]) > cap for r in selected):
        raise ValueError("Missing external records or inconsistent size-screen manifest")
    unknown = {r["source"] for r in selected} - EXTERNAL - {"Own_augmented"}
    if unknown:
        raise ValueError("Unsupported sources: " + str(unknown))
    own_copy = prep / "baseline/own_augmented_manifest.csv"
    entries = [r for r in read_csv(prep / "snapshot_provenance.csv")
               if Path(r["snapshot_path"]).resolve() == own_copy]
    if len(entries) != 1 or sha256(own_copy) != entries[0]["sha256"]:
        raise RuntimeError("Original augmented manifest provenance did not verify")
    own_root = Path(entries[0]["original_path"]).parent
    own = read_csv(own_copy)
    if len(own) != sum(r["source"] == "Own_augmented" for r in selected):
        raise ValueError("Own-data manifest and size screen disagree")

    run.mkdir(parents=True, exist_ok=False)
    for sub in ("provenance", "raw", "geometries", "anchors", "labels", "status", "logs"):
        (run / sub).mkdir()
    for src in (prep / "manifests/size_eligible.csv", prep / "manifests/size_deferred.csv",
                prep / "snapshot_provenance.csv", own_copy, Path(__file__).resolve()):
        snapshot(src, run / "provenance" / src.name)

    own_rows, anchor_candidates = [], []
    for index, row in enumerate(own):
        feature = (own_root / row["output_file"]).resolve()
        g = from_npz(feature, legacy_periodic=True)
        basic_qc(g, int(row["n_atoms"]), cap)
        own_rows.append(dict(source="Own_augmented", record_id="own_{:04d}".format(index),
                             record=row["source_file"], family=row["phase_pair"],
                             phase_pair_group=pair_key("Own_augmented", row["phase_pair"]),
                             existing_split=row["split"], n_atoms=len(g["atomic_numbers"]),
                             geometry_sha256=geometry_hash(g), feature_file=str(feature),
                             source_shard=row["source_shard"],
                             pbc_note="fully_periodic_as_in_existing_trainer"))
        if row["split"] == "train":
            anchor_candidates.append((int(row["n_atoms"]), row["source_file"], row, feature))
    if len(anchor_candidates) < 2:
        raise ValueError("Need two existing training records for teacher consistency checks")
    anchor_candidates.sort(key=lambda x: (x[0], x[1]))
    anchor_rows = []
    for idx, (_, _, row, feature) in enumerate((anchor_candidates[0], anchor_candidates[-1])):
        shard = Path(row["source_shard"]).expanduser().resolve()
        g = from_npz(feature, legacy_periodic=True)
        with np.load(shard, allow_pickle=False) as d:
            e = float(np.asarray(d["mace_energy_total_eV"]).item())
            f = np.array(d["mace_forces"], dtype=np.float64)
        check_labels(e, f, len(g["atomic_numbers"]))
        path = run / "anchors" / ("anchor_{:02d}.npz".format(idx))
        write_npz(path, dict(g, mace_energy_total_eV=e, mace_forces=f))
        anchor_rows.append(dict(source_file=row["source_file"], n_atoms=len(f),
                                anchor_file=str(path), anchor_sha256=sha256(path),
                                feature_file=str(feature), feature_sha256=sha256(feature),
                                source_shard=str(shard), source_shard_sha256=sha256(shard)))
    write_csv(run / "anchors/manifest.csv", anchor_rows)
    write_csv(run / "own_records.csv", own_rows)

    records, unique = [], {}
    for index, src in enumerate(external):
        rid = "external_{:04d}".format(index)
        row = dict(src, record_id=rid, existing_split="unassigned",
                   geometry_sha256="", geometry_file="", qc_status="HOLD", qc_error="")
        try:
            origin = Path(src["origin"]).resolve()
            digest = sha256(origin)
            raw = run / "raw" / (digest + ".extxyz")
            if not raw.exists():
                if snapshot(origin, raw) != digest:
                    raise RuntimeError("Source changed during snapshot")
            if sha256(raw) != digest:
                raise RuntimeError("Raw snapshot hash mismatch")
            row.update(source_file_sha256=digest, raw_snapshot=str(raw),
                       phase_pair_group=pair_key(src["source"], src["family"]))
            row["lineage_group"] = ("Oishi:" + src["family"] if src["source"].startswith("Oishi")
                                    else "Siddharth:" + src["family"] + ":" + src["record"])
            a = read(raw, index=int(src["frame"] or 0), format="extxyz")
            g = geometry(dict(atomic_numbers=a.numbers, positions=a.positions,
                              cell=np.asarray(a.cell), pbc=a.pbc))
            basic_qc(g, int(src["n_atoms"]), cap)
            row.update(neighbor_qc(g, args.overlap_A, args.bond_cutoff_A))
            gh = geometry_hash(g)
            path = run / "geometries" / (gh + ".npz")
            if gh not in unique:
                write_npz(path, g)
                unique[gh] = dict(geometry_sha256=gh, geometry_file=str(path),
                                  geometry_file_sha256=sha256(path), n_atoms=len(a),
                                  first_record_id=rid)
            row.update(geometry_sha256=gh, geometry_file=str(path), qc_status="PASS")
        except Exception as exc:
            row["qc_error"] = type(exc).__name__ + ": " + str(exc)
        records.append(row)
        print("QC {}/{} {} {}".format(index + 1, len(external), rid, row["qc_status"]), flush=True)
    write_csv(run / "external_records.csv", records)
    queue = sorted(unique.values(), key=lambda r: (r["n_atoms"], r["geometry_sha256"]))
    write_csv(run / "label_queue.csv", queue,
              ["geometry_sha256", "geometry_file", "geometry_file_sha256", "n_atoms", "first_record_id"])
    write_group_reports(run, own_rows, records)
    config = dict(created=now(), preparation=str(prep), size_limit_atoms=cap,
                  teacher=TEACHER, dtype="float64", dispersion=False,
                  overlap_hold_threshold_A=args.overlap_A, coordination_cutoff_A=args.bond_cutoff_A,
                  anchor_energy_tolerance_eV_per_atom=1e-5, anchor_force_rmse_tolerance_eV_A=1e-4,
                  external_records=len(records), unique_qc_pass_geometries=len(queue),
                  qc_counts=dict(Counter(r["qc_status"] for r in records)),
                  source_counts=dict(Counter(r["source"] for r in records)),
                  own_records=len(own_rows), old_splits_preserved_as_metadata=True,
                  new_splits_assigned=False, own_labels_fully_audited=False,
                  notes=["Coordination flags are for review, not automatic rejection.",
                         "0.5 A default overlap threshold checks gross corruption, not bonding quality.",
                         "Hashes detect exact ordered geometries, not symmetry/permutation equivalence.",
                         "Old holdout overlap must be resolved before expanded training.",
                         "Oishi originals and relaxed variants are related configurations.",
                         "AB labels alone do not define interface energies."])
    for name in ("label_queue.csv", "anchors/manifest.csv", "external_records.csv"):
        config[name + "_sha256"] = sha256(run / name)
    write_json(run / "preparation.json", config)  # Completion marker, written last.
    print("\nPrepared: " + str(run))
    print("External records by source: " + json.dumps(config["source_counts"], sort_keys=True))
    print("Geometry QC: " + json.dumps(config["qc_counts"]))
    print("Unique QC-pass geometries to label: " + str(len(queue)))
    print("Coordination review flags: " + str(sum(r.get("coordination_review", False) for r in records)))
    print("All new splits remain unassigned. No calculator loaded or labels generated.")
    return 0 if queue else 2


def recover_parser_holds(args):
    """Recheck only records previously held by the old Oishi name parser.

    Run after labeling workers have finished. Existing geometry, label, status,
    anchor and teacher files are not modified. Manifests are backed up and the
    queue is extended only after the original geometry checks pass. No MACE.
    """
    from ase.io import read
    run = args.run.resolve()
    cfg = load_config(run)
    records = read_csv(run / "external_records.csv")
    selected = [r for r in records if r["source"].startswith("Oishi")
                and r["qc_status"] != "PASS"
                and r["qc_error"].startswith("ValueError: Unrecognized Oishi case:")]
    if not selected:
        print("No remaining Oishi filename-parser holds; nothing changed.")
        return 0
    queue_rows = read_csv(run / "label_queue.csv")
    queued = {q["geometry_sha256"]: q for q in queue_rows}
    report, new_geometries = [], {}
    for row in selected:
        before = dict(row)
        try:
            pair = pair_key(row["source"], row["family"])
            # The old parser failed while evaluating row.update arguments, so
            # raw_snapshot/source_file_sha256 may not have been recorded.
            # In that case require a matching file ALREADY in the raw snapshot.
            digest = row.get("source_file_sha256") or sha256(Path(row["origin"]))
            raw = run / "raw" / (digest + ".extxyz")
            if not raw.is_file() or sha256(raw) != digest:
                raise ValueError("Cannot verify this record against its original raw snapshot")
            row.update(phase_pair_group=pair, source_file_sha256=digest,
                       raw_snapshot=str(raw), lineage_group="Oishi:" + row["family"])
            a = read(raw, index=int(row["frame"] or 0), format="extxyz")
            g = geometry(dict(atomic_numbers=a.numbers, positions=a.positions,
                              cell=np.asarray(a.cell), pbc=a.pbc))
            basic_qc(g, int(row["n_atoms"]), cfg["size_limit_atoms"])
            row.update(neighbor_qc(g, cfg["overlap_hold_threshold_A"], cfg["coordination_cutoff_A"]))
            gh = geometry_hash(g)
            path = run / "geometries" / (gh + ".npz")
            if gh not in queued:
                # Write after the manifest backup. Reuse an earlier file only
                # when its stored geometry verifies exactly.
                if path.exists() and geometry_hash(from_npz(path)) != gh:
                    raise ValueError("Existing geometry file does not match its name")
                new_geometries[gh] = (g, path, row["record_id"])
            row.update(geometry_sha256=gh, geometry_file=str(path), qc_status="PASS", qc_error="")
        except Exception as exc:
            row.update(qc_status="HOLD", qc_error=type(exc).__name__ + ": " + str(exc))
        report.append(dict(record_id=row["record_id"], previous=before, current=dict(row)))
        print("{} {} N={} {} {}".format(row["record_id"], row.get("phase_pair_group", ""),
                                        row["n_atoms"], row["qc_status"], row["qc_error"]), flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    backup = run / "provenance" / ("parser_recovery_" + stamp)
    backup.mkdir(exist_ok=False)
    before_hashes = {}
    for name in ("preparation.json", "external_records.csv", "label_queue.csv", "own_records.csv",
                 "phase_pair_overlap.csv", "exact_geometry_overlaps.csv", "label_status_by_record.csv",
                 "completed_unique_labels.csv", "label_summary.json"):
        src = run / name
        if src.exists():
            before_hashes[name] = snapshot(src, backup / name)
    snapshot(Path(__file__).resolve(), backup / Path(__file__).name)
    write_json(backup / "recovery_report.json", dict(created=now(), records=report, before_hashes=before_hashes))
    marker = run / "parser_recovery_in_progress.json"
    write_json(marker, dict(backup=str(backup), started=now()))
    for gh, (g, path, record_id) in new_geometries.items():
        if not path.exists():
            write_npz(path, g)
        queued[gh] = dict(geometry_sha256=gh, geometry_file=str(path), geometry_file_sha256=sha256(path),
                          n_atoms=len(g["atomic_numbers"]), first_record_id=record_id)
    queue_rows = sorted(queued.values(), key=lambda q: (int(q["n_atoms"]), q["geometry_sha256"]))
    write_csv(run / "external_records.csv", records)
    write_csv(run / "label_queue.csv", queue_rows,
              ["geometry_sha256", "geometry_file", "geometry_file_sha256", "n_atoms", "first_record_id"])
    write_group_reports(run, read_csv(run / "own_records.csv"), records)
    cfg.update(qc_counts=dict(Counter(r["qc_status"] for r in records)),
               unique_qc_pass_geometries=len(queue_rows))
    cfg.setdefault("parser_recoveries", []).append(dict(backup=str(backup), finished=now(),
        records_rechecked=len(selected), qc_pass=sum(r["qc_status"] == "PASS" for r in selected)))
    for name in ("label_queue.csv", "anchors/manifest.csv", "external_records.csv"):
        cfg[name + "_sha256"] = sha256(run / name)
    write_json(run / "preparation.json", cfg)
    marker.unlink()
    print("Previous manifests and recovery provenance: " + str(backup))
    print("Existing labels preserved. Unique labeling queue: " + str(len(queue_rows)))
    summary(args)
    return 2 if any(r["qc_status"] != "PASS" for r in selected) else 0


def check_labels(e, f, n):
    if not np.isfinite(e) or f.shape != (n, 3) or not np.isfinite(f).all():
        raise ValueError("Energy/force output is nonfinite or has an invalid shape")


def evaluate(calc, g):
    a = as_atoms(g)
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(apply_constraint=False), dtype=np.float64)
    check_labels(e, f, len(a))
    return e, f


def model_state_hash(calc):
    models = getattr(calc, "models", None)
    if models is None or len(models) != 1:
        raise ValueError("Expected one MACE model for an unambiguous teacher fingerprint")
    h = hashlib.sha256()
    for name, tensor in sorted(models[0].state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode() + b"\0" + str(a.dtype).encode() + b"\0")
        h.update(json.dumps(a.shape).encode() + b"\0" + a.tobytes())
    return h.hexdigest()


def versions():
    result = {}
    for name in ("mace-torch", "torch", "e3nn", "ase", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unknown"
    return result


def publish_contract(path, contract):
    # Publish a complete file atomically. Parallel ranks compare to the winner.
    temp = path.with_name(path.name + ".candidate." + uuid.uuid4().hex)
    try:
        write_json(temp, contract)
        try:
            os.link(temp, path)
        except FileExistsError:
            pass
        if json.loads(path.read_text()) != contract:
            raise RuntimeError("Teacher/configuration differs from this run's saved contract")
    finally:
        temp.unlink(missing_ok=True)


def load_config(run):
    marker = run / "parser_recovery_in_progress.json"
    if marker.exists():
        raise RuntimeError("Manifest recovery was interrupted. Restore the manifest files listed in "
                           "recovery_report.json from this backup before retrying: " + marker.read_text())
    cfg = json.loads((run / "preparation.json").read_text())
    if cfg["teacher"] != TEACHER:
        raise ValueError("Wrong teacher in preparation")
    for name in ("label_queue.csv", "anchors/manifest.csv", "external_records.csv"):
        if sha256(run / name) != cfg[name + "_sha256"]:
            raise RuntimeError("Prepared manifest changed: " + name)
    return cfg


def completed(run, q, contract):
    marker = run / "status" / (q["geometry_sha256"] + ".json")
    if not marker.exists():
        return False, "PENDING"
    try:
        s = json.loads(marker.read_text())
        if s["status"] != "PASS":
            return False, s["status"]
        if contract is None or s["teacher_contract"] != contract:
            return False, "INVALID_TEACHER_CONTRACT"
        path = run / "labels" / (q["geometry_sha256"] + ".npz")
        if sha256(path) != s["label_sha256"]:
            return False, "INVALID_OUTPUT_HASH"
        with np.load(path, allow_pickle=False) as d:
            if geometry_hash(geometry(d)) != q["geometry_sha256"]:
                return False, "INVALID_GEOMETRY"
            if str(d["teacher_state_sha256"].item()) != contract["teacher_state_sha256"]:
                return False, "INVALID_MODEL_STATE"
            if str(d["teacher"].item()) != TEACHER:
                return False, "INVALID_TEACHER"
            check_labels(float(d["mace_energy_total_eV"].item()), d["mace_forces"], int(q["n_atoms"]))
        return True, "PASS"
    except Exception as exc:
        return False, "INVALID_OUTPUT:" + type(exc).__name__


def label(args):
    import torch
    from mace.calculators import mace_mp
    run = args.run.resolve()
    cfg = load_config(run)
    if not 0 <= args.rank < args.world or args.world < 1:
        raise ValueError("Invalid worker rank/world")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible; run label inside a GPU allocation")
    torch.set_num_threads(args.threads)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    prefix = run / "logs" / ("rank{}_{}".format(args.rank, stamp))
    calc = mace_mp(model=TEACHER, device="cuda", default_dtype="float64", dispersion=False)
    contract = dict(teacher=TEACHER, dtype="float64", dispersion=False,
                    teacher_state_sha256=model_state_hash(calc), versions=versions(),
                    model_class=type(calc.models[0]).__module__ + "." + type(calc.models[0]).__name__,
                    head=str(getattr(calc, "head", "Default")))
    anchor_results = []
    for row in read_csv(run / "anchors/manifest.csv"):
        path = Path(row["anchor_file"])
        if sha256(path) != row["anchor_sha256"]:
            raise RuntimeError("Anchor file changed: " + str(path))
        g = from_npz(path)
        e, f = evaluate(calc, g)
        with np.load(path, allow_pickle=False) as d:
            de = abs(e - float(d["mace_energy_total_eV"].item())) / len(f)
            df = float(np.sqrt(np.mean((f - d["mace_forces"]) ** 2)))
        passed = (de <= cfg["anchor_energy_tolerance_eV_per_atom"] and
                  df <= cfg["anchor_force_rmse_tolerance_eV_A"])
        anchor_results.append(dict(source_file=row["source_file"], n_atoms=len(f),
                                   abs_energy_error_eV_per_atom=de, force_rmse_eV_A=df,
                                   status="PASS" if passed else "FAIL"))
        print("rank={} teacher check {} dE/N={:.3e} dF_RMSE={:.3e}".format(
            args.rank, anchor_results[-1]["status"], de, df), flush=True)
    write_json(prefix.with_suffix(".teacher_check.json"),
               dict(contract=contract, anchors=anchor_results, gpu=torch.cuda.get_device_name(0),
                    host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"),
                    script_sha256=sha256(Path(__file__)), checked=now()))
    if any(r["status"] != "PASS" for r in anchor_results):
        raise RuntimeError("Teacher disagrees with old MATPes labels; review teacher_check.json. "
                           "No external labeling by this worker was started.")
    publish_contract(run / "teacher_contract.json", contract)
    queue = read_csv(run / "label_queue.csv")[args.rank::args.world]
    processed = skipped = failures = 0
    for q in queue:
        good, state = completed(run, q, contract)
        if good:
            skipped += 1
            continue
        if args.limit is not None and processed >= args.limit:
            break
        gh = q["geometry_sha256"]
        path = Path(q["geometry_file"])
        marker = run / "status" / (gh + ".json")
        start = time.perf_counter()
        status = dict(geometry_sha256=gh, n_atoms=int(q["n_atoms"]),
                      rank=args.rank, teacher_contract=contract, started=now())
        try:
            if sha256(path) != q["geometry_file_sha256"]:
                raise ValueError("Prepared geometry file changed")
            g = from_npz(path)
            if geometry_hash(g) != gh:
                raise ValueError("Prepared geometry hash mismatch")
            basic_qc(g, int(q["n_atoms"]), cfg["size_limit_atoms"])
            torch.cuda.reset_peak_memory_stats()
            e, f = evaluate(calc, g)
            torch.cuda.synchronize()
            out = run / "labels" / (gh + ".npz")
            write_npz(out, dict(g, mace_energy_total_eV=e, mace_forces=f,
                                teacher=TEACHER, teacher_state_sha256=contract["teacher_state_sha256"],
                                geometry_sha256=gh))
            status.update(status="PASS", label_file=str(out), label_sha256=sha256(out),
                          energy_eV=e, max_force_eV_A=float(np.linalg.norm(f, axis=1).max()),
                          peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3,
                          peak_reserved_GiB=torch.cuda.max_memory_reserved() / 1024**3)
        except Exception as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            status.update(status="CUDA_OOM" if oom else "ERROR", error=str(exc),
                          traceback=traceback.format_exc())
            failures += 1
        status.update(finished=now(), elapsed_s=time.perf_counter() - start)
        write_json(marker, status)  # Only PASS with verified output is complete.
        processed += 1
        print("rank={} {} N={} {} {:.2f}s".format(
            args.rank, q["first_record_id"], q["n_atoms"], status["status"], status["elapsed_s"]), flush=True)
        calc.results.clear()
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(rank=args.rank, world=args.world, processed=processed,
                  verified_existing_pass=skipped, failures=failures, finished=now())
    write_json(prefix.with_suffix(".worker.json"), result)
    print(json.dumps(result), flush=True)
    return 2 if failures else 0


def summary(args):
    run = args.run.resolve()
    cfg = load_config(run)
    contract_path = run / "teacher_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
    queue = read_csv(run / "label_queue.csv")
    states, completed_rows = {}, []
    for q in queue:
        good, state = completed(run, q, contract)
        states[q["geometry_sha256"]] = state
        if good:
            completed_rows.append(dict(q, label_file=str(run / "labels" / (q["geometry_sha256"] + ".npz"))))
    records = read_csv(run / "external_records.csv")
    report = []
    for row in records:
        state = "QC_HOLD" if row["qc_status"] != "PASS" else states.get(row["geometry_sha256"], "PENDING")
        report.append(dict(row, label_status=state,
                           label_file=str(run / "labels" / (row["geometry_sha256"] + ".npz")) if state == "PASS" else ""))
    write_csv(run / "label_status_by_record.csv", report)
    write_csv(run / "completed_unique_labels.csv", completed_rows,
              list(queue[0]) + ["label_file"] if queue else ["geometry_sha256", "label_file"])
    counts = Counter(states.values())
    for src in sorted(EXTERNAL):
        print("{}: {}".format(src, dict(Counter(r["label_status"] for r in report if r["source"] == src))))
    result = dict(updated=now(), external_records=len(records), unique_qc_pass_geometries=len(queue),
                  unique_label_status=dict(counts), record_status=dict(Counter(r["label_status"] for r in report)),
                  own_records=cfg["own_records"], existing_own_labels_reused_pending_final_dataset_audit=True,
                  all_external_records_labeled=bool(records) and all(r["label_status"] == "PASS" for r in report),
                  final_training_splits_assigned=False)
    write_json(run / "label_summary.json", result)
    print("Unique geometry labels: " + json.dumps(dict(counts), sort_keys=True))
    print("Completed-subset manifest: " + str(run / "completed_unique_labels.csv"))
    print("Status of every retained external record: " + str(run / "label_status_by_record.csv"))
    print("No train/validation/test assignments or gamma values have been created.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Snapshot inputs, check geometry, build unique labeling queue")
    prep.add_argument("--prep", type=Path, required=True)
    prep.add_argument("--run", type=Path, required=True)
    prep.add_argument("--overlap-A", dest="overlap_A", type=float, default=0.5)
    prep.add_argument("--bond-cutoff-A", dest="bond_cutoff_A", type=float, default=2.2)
    lab = sub.add_parser("label", help="Check teacher and label fixed geometries; resume verified PASS files")
    lab.add_argument("--run", type=Path, required=True)
    lab.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    lab.add_argument("--world", type=int, default=int(os.environ.get("SLURM_NTASKS", "1")))
    lab.add_argument("--threads", type=int, default=16)
    lab.add_argument("--limit", type=int, help="At most this many new cases per worker, for a first check")
    report = sub.add_parser("summary", help="Verify and summarize completed/pending/failed cases")
    report.add_argument("--run", type=Path, required=True)
    recovery = sub.add_parser("recover-parser-holds", help="Recheck held Oishi extended names; preserve labels")
    recovery.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare" and not (0 < args.overlap_A < args.bond_cutoff_A):
        p.error("Need 0 < overlap-A < bond-cutoff-A")
    if args.command == "label" and (args.threads <= 0 or (args.limit is not None and args.limit <= 0)):
        p.error("threads and limit must be positive")
    return {"prepare": prepare, "label": label, "summary": summary,
            "recover-parser-holds": recover_parser_holds}[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
