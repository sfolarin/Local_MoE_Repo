#!/usr/bin/env python3
"""Consolidate Local-MoE labels and assign phase-pair grouped splits.

CPU only: NumPy plus prepare_expanded_moe_labels.py in this directory.
Inputs are read-only; --out must be NEW. Old 196/22/22 assignments are preserved.

HOLDOUT VARIANT: the split of every other group is read from --split-table (a CSV
with record_id, new_split, geometry_sha256) instead of being assigned by hash order.
The table must cover exactly the records of the label run, every geometry hash must
match, every related group (phase pair / lineage / exact geometry) must sit in ONE
split, and Own_augmented must keep its original split. Any violation aborts before
anything is written. No label values or model errors are used by this script.
Oishi variants, cross-source pairs, and exact duplicate geometries stay together.
Every duplicate alias is recorded.
Output NPZs contain geometry, E/F, and any EXISTING region arrays. No SOAP
features, routing targets, fitted normalization, gamma calculation, or training.
"""
# --- added when the repository was assembled: no machine-specific paths ---------------------------------------
# MOE_HET (environment variable) = the data workspace folder that holds Local_MoE_MATPES_R2SCAN_Expanded_v1/ and
# External_Benchmark_Oishi2025/; REPO_ROOT = this repository.
import os as _os
import sys as _sys
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]
_REPO_CODE_DIRS = [str(REPO_ROOT / _d) for _d in ['src/local_moe', 'training', 'data_splits', 'evaluation', 'interface_energy', 'cost', 'md', 'deploy', 'tests', 'figures']]
for _p in _REPO_CODE_DIRS:
    if _p not in _sys.path:
        _sys.path.append(_p)
# -------------------------------------------------------------------------------------------------------------------

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import prepare_expanded_moe_labels as io

SPLITS = ("train", "val", "test")
EXPECTED_OWN = {"train": 196, "val": 22, "test": 22}
EXPECTED_EXTERNAL = {"Oishi_all179": 179, "Oishi_relaxed52": 52, "Siddharth": 41}


def load_inputs(run):
    cfg = io.load_config(run)
    contract = json.loads((run / "teacher_contract.json").read_text())
    if contract["teacher"] != io.TEACHER or contract["dtype"] != "float64":
        raise ValueError("Unexpected external teacher contract")
    own_rows = io.read_csv(run / "own_records.csv")
    external_rows = io.read_csv(run / "external_records.csv")
    if dict(Counter(r["existing_split"] for r in own_rows)) != EXPECTED_OWN:
        raise ValueError("Original split inventory changed; expected 196/22/22")
    if dict(Counter(r["source"] for r in external_rows)) != EXPECTED_EXTERNAL:
        raise ValueError("External inventory changed; expected Oishi 179+52 and Siddharth 41")
    if any(r["qc_status"] != "PASS" for r in external_rows):
        raise ValueError("Resolve remaining external QC holds before freezing this dataset")
    queue = {q["geometry_sha256"]: q for q in io.read_csv(run / "label_queue.csv")}
    for q in queue.values():
        good, state = io.completed(run, q, contract)
        if not good:
            raise ValueError("External label did not verify: {} {}".format(q["first_record_id"], state))

    baseline_path = run / "provenance/own_augmented_manifest.csv"
    provenance = io.read_csv(run / "provenance/snapshot_provenance.csv")
    entry = [r for r in provenance if Path(r["snapshot_path"]).name == baseline_path.name]
    if len(entry) != 1 or io.sha256(baseline_path) != entry[0]["sha256"]:
        raise ValueError("The original augmented-manifest snapshot did not verify")
    old_root = Path(entry[0]["original_path"]).parent
    baseline = io.read_csv(baseline_path)
    if len(baseline) != len(own_rows):
        raise ValueError("Original and prepared own-data records disagree")

    records, arrays, old_label_provenance = [], {}, []
    for index, (row, old) in enumerate(zip(own_rows, baseline)):
        feature = (old_root / old["output_file"]).resolve()
        label = Path(old["source_shard"]).expanduser().resolve()
        if (row["record_id"] != "own_{:04d}".format(index)
                or row["record"] != old["source_file"]
                or row["family"] != old["phase_pair"]
                or row["existing_split"] != old["split"]
                or int(row["n_atoms"]) != int(old["n_atoms"])
                or Path(row["feature_file"]).resolve() != feature
                or Path(row["source_shard"]).resolve() != label):
            raise ValueError("Own-data metadata changed: " + row["record_id"])
        g = io.from_npz(feature, legacy_periodic=True)
        io.basic_qc(g, int(row["n_atoms"]), cfg["size_limit_atoms"])
        if io.geometry_hash(g) != row["geometry_sha256"]:
            raise ValueError("Own-data geometry changed: " + row["record_id"])
        with np.load(label, allow_pickle=False) as d:
            energy = float(np.asarray(d["mace_energy_total_eV"]).item())
            forces = np.asarray(d["mace_forces"], dtype=np.float64).copy()
        io.check_labels(energy, forces, len(g["atomic_numbers"]))
        extras = {}
        with np.load(feature, allow_pickle=False) as d:
            for name in ("phase", "interface_center"):
                if name in d:
                    value = np.asarray(d[name]).reshape(-1).copy()
                    if value.shape != (len(forces),):
                        raise ValueError("Unexpected existing region-array shape: " + str(feature))
                    extras[name] = value
        digest = io.sha256(label)
        old_label_provenance.append(dict(record_id=row["record_id"], source_label_file=str(label),
                                         source_label_sha256=digest, feature_file=str(feature),
                                         geometry_sha256=row["geometry_sha256"]))
        records.append(dict(row, lineage_group="Own:" + row["family"], source_label_file=str(label),
                            source_label_sha256=digest, teacher_state_sha256="",
                            label_provenance="existing_MATPes_r2SCAN_labels_two_training_anchors_verified"))
        arrays[row["record_id"]] = dict(g, mace_energy_total_eV=energy, mace_forces=forces, **extras)

    for row in external_rows:
        gh = row["geometry_sha256"]
        if gh not in queue:
            raise ValueError("External record absent from verified label queue: " + row["record_id"])
        label = run / "labels" / (gh + ".npz")
        with np.load(label, allow_pickle=False) as d:
            g = io.geometry(d)
            energy = float(d["mace_energy_total_eV"].item())
            forces = np.asarray(d["mace_forces"], dtype=np.float64).copy()
        io.basic_qc(g, int(row["n_atoms"]), cfg["size_limit_atoms"])
        io.check_labels(energy, forces, len(g["atomic_numbers"]))
        records.append(dict(row, source_label_file=str(label), source_label_sha256=io.sha256(label),
                            teacher_state_sha256=contract["teacher_state_sha256"],
                            label_provenance="new_MATPes_r2SCAN_verified_label"))
        arrays[row["record_id"]] = dict(g, mace_energy_total_eV=energy, mace_forces=forces)

    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Record IDs must be unique")
    for row in records:
        if io.pair_key(row["source"], row["family"]) != row["phase_pair_group"]:
            raise ValueError("Phase-pair metadata does not match the source name: " + row["record_id"])
    return cfg, contract, records, arrays, old_label_provenance


def load_split_table(path, records):
    """record_id -> split from a CSV; must match the label run's records exactly."""
    rows = io.read_csv(path)
    if not rows or not {"record_id", "new_split", "geometry_sha256"} <= set(rows[0]):
        raise ValueError("Split table needs columns record_id, new_split, geometry_sha256: " + str(path))
    table, hashes = {}, {}
    for r in rows:
        rid = r["record_id"]
        if rid in table:
            raise ValueError("Duplicate record_id in the split table: " + rid)
        if r["new_split"] not in SPLITS:
            raise ValueError("Unknown split {!r} for {}".format(r["new_split"], rid))
        table[rid], hashes[rid] = r["new_split"], r["geometry_sha256"]
    ids = {r["record_id"] for r in records}
    if set(table) != ids:
        raise ValueError("Split table and label run disagree: {} records missing from the table, {} extra rows".format(
            len(ids - set(table)), len(set(table) - ids)))
    for r in records:
        if hashes[r["record_id"]] != r["geometry_sha256"]:
            raise ValueError("Geometry hash in the split table differs from the record: " + r["record_id"])
    return table


def grouped_assignments(records, seed, val_fraction, test_fraction, table=None):
    """Join related records, preserve old assignments, partition new groups.

    With a split table the new groups take the table's split (all records of a related group
    must agree) instead of the hash-ordered 80/10/10 assignment."""
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for key in ("phase_pair_group", "lineage_group", "geometry_sha256"):
        seen = {}
        for i, row in enumerate(records):
            value = row.get(key)
            if not value:
                continue
            if value in seen:
                join(i, seen[value])
            else:
                seen[value] = i
    components = defaultdict(list)
    for i, row in enumerate(records):
        components[find(i)].append(row)
    groups = []
    for rows in components.values():
        pairs = sorted({r["phase_pair_group"] for r in rows})
        key = "+".join(pairs)
        fixed = {r["existing_split"] for r in rows if r["source"] == "Own_augmented"}
        if len(fixed) > 1:
            raise ValueError("Existing splits conflict within related group {}: {}".format(key, sorted(fixed)))
        if fixed and not fixed.issubset(SPLITS):
            raise ValueError("Unknown original split: " + str(fixed))
        groups.append(dict(group_id=key, phase_pairs=";".join(pairs),
                           split=next(iter(fixed)) if fixed else "", anchored=bool(fixed),
                           record_ids=[r["record_id"] for r in rows], records=rows))
    if table is not None:
        for group in groups:
            wanted = {table[rid] for rid in group["record_ids"]}
            if len(wanted) != 1:
                raise ValueError("The split table puts related group {} in more than one split: {}".format(
                    group["group_id"], sorted(wanted)))
            split = next(iter(wanted))
            if group["anchored"] and split != group["split"]:
                raise ValueError("The split table would change an original Own split in group {}: {} -> {}".format(
                    group["group_id"], group["split"], split))
            group["split"] = split
        mapping = {rid: (g["split"], g["group_id"]) for g in groups for rid in g["record_ids"]}
        return mapping, sorted(groups, key=lambda g: g["group_id"])
    free = sorted([g for g in groups if not g["anchored"]], key=lambda g: (
        hashlib.sha256((str(seed) + "|" + g["group_id"]).encode()).hexdigest(), g["group_id"]))
    if len(free) < 3:
        raise ValueError("Need at least three new independent groups for train/val/test")
    n_val = max(1, int(len(free) * val_fraction + 0.5))
    n_test = max(1, int(len(free) * test_fraction + 0.5))
    if n_val + n_test >= len(free):
        raise ValueError("Split fractions leave no new training groups")
    for index, group in enumerate(free):
        group["split"] = "test" if index < n_test else ("val" if index < n_test + n_val else "train")
    mapping = {rid: (g["split"], g["group_id"]) for g in groups for rid in g["record_ids"]}
    return mapping, sorted(groups, key=lambda g: g["group_id"])


def select_unique(records, arrays, cfg):
    by_geometry = defaultdict(list)
    for row in records:
        by_geometry[row["geometry_sha256"]].append(row)
    representatives, aliases = [], []
    for gh, rows in sorted(by_geometry.items()):
        rows.sort(key=lambda r: (r["source"] != "Own_augmented", r["record_id"]))
        first = rows[0]
        representatives.append(first)
        ref = arrays[first["record_id"]]
        n = len(ref["atomic_numbers"])
        for row in rows:
            other = arrays[row["record_id"]]
            de = abs(float(other["mace_energy_total_eV"]) - float(ref["mace_energy_total_eV"])) / n
            df = float(np.sqrt(np.mean((other["mace_forces"] - ref["mace_forces"]) ** 2)))
            if de > cfg["anchor_energy_tolerance_eV_per_atom"] or df > cfg["anchor_force_rmse_tolerance_eV_A"]:
                raise ValueError("Conflicting labels for identical geometry: {} {}".format(first["record_id"], row["record_id"]))
            aliases.append(dict(row, representative_id=first["record_id"],
                                is_representative=row["record_id"] == first["record_id"],
                                duplicate_energy_delta_eV_per_atom=de, duplicate_force_rmse_eV_A=df))
    return representatives, aliases


def prepare(args):
    run, out = args.label_run.expanduser().resolve(), args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError("Use a new output directory; existing split files remain frozen: " + str(out))
    cfg, contract, records, arrays, old_provenance = load_inputs(run)
    table = load_split_table(args.split_table.expanduser().resolve(), records) if args.split_table else None
    mapping, groups = grouped_assignments(records, args.seed, args.val_fraction, args.test_fraction, table)
    representatives, aliases = select_unique(records, arrays, cfg)
    if table is not None:
        wrong = [rid for rid, (split, _) in mapping.items() if split != table[rid]]
        if wrong:
            raise AssertionError("Assigned splits differ from the split table for {} records".format(len(wrong)))
    for row in aliases:
        row["split"], row["split_group"] = mapping[row["record_id"]]
        if row["source"] == "Own_augmented" and row["split"] != row["existing_split"]:
            raise AssertionError("Original split changed")
    alias_by_id = {row["record_id"]: row for row in aliases}
    for key in ("phase_pair_group", "lineage_group", "geometry_sha256"):
        split_sets = defaultdict(set)
        for row in aliases:
            if row.get(key):
                split_sets[row[key]].add(row["split"])
        if any(len(values) > 1 for values in split_sets.values()):
            raise AssertionError("Cross-split leakage detected by " + key)

    out.mkdir(parents=True, exist_ok=False)
    for folder in ("provenance", "shards/train", "shards/val", "shards/test"):
        (out / folder).mkdir(parents=True, exist_ok=False)
    input_hashes = {}
    if table is not None:
        input_hashes["holdout_split_table.csv"] = io.snapshot(args.split_table.expanduser().resolve(),
                                                              out / "provenance" / "holdout_split_table.csv")
    for name in ("preparation.json", "external_records.csv", "own_records.csv", "label_queue.csv",
                 "teacher_contract.json", "anchors/manifest.csv", "provenance/own_augmented_manifest.csv",
                 "provenance/snapshot_provenance.csv", "provenance/size_deferred.csv"):
        source = run / name
        input_hashes[name] = io.snapshot(source, out / "provenance" / name.replace("/", "__"))
    for script in (Path(__file__).resolve(), Path(io.__file__).resolve()):
        io.snapshot(script, out / "provenance" / script.name)

    manifest = []
    for record in sorted(representatives, key=lambda r: (SPLITS.index(mapping[r["record_id"]][0]), r["record_id"])):
        rid = record["record_id"]
        row = alias_by_id[rid]
        data = arrays[rid]
        relative = "shards/{}/{}.npz".format(row["split"], rid)
        target = out / relative
        io.write_npz(target, dict(data, record_id=rid, split=row["split"],
                                  geometry_sha256=row["geometry_sha256"], teacher=io.TEACHER,
                                  teacher_state_sha256=row["teacher_state_sha256"],
                                  label_provenance=row["label_provenance"],
                                  source_label_sha256=row["source_label_sha256"]))
        energy_per_atom = float(data["mace_energy_total_eV"]) / len(data["atomic_numbers"])
        max_force = float(np.linalg.norm(data["mace_forces"], axis=1).max())
        manifest.append(dict(record_id=rid, source=row["source"], source_record=row["record"],
                              family=row["family"], phase_pair_group=row["phase_pair_group"],
                              split_group=row["split_group"], split=row["split"],
                              existing_split=row["existing_split"], n_atoms=int(row["n_atoms"]),
                              geometry_sha256=row["geometry_sha256"], data_file=relative,
                              data_sha256=io.sha256(target), source_label_file=row["source_label_file"],
                              source_label_sha256=row["source_label_sha256"],
                              label_provenance=row["label_provenance"],
                              has_region_metadata="phase" in data and "interface_center" in data,
                              energy_eV_per_atom=energy_per_atom, max_force_eV_A=max_force))
    io.write_csv(out / "manifest.csv", manifest)
    for split in SPLITS:
        io.write_csv(out / (split + ".csv"), [r for r in manifest if r["split"] == split], list(manifest[0]))
    io.write_csv(out / "all_source_records.csv", aliases)
    io.write_csv(out / "old_label_provenance.csv", old_provenance)
    io.write_csv(out / "group_assignments.csv", [
        dict(group_id=g["group_id"], phase_pairs=g["phase_pairs"], split=g["split"],
             reason="preserve_existing_assignment" if g["anchored"] else "fixed_seed_new_group_assignment",
             source_record_count=len(g["records"]), unique_geometries=len({r["geometry_sha256"] for r in g["records"]}),
             counts_by_source=json.dumps(dict(Counter(r["source"] for r in g["records"])), sort_keys=True),
             record_ids=";".join(g["record_ids"])) for g in groups])
    io.write_csv(out / "label_range_review.csv", sorted(manifest, key=lambda r: -r["max_force_eV_A"]))

    counts = Counter(r["split"] for r in manifest)
    if table is not None:
        expected = Counter(table[r["record_id"]] for r in representatives)
        if dict(expected) != dict(counts):
            raise AssertionError("Manifest split counts {} differ from the table {}".format(dict(counts), dict(expected)))
    sources = sorted({r["source"] for r in aliases})
    by_source = {s: {split: sum(r["source"] == s and r["split"] == split for r in aliases)
                     for split in SPLITS} for s in sources}
    notes = [
        "Counts are configuration records, not independent interfaces or phase pairs.",
        "Old assignments are preserved; " + ("all other groups follow the supplied split table "
        "(holdout_split_table.csv in provenance)." if table is not None else
        "new phase-pair groups target 80/10/10 by default."),
        "No errors, energies, forces, or gamma values were used to choose the split.",
        "Future data should inherit these saved group assignments; do not reshuffle existing groups.",
        "Exact ordered hashes do not certify symmetry/permutation equivalence.",
        "Legacy targets were checked for finite E/F; their exact model-state hash is unavailable.",
        "The labeling stage checked the teacher against two old training anchors, not every old target.",
        "SOAP normalization and any learned routing targets must be fitted on training data only.",
        "External region arrays are absent; they were not inferred or fabricated.",
        "These AB labels do not create new interface-energy gamma targets.",
        "Original training scripts need a loader/count update before using this dataset."
    ]
    absent = [s + ":" + split for s in sources for split in SPLITS if by_source[s][split] == 0]
    result = dict(created=io.now(), label_run=str(run), output=str(out), seed=args.seed,
                  new_group_fraction_targets=dict(train=1 - args.val_fraction - args.test_fraction,
                                                  val=args.val_fraction, test=args.test_fraction),
                  total_source_records=len(aliases), unique_geometries=len(manifest),
                  duplicate_aliases=len(aliases) - len(manifest), unique_configuration_counts=dict(counts),
                  source_record_counts=by_source, original_counts=EXPECTED_OWN,
                  group_counts=dict(Counter(g["split"] for g in groups)),
                  new_group_counts=dict(Counter(g["split"] for g in groups if not g["anchored"])),
                  phase_pair_or_lineage_or_exact_geometry_cross_split_conflicts=0,
                  empty_source_split_cells=absent, input_hashes=input_hashes,
                  manifest_sha256=io.sha256(out / "manifest.csv"),
                  features_generated=False, normalization_fitted=False, training_started=False, notes=notes,
                  split_source="table" if table is not None else "hash_order")
    io.write_json(out / "split_summary.json", result)
    print("Expanded dataset: " + str(out))
    print("Original assignments preserved: train=196, val=22, test=22")
    print("Source records: {}; unique geometries: {}; duplicate aliases: {}".format(
        len(aliases), len(manifest), len(aliases) - len(manifest)))
    print("{:<22s} {:>7s} {:>7s} {:>7s}".format("Source records", "train", "val", "test"))
    for source in sources:
        print("{:<22s} {:>7d} {:>7d} {:>7d}".format(source, *(by_source[source][s] for s in SPLITS)))
    print("Unique configuration totals: " + json.dumps(dict(counts), sort_keys=True))
    print("New independent group counts: " + json.dumps(result["new_group_counts"], sort_keys=True))
    print("Cross-split phase-pair/lineage/exact-geometry conflicts: 0")
    print("Siddharth family assignments:")
    families = defaultdict(list)
    for row in aliases:
        if row["source"] == "Siddharth":
            families[row["family"]].append(row)
    for family, rows in sorted(families.items()):
        print("  {}: {} records -> {}".format(family, len(rows), rows[0]["split"]))
    if absent:
        print("Source/split cells without records: " + ", ".join(absent))
        print("Report source-specific held-out metrics only where held-out data exist.")
    print("Geometry/label shards copied. SOAP features, routing targets, and training remain next.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--split-table", type=Path, default=None,
                        help="CSV with record_id,new_split,geometry_sha256: use these splits instead of hash order "
                             "(the fractions are then ignored)")
    args = parser.parse_args()
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1
            and args.val_fraction + args.test_fraction < 1):
        parser.error("Use positive validation/test fractions whose sum is less than one")
    return prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
