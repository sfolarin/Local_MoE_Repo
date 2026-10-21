#!/usr/bin/env python3
"""Cache raw local TorchSOAP features and summarize training-only moments.

Uses the saved C1b live_raw_features implementation (float64 geometry/TorchSOAP,
float32 392-component local features). No model is trained. These caches support
normalization/routing preparation; conservative force training must recompute
features from live coordinates and retain the routing coordinate gradients.

compute uses one Slurm task per GPU. finalize is CPU-only. A feature contract
binds outputs to the frozen dataset and source snapshots. Resume reuses only
verified PASS files. Existing checkpoints and their normalization are untouched.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import prepare_expanded_moe_labels as io

TRAINER = "train_c1b_frozen_routing_conservative_residual_v3_196train.py"
COMMON = "train_conservative_common_energy_baseline_augmented.py"
SOAP = "torchsoap_autograd_module_periodic_regnorm.py"
SPLITS = ("train", "val", "test")


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs(args):
    dataset = args.dataset.expanduser().resolve()
    snapshot = args.snapshot.expanduser().resolve()
    manifest = dataset / "manifest.csv"
    split_summary = json.loads((dataset / "split_summary.json").read_text())
    if io.sha256(manifest) != split_summary["manifest_sha256"]:
        raise ValueError("Dataset manifest changed after its split was frozen")
    rows = io.read_csv(manifest)
    counts = dict(Counter(r["split"] for r in rows))
    if counts != split_summary["unique_configuration_counts"]:
        raise ValueError("Dataset split counts disagree with its summary")
    source_index = {str(Path(r["snapshot_path"]).resolve()): r
                    for r in io.read_csv(snapshot / "snapshot_provenance.csv")}
    paths = {name: snapshot / "src_snapshot" / name for name in (TRAINER, COMMON, SOAP)}
    hashes = {}
    for name, path in paths.items():
        entry = source_index.get(str(path))
        if entry is None or io.sha256(path) != entry["sha256"]:
            raise ValueError("Source snapshot did not verify: " + str(path))
        hashes[name] = entry["sha256"]
    alias_file = dataset / "all_source_records.csv"
    aliases = io.read_csv(alias_file)
    original_features = {r["representative_id"]: Path(r["feature_file"])
                         for r in aliases if r["source"] == "Own_augmented"
                         and r["record_id"] == r["representative_id"]}
    contract = dict(dataset=str(dataset), dataset_manifest_sha256=io.sha256(manifest),
                    source_snapshot=str(snapshot), source_sha256=hashes,
                    alias_manifest_sha256=io.sha256(alias_file),
                    feature_dim=392, soap_dim=390, species_order=["Si", "O"],
                    geometry_dtype="float64", feature_dtype="float32",
                    r_cut_A=5.0, n_max=6, l_max=4,
                    baseline_feature_check_atol=1e-5, baseline_feature_check_rtol=1e-5,
                    counts=counts, feature_preparation_script_sha256=io.sha256(Path(__file__)))
    return dataset, rows, paths, original_features, contract


def cache_paths(out, row):
    rid = row["record_id"]
    if Path(rid).name != rid or rid in (".", "..", ""):
        raise ValueError("Unsafe record ID")
    return out / "features" / (rid + ".npz"), out / "status" / (rid + ".json")


def valid_cache(out, row, contract_hash):
    path, marker = cache_paths(out, row)
    if not marker.exists():
        return False, "PENDING"
    try:
        status = json.loads(marker.read_text())
        if status["status"] != "PASS":
            return False, status["status"]
        if status["contract_sha256"] != contract_hash or io.sha256(path) != status["feature_sha256"]:
            return False, "INVALID_PROVENANCE"
        with np.load(path, allow_pickle=False) as d:
            x = d["X_local_torchsoap"]
            if x.dtype != np.float32 or x.shape != (int(row["n_atoms"]), 392) or not np.isfinite(x).all():
                return False, "INVALID_FEATURES"
            if d["geometry_sha256"].item() != row["geometry_sha256"] or d["split"].item() != row["split"]:
                return False, "INVALID_GEOMETRY_OR_SPLIT"
        return True, "PASS"
    except Exception as exc:
        return False, "INVALID_CACHE:" + type(exc).__name__


def copy_atomic(source, target, expected):
    # Multiple workers may reach this at once; readers see only complete files.
    if not target.exists():
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("Source changed during snapshot")
        io.atomic_write(target, lambda p: p.write_bytes(payload))
    if io.sha256(target) != expected:
        raise ValueError("Feature-stage source snapshot mismatch: " + str(target))


def compute(args):
    import gc
    import torch
    dataset, rows, paths, original_features, contract = inputs(args)
    out = args.out.expanduser().resolve()
    if not 0 <= args.rank < args.world:
        raise ValueError("Invalid worker rank/world")
    if not torch.cuda.is_available():
        raise RuntimeError("Feature computation requires a GPU allocation")
    if args.world > 1 and torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see one GPU; use --gpus-per-task=1 --gpu-bind=single:1")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    contract["torch_version"] = str(torch.__version__)
    contract["cuda_version"] = str(torch.version.cuda)
    for folder in ("features", "status", "logs", "src_snapshot"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    io.publish_contract(out / "feature_contract.json", contract)
    contract_hash = io.sha256(out / "feature_contract.json")
    for name, path in paths.items():
        copy_atomic(path, out / "src_snapshot" / name, contract["source_sha256"][name])
    trainer = import_file(out / "src_snapshot" / TRAINER, "expanded_moe_feature_trainer")
    soap = trainer.import_torchsoap(out / "src_snapshot" / SOAP)
    descriptor = soap.TorchSOAPDescriptor(soap.TorchSOAPConfig(
        species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    descriptor.eval()
    for parameter in descriptor.parameters():
        parameter.requires_grad_(False)
    processed = skipped = failures = 0
    for row in rows[args.rank::args.world]:
        good, _ = valid_cache(out, row, contract_hash)
        if good:
            skipped += 1
            continue
        target, marker = cache_paths(out, row)
        status = dict(record_id=row["record_id"], split=row["split"], n_atoms=int(row["n_atoms"]),
                      rank=args.rank, contract_sha256=contract_hash, started=io.now())
        start = time.perf_counter()
        positions = cell = x_tensor = None
        try:
            source = dataset / row["data_file"]
            if io.sha256(source) != row["data_sha256"]:
                raise ValueError("Dataset shard changed: " + str(source))
            g = io.from_npz(source)
            if io.geometry_hash(g) != row["geometry_sha256"]:
                raise ValueError("Dataset geometry hash mismatch")
            symbols, masses = trainer.build_symbols_masses(g["atomic_numbers"])
            record = SimpleNamespace(source_file=row["record_id"], n_atoms=len(symbols),
                                     atomic_numbers=g["atomic_numbers"], symbols=symbols, masses=masses)
            positions = torch.tensor(g["positions"], dtype=torch.float64, device=device)
            cell = torch.tensor(g["cell"], dtype=torch.float64, device=device)
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                x_tensor = trainer.live_raw_features(record, positions, cell, descriptor)
                x = x_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
            if x.shape != (record.n_atoms, 392) or not np.isfinite(x).all():
                raise ValueError("Invalid raw feature shape or nonfinite values")
            expected_species = np.stack([g["atomic_numbers"] == 14, g["atomic_numbers"] == 8], axis=1)
            if not np.array_equal(x[:, 390:], expected_species.astype(np.float32)):
                raise ValueError("Species feature order differs from [Si, O]")
            if row["record_id"] in original_features:
                with np.load(original_features[row["record_id"]], allow_pickle=False) as d:
                    old_x = np.asarray(d["X_local_torchsoap"], dtype=np.float32)
                if old_x.shape != x.shape or not np.isfinite(old_x).all():
                    raise ValueError("Existing B1 features have invalid shape/values")
                status["old_B1_max_abs_feature_difference"] = float(np.abs(x - old_x).max())
                if not np.allclose(x, old_x, atol=1e-5, rtol=1e-5):
                    raise ValueError("Live features disagree with existing B1 features; investigate descriptor provenance")
            io.write_npz(target, dict(X_local_torchsoap=x, record_id=row["record_id"],
                                      split=row["split"], geometry_sha256=row["geometry_sha256"]))
            status.update(status="PASS", feature_file=str(target), feature_sha256=io.sha256(target),
                          peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024**3)
        except Exception as exc:
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            status.update(status="CUDA_OOM" if oom else "ERROR", error=str(exc), traceback=traceback.format_exc())
            failures += 1
        status.update(finished=io.now(), elapsed_s=time.perf_counter() - start)
        io.write_json(marker, status)
        processed += 1
        print("rank={} {} {} N={} {:.2f}s".format(args.rank, row["record_id"], status["status"],
                                                 row["n_atoms"], status["elapsed_s"]), flush=True)
        del positions, cell, x_tensor
        gc.collect()
        torch.cuda.empty_cache()
    result = dict(rank=args.rank, processed=processed, reused=skipped, failures=failures, finished=io.now())
    io.write_json(out / "logs" / ("worker_{}_{}.json".format(args.rank, time.time_ns())), result)
    print(json.dumps(result), flush=True)
    return 2 if failures else 0


def update_moments(count, mean, m2, x):
    """Float64 parallel-Welford update; each training atom has equal weight."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    block_mean = x.mean(axis=0)
    block_m2 = np.sum((x - block_mean) ** 2, axis=0)
    if count == 0:
        return n, block_mean, block_m2
    delta = block_mean - mean
    total = count + n
    return total, mean + delta * (n / total), m2 + block_m2 + delta**2 * (count * n / total)


def training_context(out, paths):
    """Small source excerpt for adapting the existing common-energy trainer."""
    text = paths[COMMON].read_text()
    lines = text.splitlines()
    tree = ast.parse(text)
    sections = ["COMMON-ENERGY TRAINER SNAPSHOT", str(paths[COMMON]), "",
                "TOP-LEVEL FUNCTIONS/CLASSES"]
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    for n in nodes:
        signature = ast.unparse(n.args) if isinstance(n, ast.FunctionDef) else "class"
        sections.append("{}:{} {}({})".format(n.lineno, n.end_lineno, n.name, signature))
    selected = [n for n in nodes if n.name in {"parse_args", "MLP", "StructureRecord"}
                or any(word in n.name.lower() for word in ("normal", "stat", "reference", "e_ref"))]
    for n in selected:
        sections += ["", "SOURCE: " + n.name]
        sections += lines[n.lineno - 1:min(n.end_lineno, n.lineno + 159)]
        if n.end_lineno > n.lineno + 159:
            sections.append("[remaining lines omitted]")
    # Capture reference/normalization settings in main without dumping its loop.
    indices = set()
    for i, line in enumerate(lines):
        if any(word in line for word in ("e_ref =", "feature_mean =", "feature_std =", "std_floor", "min_std")):
            indices.update(range(max(0, i - 2), min(len(lines), i + 8)))
    sections += ["", "REFERENCE/NORMALIZATION CALL SITES"]
    sections += ["{}: {}".format(i + 1, lines[i]) for i in sorted(indices)]
    provenance = io.read_csv(paths[COMMON].parent.parent / "snapshot_provenance.csv")
    common_entry = [r for r in provenance if Path(r["snapshot_path"]).resolve() == paths[COMMON]]
    if len(common_entry) == 1:
        het = Path(common_entry[0]["original_path"]).parent
        sections += ["", "ROUTING/TEACHER/EXPERT SCRIPT CANDIDATES (filenames only)", str(het)]
        candidates = sorted(p for p in het.glob("*.py")
                            if any(word in p.name.lower() for word in ("routing", "teacher", "local_expert", "gate")))
        sections += [p.name for p in candidates[:80]]
        if len(candidates) > 80:
            sections.append("[additional filenames omitted]")
    io.atomic_write(out / "training_adapter_context.txt", lambda p: p.write_text("\n".join(sections) + "\n"))


def finalize(args):
    dataset, rows, paths, _, expected = inputs(args)
    out = args.out.expanduser().resolve()
    contract_path = out / "feature_contract.json"
    saved = json.loads(contract_path.read_text())
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError("Feature contract does not match the requested dataset/source: " + key)
    contract_hash = io.sha256(contract_path)
    statuses, manifest = [], []
    for row in rows:
        good, state = valid_cache(out, row, contract_hash)
        path, marker = cache_paths(out, row)
        statuses.append(dict(record_id=row["record_id"], source=row["source"], split=row["split"], status=state))
        if good:
            manifest.append(dict(row, feature_file=str(path), feature_sha256=io.sha256(path)))
    io.write_csv(out / "feature_status.csv", statuses)
    counts = Counter(r["status"] for r in statuses)
    print("Feature status: " + json.dumps(dict(counts), sort_keys=True))
    if any(r["status"] != "PASS" for r in statuses):
        io.write_json(out / "feature_summary.json",
                      dict(created=io.now(), finalization_status="INCOMPLETE",
                           feature_status_counts=dict(counts), training_moments_ready=False))
        print("Resolve failed/pending features before finalization: " + str(out / "feature_status.csv"))
        return 2
    count = 0
    mean, m2 = np.zeros(392), np.zeros(392)
    train_ids = []
    for row in manifest:
        if row["split"] != "train":
            continue
        with np.load(row["feature_file"], allow_pickle=False) as d:
            count, mean, m2 = update_moments(count, mean, m2, d["X_local_torchsoap"])
        train_ids.append(row["record_id"])
    if not count:
        raise ValueError("No training atoms")
    variance = np.maximum(m2 / count, 0.0)
    # Raw moments, NOT an overwritten checkpoint normalization. The trainer's
    # established std-floor convention will be applied when building that model.
    stats = out / "training_feature_moments.npz"
    io.write_npz(stats, dict(mean=mean, population_variance=variance,
                             population_std=np.sqrt(variance), n_training_atoms=count,
                             n_training_structures=len(train_ids), train_record_ids=np.array(train_ids),
                             weighting="equal_per_training_atom", std_floor_applied=False))
    io.write_csv(out / "manifest.csv", manifest)
    training_context(out, paths)
    summary = dict(created=io.now(), finalization_status="PASS", training_moments_ready=True,
                   dataset=str(dataset), source_snapshot=str(args.snapshot.resolve()),
                   contract_sha256=contract_hash, feature_counts=dict(Counter(r["split"] for r in manifest)),
                   training_structures=len(train_ids), training_atoms=count,
                   moments_weighting="equal_per_training_atom", feature_dim=392,
                   validation_test_used_for_moments=False, checkpoint_normalization_modified=False,
                   std_floor_applied=False, routing_targets_generated=False, training_started=False,
                   training_moments_sha256=io.sha256(stats),
                   manifest_sha256=io.sha256(out / "manifest.csv"))
    io.write_json(out / "feature_summary.json", summary)
    print("Features by split: " + json.dumps(summary["feature_counts"], sort_keys=True))
    print("Training-only moments: {} structures, {} atoms".format(len(train_ids), count))
    print("Raw features: " + str(out / "manifest.csv"))
    print("Training moments: " + str(stats))
    print("Trainer excerpt to share: " + str(out / "training_adapter_context.txt"))
    print("No checkpoints changed or model training performed. Force training requires live features.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compute", "finalize"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    parser.add_argument("--world", type=int, default=int(os.environ.get("SLURM_NTASKS", "1")))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if args.threads < 1 or args.world < 1:
        parser.error("threads and world must be positive")
    return compute(args) if args.command == "compute" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
