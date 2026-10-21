#!/usr/bin/env python3
"""Train the expanded conservative common-energy model using the saved trainer.

Requires prepare_expanded_moe_labels.py beside this file and the existing
silica_mpi environment. Run with ONE Slurm task on ONE 80-GB GPU.

The original AtomicEnergyMLP, live descriptor, predict_energy_forces,
structure_loss, and compute_feature_normalization are used directly. The new
loader understands expanded shards and leaves absent region annotations as
None. The adapter supplies an epoch loop independent of the old split sizes.
No routing parameters/targets or gamma targets are trained here.

Features are regenerated from differentiable coordinates for every E/F step.
Cached raw features are used only for train-only normalization and a startup
consistency check. The old common checkpoint supplies ONLY the constant
e_ref; all common-network parameters start fresh.

Selection: 44 validation configurations; no test evaluation during training.
Loss: 10*(delta_E/N)^2 + 1000*mean_components(delta_F^2), with one complete
configuration per AdamW step. Defaults match the supplied trainer arguments.
This adapter explicitly uses ReduceLROnPlateau(factor=0.5, patience=10).

An automatic two-step real-target check on the largest TRAINING structure
uses a disposable model, checks force-loss backward and the C1b loader, then
re-seeds before production. It is not a completed model-training run.

latest_checkpoint.pt is atomic and resumable at completed-epoch boundaries.
An interrupted partial epoch is replayed from its preceding checkpoint.
best_checkpoint.pt is compatible with the saved C1b load_common_checkpoint.
Use 'evaluate' after training finishes to report metrics at that checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time
import traceback

import numpy as np
import prepare_expanded_moe_labels as io

COMMON = "train_conservative_common_energy_baseline_augmented.py"
C1B = "train_c1b_frozen_routing_conservative_residual_v3_196train.py"
SOAP = "torchsoap_autograd_module_periodic_regnorm.py"
SPLITS = ("train", "val", "test")
SETTING_NAMES = (
    "hidden_dims", "energy_weight", "force_weight", "lr", "weight_decay",
    "grad_clip", "patience", "scheduler_patience", "min_delta", "seed", "threads",
)


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checked_hash(path, expected):
    actual = io.sha256(path)
    if actual != expected:
        raise ValueError("Changed or mismatched input: " + str(path))
    return actual


def checked_index(rows, label):
    result = {r["record_id"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError("Repeated record IDs in " + label)
    return result


def verify_inputs(feature_root):
    """Verify manifests and grouping without using held-out labels."""
    feature_root = feature_root.expanduser().resolve()
    fc_path = feature_root / "feature_contract.json"
    fc = json.loads(fc_path.read_text())
    summary = json.loads((feature_root / "feature_summary.json").read_text())
    if summary["finalization_status"] != "PASS" or not summary["training_moments_ready"]:
        raise ValueError("Features have not passed finalization")
    checked_hash(fc_path, summary["contract_sha256"])
    fm = feature_root / "manifest.csv"
    checked_hash(fm, summary["manifest_sha256"])
    rows = io.read_csv(fm)
    by_id = checked_index(rows, "feature manifest")
    counts = dict(Counter(r["split"] for r in rows))
    if set(counts) != set(SPLITS) or counts != fc["counts"] or counts != summary["feature_counts"]:
        raise ValueError("Feature split counts disagree")
    if (fc["feature_dim"], fc["soap_dim"], fc["species_order"]) != (392, 390, ["Si", "O"]):
        raise ValueError("Unexpected descriptor dimensions/species order")
    dataset = Path(fc["dataset"]).resolve()
    dm = dataset / "manifest.csv"
    ds = json.loads((dataset / "split_summary.json").read_text())
    checked_hash(dm, fc["dataset_manifest_sha256"])
    checked_hash(dm, ds["manifest_sha256"])
    original = checked_index(io.read_csv(dm), "dataset manifest")
    if set(by_id) != set(original) or counts != ds["unique_configuration_counts"]:
        raise ValueError("Feature/dataset record membership differs")
    for rid, row in by_id.items():
        if any(row.get(k) != v for k, v in original[rid].items()):
            raise ValueError("Feature manifest changed dataset metadata: " + rid)
    for key in ("split_group", "phase_pair_group", "geometry_sha256"):
        groups = defaultdict(set)
        for row in rows:
            if row.get(key):
                groups[row[key]].add(row["split"])
        if any(len(s) != 1 for s in groups.values()):
            raise ValueError("Cross-split conflict: " + key)
    sources = {}
    for name in (COMMON, C1B, SOAP):
        path = feature_root / "src_snapshot" / name
        checked_hash(path, fc["source_sha256"][name])
        sources[name] = path
    prep = Path(fc["source_snapshot"]).resolve()
    provenance = {str(Path(r["snapshot_path"]).resolve()): r["sha256"]
                  for r in io.read_csv(prep / "snapshot_provenance.csv")}
    reference = prep / "baseline" / "common.pt"
    if str(reference) not in provenance:
        raise ValueError("Baseline common checkpoint has no snapshot provenance")
    reference_hash = checked_hash(reference, provenance[str(reference)])
    moments = feature_root / "training_feature_moments.npz"
    checked_hash(moments, summary["training_moments_sha256"])
    train_ids = [r["record_id"] for r in rows if r["split"] == "train"]
    train_atoms = sum(int(r["n_atoms"]) for r in rows if r["split"] == "train")
    with np.load(moments, allow_pickle=False) as d:
        saved_ids = d["train_record_ids"].tolist()
        if (len(saved_ids) != len(set(saved_ids)) or set(saved_ids) != set(train_ids)
                or int(d["n_training_atoms"]) != train_atoms
                or int(d["n_training_structures"]) != len(train_ids)):
            raise ValueError("Training moment membership/count mismatch")
        raw_mean = np.asarray(d["mean"], dtype=np.float64)
    if train_atoms != summary["training_atoms"] or len(train_ids) != summary["training_structures"]:
        raise ValueError("Feature summary training counts disagree")
    binding = dict(
        features=str(feature_root), dataset=str(dataset), counts=counts,
        feature_contract_sha256=io.sha256(fc_path), feature_manifest_sha256=io.sha256(fm),
        dataset_manifest_sha256=io.sha256(dm),
        training_moments_sha256=io.sha256(moments),
        source_sha256=fc["source_sha256"],
        reference_checkpoint=str(reference), reference_checkpoint_sha256=reference_hash,
        training_structures=len(train_ids), training_atoms=train_atoms,
    )
    return dict(rows=rows, dataset=dataset, sources=sources, reference=reference,
                raw_mean=raw_mean, binding=binding)


def load_records(inputs, trainer, splits):
    records = {s: [] for s in splits}
    for row in inputs["rows"]:
        if row["split"] not in records:
            continue
        rid, n = row["record_id"], int(row["n_atoms"])
        path = (inputs["dataset"] / row["data_file"]).resolve()
        checked_hash(path, row["data_sha256"])
        with np.load(path, allow_pickle=False) as d:
            g = io.geometry(d)
            io.basic_qc(g, n, 4608)
            if io.geometry_hash(g) != row["geometry_sha256"]:
                raise ValueError("Geometry mismatch: " + rid)
            if (d["record_id"].item(), d["split"].item(), d["teacher"].item()) != (
                    rid, row["split"], io.TEACHER):
                raise ValueError("Label identity/teacher mismatch: " + rid)
            energy = float(d["mace_energy_total_eV"])
            forces = np.asarray(d["mace_forces"], dtype=np.float64)
            io.check_labels(energy, forces, n)
            phase = np.asarray(d["phase"]).copy() if "phase" in d else None
            center = np.asarray(d["interface_center"]).copy() if "interface_center" in d else None
        # No region metrics or invented bulk/interface labels in this adapter.
        feature = Path(row["feature_file"])
        checked_hash(feature, row["feature_sha256"])
        with np.load(feature, allow_pickle=False) as d:
            x = d["X_local_torchsoap"]
            if x.shape != (n, 392) or x.dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError("Invalid features: " + rid)
            if (d["record_id"].item(), d["split"].item(), d["geometry_sha256"].item()) != (
                    rid, row["split"], row["geometry_sha256"]):
                raise ValueError("Feature identity mismatch: " + rid)
        symbols, masses = trainer.build_symbols_masses(g["atomic_numbers"])
        record = trainer.StructureRecord(
            source_file=rid, phase_pair=row["phase_pair_group"], split=row["split"],
            n_atoms=n, feature_file=feature, source_shard=path,
            atomic_numbers=g["atomic_numbers"], positions_np=g["positions"],
            cell_np=g["cell"], phase=phase, interface_center=center,
            target_energy_total_eV=energy, target_forces_np=forces, symbols=symbols, masses=masses)
        records[row["split"]].append(record)
    return records


def normalization(trainer, records, raw_mean):
    # Call the original function, including its exact floor and '<' comparison.
    mean, std = trainer.compute_feature_normalization(records)
    mean, std = np.asarray(mean), np.asarray(std)
    if (mean.shape != (392,) or std.shape != (392,) or mean.dtype != np.float32
            or std.dtype != np.float32 or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or not (std > 0).all()):
        raise ValueError("Invalid normalization from original trainer")
    if not np.allclose(mean, raw_mean, atol=1e-6, rtol=1e-5):
        raise ValueError("Original train-only mean disagrees with saved training moments")
    return mean, std


def vector_digest(mean, std):
    return hashlib.sha256(np.asarray(mean, dtype="<f4").tobytes()
                          + np.asarray(std, dtype="<f4").tobytes()).hexdigest()


def new_model(ctx, args):
    model = ctx.trainer.AtomicEnergyMLP(input_dim=392, hidden_dims=list(args.hidden_dims))
    return model.to(device=ctx.device, dtype=ctx.torch.float32)


def new_optimizer(ctx, model, args):
    return ctx.torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def prediction(ctx, model, record, create_graph):
    result = ctx.trainer.predict_energy_forces(
        model=model, descriptor=ctx.descriptor, record=record,
        feature_mean=ctx.mean, feature_std=ctx.std, e_ref=ctx.e_ref,
        device=ctx.device, create_graph=create_graph)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("Original predict_energy_forces must return (energy_total, forces)")
    energy, forces = result
    if energy.numel() != 1 or tuple(forces.shape) != (record.n_atoms, 3):
        raise ValueError("Unexpected prediction shapes: " + record.source_file)
    if energy.dtype != ctx.torch.float64 or forces.dtype != ctx.torch.float64:
        raise TypeError("Expected the original float64 total energy and forces")
    if not bool(ctx.torch.isfinite(energy).all()) or not bool(ctx.torch.isfinite(forces).all()):
        raise FloatingPointError("Nonfinite prediction: " + record.source_file)
    if create_graph and not forces.requires_grad:
        raise RuntimeError("Force training graph is detached")
    return energy, forces


def metric_row(energy, forces, record, args):
    predicted = float(energy.detach().cpu())
    df = forces.detach().cpu().numpy() - record.target_forces_np
    de = (predicted - record.target_energy_total_eV) / record.n_atoms
    fmse = float(np.mean(df**2))
    return dict(record_id=record.source_file, split=record.split, n_atoms=record.n_atoms,
                E_reference_eV=record.target_energy_total_eV, E_predicted_eV=predicted,
                dE_eV_per_atom=de, force_MSE_eV2_A2=fmse,
                force_MAE_eV_A=float(np.mean(np.abs(df))),
                loss=args.energy_weight * de**2 + args.force_weight * fmse)


def aggregate(rows):
    if not rows:
        raise ValueError("Cannot report metrics for an empty subset")
    de = np.asarray([r["dE_eV_per_atom"] for r in rows])
    fmse = np.asarray([r["force_MSE_eV2_A2"] for r in rows])
    n = np.asarray([r["n_atoms"] for r in rows])
    return dict(
        configurations=len(rows), atoms=int(n.sum()), loss=float(np.mean([r["loss"] for r in rows])),
        energy_RMSE_meV_per_atom=float(1000 * np.sqrt(np.mean(de**2))),
        energy_MAE_meV_per_atom=float(1000 * np.mean(np.abs(de))),
        energy_bias_meV_per_atom=float(1000 * np.mean(de)),
        force_RMSE_structure_balanced_eV_A=float(np.sqrt(np.mean(fmse))),
        force_RMSE_atom_weighted_eV_A=float(np.sqrt(np.average(fmse, weights=n))),
        force_MAE_structure_balanced_eV_A=float(np.mean([r["force_MAE_eV_A"] for r in rows])),
    )


def training_step(ctx, model, optimizer, record, args):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    energy, forces = prediction(ctx, model, record, create_graph=True)
    returned = ctx.trainer.structure_loss(
        energy_total=energy, forces=forces, record=record,
        energy_weight=args.energy_weight, force_weight=args.force_weight, device=ctx.device)
    loss = returned[0] if isinstance(returned, (tuple, list)) else returned
    row = metric_row(energy, forces, record, args)
    if not math.isfinite(float(loss.detach().cpu())):
        raise FloatingPointError("Nonfinite loss: " + record.source_file)
    if not math.isclose(float(loss.detach().cpu()), row["loss"], rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError("Original trainer loss differs from the documented objective")
    loss.backward()
    norm = ctx.torch.nn.utils.clip_grad_norm_(
        model.parameters(), args.grad_clip, error_if_nonfinite=True)
    row["gradient_norm_before_clip"] = float(norm.detach().cpu())
    optimizer.step()
    return row


def evaluate_records(ctx, model, records, args, stop_requested=None):
    model.eval()
    rows = []
    # Force evaluation still needs coordinate autograd; never wrap in no_grad.
    with ctx.torch.enable_grad():
        for record in records:
            energy, forces = prediction(ctx, model, record, create_graph=False)
            rows.append(metric_row(energy, forces, record, args))
            del energy, forces
            if stop_requested is not None and stop_requested():
                raise RequestedPause()
    return rows


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def common_fields(ctx, args, contract_hash):
    return dict(input_dim=392, hidden_dims=list(args.hidden_dims),
                feature_mean=ctx.mean.detach().cpu(), feature_std=ctx.std.detach().cpu(),
                e_ref_eV_per_atom=ctx.e_ref, energy_weight=args.energy_weight,
                force_weight=args.force_weight, seed=args.seed,
                training_contract_sha256=contract_hash, stage="expanded_common_energy",
                energy_reference_policy="unchanged_constant_from_baseline_common_checkpoint")


def save_torch(ctx, path, payload):
    io.atomic_write(path, lambda temporary: ctx.torch.save(payload, temporary))


def smoke_check(ctx, inputs, records, args, out, contract_hash):
    marker = out / "startup_check.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get("status") == "PASS" and previous.get("training_contract_sha256") == contract_hash:
            print("Verified existing common-model startup check: PASS", flush=True)
            return
    record = max(records, key=lambda r: (r.n_atoms, r.source_file))
    print("Common-model startup check: {} N={} (two disposable training steps)".format(
        record.source_file, record.n_atoms), flush=True)
    model = new_model(ctx, args)
    optimizer = new_optimizer(ctx, model, args)
    original_live = ctx.trainer.live_features
    check_count = 0

    def checked_live(*a, **kw):
        nonlocal check_count
        x = original_live(*a, **kw)
        if not x.requires_grad:
            raise RuntimeError("Live common features are detached from coordinates")
        if check_count == 0:
            with np.load(record.feature_file, allow_pickle=False) as d:
                raw = ctx.torch.from_numpy(d["X_local_torchsoap"]).to(ctx.device)
            expected = (raw - ctx.mean) / ctx.std
            if not ctx.torch.allclose(x.detach(), expected, atol=1e-5, rtol=1e-5):
                raise ValueError("Common live features disagree with normalized cached features")
        check_count += 1
        return x

    ctx.trainer.live_features = checked_live
    temporary_checkpoint = out / "startup_common_checkpoint.pt"
    try:
        ctx.torch.cuda.reset_peak_memory_stats(ctx.device)
        before = cpu_state(model)
        steps = []
        for _ in range(2):
            ctx.torch.cuda.synchronize(ctx.device)
            start = time.perf_counter()
            row = training_step(ctx, model, optimizer, record, args)
            ctx.torch.cuda.synchronize(ctx.device)
            row["elapsed_s"] = time.perf_counter() - start
            steps.append(row)
        if check_count != 2 or all(ctx.torch.equal(before[k], v.detach().cpu())
                                   for k, v in model.state_dict().items()):
            raise RuntimeError("Startup did not execute live feature training/update")
        # Verify the actual existing C1b checkpoint loader, not a guessed schema.
        payload = common_fields(ctx, args, contract_hash)
        payload["model_state_dict"] = cpu_state(model)
        save_torch(ctx, temporary_checkpoint, payload)
        next_trainer = import_file(inputs["sources"][C1B], "expanded_common_c1b_compatibility")
        loaded, mean, std, e_ref, _ = next_trainer.load_common_checkpoint(
            temporary_checkpoint, ctx.device)
        if (not ctx.torch.equal(mean, ctx.mean) or not ctx.torch.equal(std, ctx.std)
                or e_ref != ctx.e_ref):
            raise RuntimeError("C1b loader changed common-model preprocessing")
        model.eval()
        with ctx.torch.no_grad():
            x = ctx.torch.zeros((3, 392), dtype=ctx.torch.float32, device=ctx.device)
            if not ctx.torch.allclose(model(x).reshape(-1), loaded(x).reshape(-1), atol=1e-7, rtol=1e-6):
                raise RuntimeError("C1b common-network predictions disagree")
        result = dict(status="PASS", training_contract_sha256=contract_hash,
                      record_id=record.source_file, split="train", n_atoms=record.n_atoms,
                      targets="actual_MATPes_labels", steps=steps,
                      peak_allocated_GiB=ctx.torch.cuda.max_memory_allocated(ctx.device) / 1024**3,
                      peak_reserved_GiB=ctx.torch.cuda.max_memory_reserved(ctx.device) / 1024**3,
                      C1b_common_loader="PASS", weights_discarded=True, finished=io.now())
        io.write_json(marker, result)
        print("Startup PASS: {:.3f} GiB allocated, {:.3f} GiB reserved; C1b loader PASS".format(
            result["peak_allocated_GiB"], result["peak_reserved_GiB"]), flush=True)
    except Exception as exc:
        io.write_json(marker, dict(status="FAIL", record_id=record.source_file,
                                  training_contract_sha256=contract_hash, error=str(exc),
                                  traceback=traceback.format_exc(), finished=io.now()))
        raise
    finally:
        ctx.trainer.live_features = original_live
        temporary_checkpoint.unlink(missing_ok=True)
    del model, optimizer, loaded, payload, before
    ctx.torch.cuda.empty_cache()


def make_checkpoint(ctx, model, optimizer, scheduler, rng, args, contract_hash, state):
    result = dict(state)
    result.update(common_fields(ctx, args, contract_hash))
    result.update(model_state_dict=cpu_state(model),
                  optimizer_state_dict=optimizer.state_dict(),
                  scheduler_state_dict=scheduler.state_dict(),
                  rng_state=rng.bit_generator.state, numpy_rng_state=np.random.get_state(),
                  python_rng_state=random.getstate(), torch_rng_state=ctx.torch.get_rng_state(),
                  cuda_rng_state_all=ctx.torch.cuda.get_rng_state_all())
    return result


def publish_reports(ctx, checkpoint, out):
    io.write_csv(out / "training_history.csv", checkpoint["history"],
                 fields=list(checkpoint["history"][0]) if checkpoint["history"] else ["epoch"])
    if checkpoint["best_model_state_dict"] is not None:
        best = {k: checkpoint[k] for k in (
            "input_dim", "hidden_dims", "feature_mean", "feature_std", "e_ref_eV_per_atom",
            "energy_weight", "force_weight", "seed", "training_contract_sha256", "stage",
            "energy_reference_policy", "best_epoch", "best_val_loss")}
        best.update(model_state_dict=checkpoint["best_model_state_dict"],
                    epoch=checkpoint["best_epoch"], selection_split="val")
        save_torch(ctx, out / "best_checkpoint.pt", best)


def restore(ctx, checkpoint, model, optimizer, scheduler, rng):
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    rng.bit_generator.state = checkpoint["rng_state"]
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    ctx.torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    ctx.torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng_state_all"]])


@contextmanager
def exclusive_run(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is using this output run: " + str(out))
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def setup(args, out):
    from types import SimpleNamespace
    import torch
    if int(os.environ.get("SLURM_NTASKS", "1")) != 1:
        raise RuntimeError("This trainer is single-process; launch srun -n1")
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside the GPU allocation with silica_mpi activated")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    inputs = verify_inputs(args.features)
    (out / "src_snapshot").mkdir(exist_ok=True)
    for name, path in list(inputs["sources"].items()):
        target = out / "src_snapshot" / name
        if target.exists():
            checked_hash(target, inputs["binding"]["source_sha256"][name])
        else:
            io.snapshot(path, target)
        inputs["sources"][name] = target
    for path in (Path(__file__).resolve(), Path(io.__file__).resolve()):
        target = out / "src_snapshot" / path.name
        if target.exists():
            checked_hash(target, io.sha256(path))
        else:
            io.snapshot(path, target)
    trainer = import_file(inputs["sources"][COMMON], "expanded_moe_common_original")
    trainer.seed_everything(args.seed)
    records = load_records(inputs, trainer, ("train", "val") if args.command == "train" else ("train",))
    mean_np, std_np = normalization(trainer, records["train"], inputs["raw_mean"])
    old = torch.load(inputs["reference"], map_location="cpu", weights_only=False)
    e_ref = float(old["e_ref_eV_per_atom"])
    if not math.isfinite(e_ref):
        raise ValueError("Nonfinite baseline energy reference")
    del old
    soap = trainer.import_torchsoap(inputs["sources"][SOAP])
    descriptor = soap.TorchSOAPDescriptor(soap.TorchSOAPConfig(
        species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    descriptor.eval()
    for parameter in descriptor.parameters():
        parameter.requires_grad_(False)
    ctx = SimpleNamespace(torch=torch, trainer=trainer, descriptor=descriptor, device=device,
                          mean=torch.from_numpy(mean_np).to(device),
                          std=torch.from_numpy(std_np).to(device), e_ref=e_ref)
    contract = dict(schema_version=1, inputs=inputs["binding"],
                    adapter_sha256=io.sha256(Path(__file__)), io_helper_sha256=io.sha256(Path(io.__file__)),
                    settings={k: getattr(args, k) for k in SETTING_NAMES},
                    normalization_sha256=vector_digest(mean_np, std_np),
                    normalization_policy="original_compute_feature_normalization_on_train_only",
                    e_ref_eV_per_atom=e_ref, energy_reference_policy="preserve_baseline_constant",
                    initialization="fresh_common_network", geometry_dtype="float64", network_dtype="float32",
                    optimizer="AdamW", scheduler="ReduceLROnPlateau", scheduler_factor=0.5,
                    selection_split="val", force_metric="structure_balanced",
                    torch_version=str(torch.__version__), cuda_version=str(torch.version.cuda),
                    numpy_version=np.__version__, python_version=sys.version.split()[0])
    io.publish_contract(out / "training_contract.json", contract)
    io.write_npz(out / "common_normalization.npz", dict(
        feature_mean=mean_np, feature_std=std_np,
        n_training_atoms=inputs["binding"]["training_atoms"],
        n_training_structures=inputs["binding"]["training_structures"]))
    return ctx, inputs, records, io.sha256(out / "training_contract.json")


class RequestedPause(Exception):
    pass


def train(ctx, inputs, records, args, out, contract_hash, start_time):
    torch = ctx.torch
    latest = out / "latest_checkpoint.pt"
    if latest.exists() and not args.resume:
        raise FileExistsError("Use --resume to continue this run")
    smoke_check(ctx, inputs, records["train"], args, out, contract_hash)
    # Startup model updates and random draws never enter the production model.
    ctx.trainer.seed_everything(args.seed)
    model = new_model(ctx, args)
    optimizer = new_optimizer(ctx, model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience)
    rng = np.random.default_rng(args.seed)
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        if ck["training_contract_sha256"] != contract_hash:
            raise ValueError("Latest checkpoint belongs to different inputs/settings")
        restore(ctx, ck, model, optimizer, scheduler, rng)
        state = {k: ck[k] for k in (
            "epoch", "best_epoch", "best_val_loss", "best_model_state_dict", "patience_reference_loss",
            "bad_epochs", "history", "stop_reason")}
        publish_reports(ctx, ck, out)
        del ck
        print("Resuming after completed epoch {}".format(state["epoch"]), flush=True)
    else:
        state = dict(epoch=0, best_epoch=0, best_val_loss=None, best_model_state_dict=None,
                     patience_reference_loss=None, bad_epochs=0, history=[], stop_reason=None)
        save_torch(ctx, latest, make_checkpoint(ctx, model, optimizer, scheduler, rng, args, contract_hash, state))
    print("Common training: {} train / {} val; {} test reserved".format(
        len(records["train"]), len(records["val"]), inputs["binding"]["counts"]["test"]), flush=True)
    print("e_ref={:.15g} eV/atom; train atoms={}; one structure/step".format(
        ctx.e_ref, inputs["binding"]["training_atoms"]), flush=True)
    stop = dict(requested=False)

    def on_signal(signum, frame):
        stop["requested"] = True
        print("Signal {}: will stop after the current structure; partial epoch will replay.".format(signum),
              flush=True)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, on_signal)
    pause_reason = None
    while state["epoch"] < args.max_epochs and state["stop_reason"] != "early_stopping":
        if stop["requested"]:
            pause_reason = "signal"
            break
        elapsed = time.monotonic() - start_time
        estimate = state["history"][-1]["epoch_seconds"] if state["history"] else 0.0
        if args.max_hours > 0 and elapsed + 1.2 * estimate >= args.max_hours * 3600:
            pause_reason = "walltime_budget"
            break
        epoch = state["epoch"] + 1
        epoch_start = time.monotonic()
        io.write_json(out / "run_status.json", dict(
            status="TRAINING", current_epoch=epoch, last_completed_epoch=state["epoch"],
            host=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID"), updated=io.now()))
        try:
            online_rows = []
            for step, index in enumerate(rng.permutation(len(records["train"])), 1):
                record = records["train"][int(index)]
                online_rows.append(training_step(ctx, model, optimizer, record, args))
                if stop["requested"]:
                    raise RequestedPause()
                if step == 1 or step % args.log_every == 0 or step == len(records["train"]):
                    print("epoch={} step={}/{} record={} N={} loss={:.6g}".format(
                        epoch, step, len(records["train"]), record.source_file,
                        record.n_atoms, online_rows[-1]["loss"]), flush=True)
            optimizer.zero_grad(set_to_none=True)
            validation_rows = evaluate_records(
                ctx, model, records["val"], args, stop_requested=lambda: stop["requested"])
            if stop["requested"]:
                raise RequestedPause()
        except RequestedPause:
            pause_reason = "signal_partial_epoch_replayed_on_resume"
            break
        train_metrics, val_metrics = aggregate(online_rows), aggregate(validation_rows)
        val_loss = val_metrics["loss"]
        if state["best_val_loss"] is None or val_loss < state["best_val_loss"]:
            state.update(best_val_loss=val_loss, best_epoch=epoch, best_model_state_dict=cpu_state(model))
        if (state["patience_reference_loss"] is None
                or val_loss < state["patience_reference_loss"] - args.min_delta):
            state.update(patience_reference_loss=val_loss, bad_epochs=0)
        else:
            state["bad_epochs"] += 1
        scheduler.step(val_loss)
        history_row = dict(epoch=epoch, epoch_seconds=time.monotonic() - epoch_start,
                           learning_rate_next_epoch=float(optimizer.param_groups[0]["lr"]),
                           best_epoch=state["best_epoch"], bad_epochs=state["bad_epochs"])
        history_row.update({"train_online_" + k: v for k, v in train_metrics.items()})
        history_row.update({"val_" + k: v for k, v in val_metrics.items()})
        state["history"].append(history_row)
        state["epoch"] = epoch
        state["stop_reason"] = ("early_stopping" if state["bad_epochs"] >= args.patience else
                                "max_epochs" if epoch >= args.max_epochs else None)
        checkpoint = make_checkpoint(ctx, model, optimizer, scheduler, rng, args, contract_hash, state)
        save_torch(ctx, latest, checkpoint)
        publish_reports(ctx, checkpoint, out)
        del checkpoint
        print("epoch={} val: E_RMSE={:.4f} meV/atom F_RMSE={:.6f} eV/A loss={:.6g} best={}".format(
            epoch, val_metrics["energy_RMSE_meV_per_atom"],
            val_metrics["force_RMSE_structure_balanced_eV_A"], val_loss, state["best_epoch"]), flush=True)
    complete = state["stop_reason"] in ("early_stopping", "max_epochs") and pause_reason is None
    result = dict(status="COMPLETE" if complete else "PAUSED", last_completed_epoch=state["epoch"],
                  best_epoch=state["best_epoch"], best_val_loss=state["best_val_loss"],
                  reason=state["stop_reason"] if complete else pause_reason,
                  test_evaluated=False, finished=io.now())
    io.write_json(out / "run_status.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    print("Latest: " + str(latest), flush=True)
    if state["best_epoch"]:
        print("Best common checkpoint: " + str(out / "best_checkpoint.pt"), flush=True)
    return 0


def final_evaluation(ctx, inputs, args, out, contract_hash):
    checkpoint = ctx.torch.load(out / "latest_checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint["training_contract_sha256"] != contract_hash:
        raise ValueError("Checkpoint contract mismatch")
    if checkpoint["stop_reason"] not in ("early_stopping", "max_epochs"):
        raise RuntimeError("Training is paused/incomplete. Resume it before held-out evaluation.")
    if checkpoint["best_model_state_dict"] is None:
        raise RuntimeError("No completed validation-selected checkpoint")
    publish_reports(ctx, checkpoint, out)
    model = new_model(ctx, args)
    model.load_state_dict(checkpoint["best_model_state_dict"])
    records = load_records(inputs, ctx.trainer, args.splits)
    metadata = {r["record_id"]: r for r in inputs["rows"]}
    report_rows, configuration_rows = [], []
    for split in args.splits:
        rows = evaluate_records(ctx, model, records[split], args)
        for row in rows:
            source = metadata[row["record_id"]]
            row.update(source=source["source"], family=source["family"],
                       phase_pair_group=source["phase_pair_group"], split_group=source["split_group"])
        configuration_rows.extend(rows)
        report_rows.append(dict(split=split, source="ALL", family="ALL", **aggregate(rows)))
        for source in sorted({r["source"] for r in rows}):
            subset = [r for r in rows if r["source"] == source]
            report_rows.append(dict(split=split, source=source, family="ALL", **aggregate(subset)))
            if source == "Siddharth":
                for family in sorted({r["family"] for r in subset}):
                    report_rows.append(dict(split=split, source=source, family=family,
                                            **aggregate([r for r in subset if r["family"] == family])))
    folder = out / ("evaluation_" + "_".join(args.splits))
    folder.mkdir(exist_ok=True)
    io.write_csv(folder / "per_configuration.csv", configuration_rows)
    io.write_csv(folder / "metrics.csv", report_rows)
    io.write_json(folder / "evaluation_summary.json", dict(
        finished=io.now(), selected_epoch=checkpoint["best_epoch"], selection_split="val",
        training_contract_sha256=contract_hash,
        best_checkpoint_sha256=io.sha256(out / "best_checkpoint.pt"),
        evaluated_splits=args.splits, metrics=report_rows,
        gamma_evaluated=False, region_metrics_computed=False,
        note="Siddharth test covers CHA_AEI only; no Siddharth validation data exist."))
    for row in report_rows:
        print("{} {} {} Ncfg={} E_RMSE={:.4f} meV/atom F_RMSE={:.6f} eV/A".format(
            row["split"], row["source"], row["family"], row["configurations"],
            row["energy_RMSE_meV_per_atom"], row["force_RMSE_structure_balanced_eV_A"]), flush=True)
    print("Evaluation: " + str(folder), flush=True)
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("train", "evaluate"))
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Start or resume a matching run; never overwrite it.")
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128])
    p.add_argument("--energy-weight", type=float, default=10.0)
    p.add_argument("--force-weight", type=float, default=1000.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-7)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="Budget for this invocation; checked at epoch boundaries. Zero means unlimited.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--splits", choices=SPLITS, nargs="+", default=["val", "test"],
                   help="Only used by evaluate, after checkpoint selection has finished.")
    args = p.parse_args()
    positive = ("energy_weight", "force_weight", "lr", "grad_clip",
                "max_epochs", "patience", "threads", "log_every")
    nonnegative = ("weight_decay", "scheduler_patience", "min_delta", "max_hours")
    if (any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in positive)
            or any(not math.isfinite(getattr(args, k)) or getattr(args, k) < 0 for k in nonnegative)
            or any(n <= 0 for n in args.hidden_dims)):
        p.error("Invalid numerical training option")
    if len(set(args.splits)) != len(args.splits):
        p.error("Do not repeat evaluation splits")
    return args


def main():
    args = parse_args()
    out = args.out.expanduser().resolve()
    if args.command == "evaluate":
        if not (out / "training_contract.json").is_file():
            raise FileNotFoundError("No training contract in " + str(out))
        saved = json.loads((out / "training_contract.json").read_text())
        for name in SETTING_NAMES:
            setattr(args, name, saved["settings"][name])
    elif out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Output is not empty; use --resume for this run, or a new --out")
    start = time.monotonic()
    with exclusive_run(out):
        try:
            ctx, inputs, records, contract_hash = setup(args, out)
            if args.command == "evaluate":
                return final_evaluation(ctx, inputs, args, out, contract_hash)
            return train(ctx, inputs, records, args, out, contract_hash, start)
        except Exception as exc:
            # Record errors separately: a failed command never turns an existing
            # completed checkpoint into a success or erases its run status.
            io.write_json(out / "last_error.json", dict(
                command=args.command, error=type(exc).__name__ + ": " + str(exc),
                traceback=traceback.format_exc(), failed=io.now()))
            raise


if __name__ == "__main__":
    raise SystemExit(main())
