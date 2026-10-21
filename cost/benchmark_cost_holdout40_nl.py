#!/usr/bin/env python3
"""
benchmark_cost_holdout40.py

Computational cost of the holdout-40 Local-MoE model against MACE-MATPes-r2SCAN
on the holdout-40 TEST structures (the 105 structures never used for training or
model selection), restricted to N <= --max-atoms (default 3000).

What is timed (one forced energy + force evaluation, atoms in, numpy out):

  matpes64  mace_mp("mace-matpes-r2scan-0"), float64 (as in the earlier benchmark)
  matpes32  the same model in float32 (the usual production precision of MACE)
  common    Local-MoE common network alone: TorchSOAP + common MLP + autograd forces
  full      Local-MoE full model: TorchSOAP + common MLP + router + residual head
            + autograd forces, through the trainer's own predict_energy_forces()

The two Local-MoE rows run with the precision the model was trained and evaluated
in: float32 networks and features, float64 positions and energy sums (the same code
path that produced every accuracy number). Descriptor evaluation, the Atoms ->
record conversion, the host-to-device copies and the copy of the forces back to the
host are all inside the timed region. MATPes is timed through its ASE calculator
(neighbour list, model, forces back to the host), exactly as in the earlier benchmark.

Protocol per (model, structure): warm-up calls (default 5; only 2 if a call takes
more than 0.5 s), then repeated forced evaluations with a CUDA synchronisation before
and after every call; at least --min-repeat and at most --n-repeat repeats, stopping
early once the repeats have used --cap-s seconds. Reported: median time, and the
peak allocated GPU memory (torch.cuda.max_memory_allocated after resetting the peak
statistics following the warm-up).

Gates (run before timing, nothing is timed if one fails):
  G1  common-only energy/forces == full model with every residual parameter zeroed
      (same process, tolerance --tol-same)
  G2  full model reproduces the stored predictions of the parity export on the
      gate structures (tolerance --tol-ref; needs --reference-dir)
  G3  GPU is not in use by another process (used memory <= --busy-mib)
  info  MATPes float64 against the stored labels (these were produced by the same
        model); printed, with a warning above 1e-3

Results are appended row by row to <out>/computational_cost_raw.csv, so a run that
is cut off keeps what it measured; --resume continues it. Without --resume an
existing raw CSV is never overwritten.

Run from any directory, inside a GPU allocation, on an otherwise idle GPU:
    CUDA_VISIBLE_DEVICES=0 python benchmark_cost_holdout40.py --smoke
    CUDA_VISIBLE_DEVICES=0 python benchmark_cost_holdout40.py
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
import copy
import csv
import gc
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HET_DEFAULT = (_os.environ.get("MOE_HET", "."))
REPO_DEFAULT = str(REPO_ROOT)
MIB = 1024.0 ** 2
LABEL = {"matpes64": "MATPes-r2SCAN", "matpes32": "MATPes-r2SCAN (float32)",
         "common": "Local-MoE common network", "full": "Local-MoE full"}
PRECISION = {"matpes64": "float64", "matpes32": "float32",
             "common": "float32 nets / float64 positions", "full": "float32 nets / float64 positions"}
MODEL_KEYS = tuple(LABEL)
FIELDS = ["model", "model_key", "precision", "record_index", "record_id", "phase_pair", "n_atoms",
          "images_enumerated", "status", "n_warmup", "n_repeat", "median_s", "mean_s", "std_s", "min_s",
          "p90_s", "steps_per_s", "atoms_per_s", "base_gpu_allocated_MiB", "peak_gpu_allocated_MiB",
          "incremental_peak_MiB", "last_energy_eV", "last_fmax_eV_A", "gpu"]

torch = None  # imported in main(); a module-level name so the helpers below can use it


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def import_file(path, name):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plane_spacings(cell):
    c = np.asarray(cell, float)
    vol = abs(np.linalg.det(c))
    return np.array([vol / np.linalg.norm(np.cross(c[(k + 1) % 3], c[(k + 2) % 3])) for k in range(3)])


def record_from_atoms(trainer, atoms, name):
    """Wrap an ASE structure into the trainer's StructureRecord (targets are unused dummies)."""
    n = len(atoms)
    z = np.asarray(atoms.numbers).astype(np.int16)
    symbols, masses = trainer.build_symbols_masses(z)
    return trainer.StructureRecord(
        source_file=name, phase_pair="", split="bench", n_atoms=n, feature_file=Path(""), source_shard=Path(""),
        atomic_numbers=z, positions_np=np.array(atoms.positions, dtype=np.float64),
        cell_np=np.array(atoms.cell, dtype=np.float64),
        phase=np.zeros(n, dtype=np.int16), interface_center=np.zeros(n, dtype=np.int16),
        target_energy_total_eV=0.0, target_forces_np=np.zeros((n, 3)), symbols=symbols, masses=masses)


def sync():
    torch.cuda.synchronize()


def is_oom(exc):
    return exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    sync()


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
def bench(fn, atoms, n_warmup, n_repeat, min_repeat, cap_s, gpu_name):
    """Time fn(atoms) -> (energy, forces). Returns the measured columns."""
    used = 0
    for i in range(n_warmup):
        sync()
        t0 = time.perf_counter()
        e, f = fn(atoms)
        sync()
        dt = time.perf_counter() - t0
        used += 1
        if not np.isfinite(e) or not np.all(np.isfinite(f)):
            raise FloatingPointError("non-finite energy or forces")
        if dt > 0.5 and i >= 1:
            break
    sync()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    t_start = time.perf_counter()
    for rep in range(n_repeat):
        sync()
        t0 = time.perf_counter()
        e, f = fn(atoms)
        sync()
        times.append(time.perf_counter() - t0)
        if len(times) >= min_repeat and (time.perf_counter() - t_start) > cap_s:
            break
    peak = torch.cuda.max_memory_allocated()
    arr = np.asarray(times, float)
    med = float(np.median(arr))
    f = np.asarray(f, float)
    return {
        "n_warmup": used, "n_repeat": len(arr), "median_s": med, "mean_s": float(arr.mean()),
        "std_s": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"), "min_s": float(arr.min()),
        "p90_s": float(np.percentile(arr, 90.0)), "steps_per_s": 1.0 / med, "atoms_per_s": len(atoms) / med,
        "base_gpu_allocated_MiB": base / MIB, "peak_gpu_allocated_MiB": peak / MIB,
        "incremental_peak_MiB": (peak - base) / MIB, "last_energy_eV": float(e),
        "last_fmax_eV_A": float(np.linalg.norm(f, axis=1).max()), "gpu": gpu_name,
    }


class RawCsv:
    def __init__(self, path, resume):
        self.path = Path(path)
        self.done = set()
        if self.path.exists() and self.path.stat().st_size > 0:
            if not resume:
                raise SystemExit(f"{self.path} exists; use --resume to continue it or choose another --out "
                                 "(it is never overwritten)")
            with open(self.path, newline="") as f:
                for r in csv.DictReader(f):
                    if r["status"] in ("ok", "OOM"):
                        self.done.add((r["model_key"], r["record_id"]))
            self.fresh = False
        else:
            self.fresh = True

    def add(self, row):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if self.fresh:
                w.writeheader()
                self.fresh = False
            w.writerow({k: row.get(k, "") for k in FIELDS})
            f.flush()
            os.fsync(f.fileno())
        self.done.add((row["model_key"], row["record_id"]))


# ---------------------------------------------------------------------------
# Local-MoE builders
# ---------------------------------------------------------------------------
def build_local(trainer, paths, device, which):
    """which: 'common' (common net + descriptor), 'full' or 'gate' (everything)."""
    common_model, common_mean, common_std, e_ref, _ = trainer.load_common_checkpoint(paths["common"], device)
    soap = trainer.import_torchsoap(paths["soap"])
    descriptor = soap.TorchSOAPDescriptor(
        soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    for p in descriptor.parameters():
        p.requires_grad_(False)
    parts = dict(common_model=common_model, common_mean=common_mean, common_std=common_std, e_ref=e_ref,
                 descriptor=descriptor, device=device)
    if which in ("full", "gate"):
        routing_model, routing_mean, routing_std, _ = trainer.load_routing_checkpoint(paths["routing"], device)
        bck = torch.load(paths["c1b"], map_location="cpu", weights_only=False)
        if [str(x) for x in bck["expert_names"]] != list(trainer.EXPERTS):
            raise SystemExit("C1b expert order does not match the trainer")
        residual_model = trainer.MLP(input_dim=int(bck["input_dim"]),
                                     hidden_dims=[int(x) for x in bck["residual_hidden_dims"]],
                                     output_dim=7, dropout=0.0).to(device)
        residual_model.load_state_dict(bck["residual_model_state_dict"])
        residual_model.eval()
        parts.update(routing_model=routing_model, routing_mean=routing_mean, routing_std=routing_std,
                     residual_model=residual_model, residual_mean=common_mean, residual_std=common_std)
    return parts


def make_common_fn(trainer, parts):
    device = parts["device"]

    def fn(atoms):
        rec = record_from_atoms(trainer, atoms, "bench")
        positions = torch.tensor(rec.positions_np, dtype=torch.float64, device=device, requires_grad=True)
        cell = torch.tensor(rec.cell_np, dtype=torch.float64, device=device)
        with torch.enable_grad():
            x = trainer.live_raw_features(record=rec, positions=positions, cell=cell, descriptor=parts["descriptor"])
            atomic = parts["common_model"]((x - parts["common_mean"]) / parts["common_std"]).squeeze(-1)
            total = positions.new_tensor(rec.n_atoms * parts["e_ref"]) + atomic.sum().to(dtype=torch.float64)
            forces = -torch.autograd.grad(total, positions, create_graph=False, retain_graph=False,
                                          allow_unused=False)[0]
        return float(total.detach().cpu()), forces.detach().cpu().numpy()
    return fn


def make_full_fn(trainer, parts, residual_override=None):
    device = parts["device"]
    residual = parts["residual_model"] if residual_override is None else residual_override

    def fn(atoms):
        rec = record_from_atoms(trainer, atoms, "bench")
        with torch.enable_grad():
            terms, forces = trainer.predict_energy_forces(
                record=rec, descriptor=parts["descriptor"], common_model=parts["common_model"],
                common_mean=parts["common_mean"], common_std=parts["common_std"],
                routing_model=parts["routing_model"], routing_mean=parts["routing_mean"],
                routing_std=parts["routing_std"], residual_model=residual,
                residual_mean=parts["residual_mean"], residual_std=parts["residual_std"],
                e_ref=parts["e_ref"], device=device, create_graph=False)
        return float(terms["total_energy"].detach().cpu()), forces.detach().cpu().numpy()
    return fn


def make_matpes_fn(calc):
    from ase.calculators.calculator import all_changes

    def fn(atoms):
        calc.calculate(atoms, properties=["energy", "forces"], system_changes=all_changes)
        return float(calc.results["energy"]), np.asarray(calc.results["forces"], dtype=float)
    return fn


def load_reference(ref_dir, rid):
    p = Path(ref_dir) / (str(rid).replace("/", "__") + ".npz")
    if not p.is_file():
        raise SystemExit(f"reference file missing: {p}")
    with np.load(p, allow_pickle=False) as d:
        if str(d["record_id"]) != str(rid):
            raise SystemExit(f"{p}: record_id {d['record_id']} != {rid}")
        return {k: np.asarray(d[k], float) for k in ("E_reference_eV", "E_predicted_eV",
                                                       "reference_forces", "predicted_forces")}


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def summarize(raw_path, out_dir, meta):
    with open(raw_path, newline="") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault(r["model_key"], {})[r["record_id"]] = r
    summary = {"meta": meta, "models": {}, "speedups": {}}
    print("\n" + "=" * 100)
    print(f"{'model':32s} {'n_ok':>5s} {'n_oom':>5s} {'median t (ms)':>14s} {'max N':>6s} {'peak mem max (GiB)':>19s}")
    for k in MODEL_KEYS:
        if k not in by:
            continue
        ok = [r for r in by[k].values() if r["status"] == "ok"]
        oom = [r for r in by[k].values() if r["status"] == "OOM"]
        t = np.array([float(r["median_s"]) for r in ok]) if ok else np.array([np.nan])
        m = np.array([float(r["peak_gpu_allocated_MiB"]) for r in ok]) if ok else np.array([np.nan])
        nmax = max((int(r["n_atoms"]) for r in ok), default=0)
        summary["models"][k] = {"label": LABEL[k], "n_ok": len(ok), "n_oom": len(oom),
                                "median_of_medians_ms": float(np.median(t) * 1e3), "max_N": nmax,
                                "peak_mem_max_GiB": float(np.nanmax(m) / 1024.0)}
        print(f"{LABEL[k]:32s} {len(ok):5d} {len(oom):5d} {np.median(t) * 1e3:14.2f} {nmax:6d} {np.nanmax(m) / 1024.0:19.2f}")
    for ref in ("matpes64", "matpes32"):
        for k in ("common", "full"):
            if ref in by and k in by:
                both = [rid for rid in by[k] if rid in by[ref] and by[k][rid]["status"] == "ok"
                        and by[ref][rid]["status"] == "ok"]
                if both:
                    sp = np.array([float(by[ref][rid]["median_s"]) / float(by[k][rid]["median_s"]) for rid in both])
                    mr = np.array([100.0 * (1.0 - float(by[k][rid]["peak_gpu_allocated_MiB"]) /
                                            float(by[ref][rid]["peak_gpu_allocated_MiB"])) for rid in both])
                    summary["speedups"][f"{ref}/{k}"] = {
                        "n": len(both), "speedup_mean": float(sp.mean()), "speedup_median": float(np.median(sp)),
                        "speedup_min": float(sp.min()), "speedup_max": float(sp.max()),
                        "mem_reduction_median_pct": float(np.median(mr))}
                    print(f"speedup {LABEL[ref]} / {LABEL[k]}: n={len(both)} mean {sp.mean():.2f}  median "
                          f"{np.median(sp):.2f}  min {sp.min():.2f}  max {sp.max():.2f} | memory reduction median "
                          f"{np.median(mr):.1f}%")
    with open(Path(out_dir) / "computational_cost_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
def main():
    global torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--het", default=HET_DEFAULT)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--c1b-script", type=Path, default=None)
    ap.add_argument("--stage-root", type=Path, default=None)
    ap.add_argument("--common-checkpoint", type=Path, default=None)
    ap.add_argument("--routing-checkpoint", type=Path, default=None)
    ap.add_argument("--c1b-checkpoint", type=Path, default=None, help="default: runs/c1b_holdout40_e10000_v1")
    ap.add_argument("--torchsoap-module", type=Path, default=None)
    ap.add_argument("--reference-dir", type=Path, default=None,
                    help="raw_predictions/test of the parity export of the same C1b run")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--models", default=",".join(MODEL_KEYS), help="comma list from: " + ",".join(MODEL_KEYS))
    ap.add_argument("--max-atoms", type=int, default=3000)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-repeat", type=int, default=30)
    ap.add_argument("--min-repeat", type=int, default=7)
    ap.add_argument("--cap-s", type=float, default=2.0)
    ap.add_argument("--n-gate", type=int, default=3, help="smallest structures used for the gates")
    ap.add_argument("--tol-same", type=float, default=1e-7)
    ap.add_argument("--tol-ref", type=float, default=1e-4)
    ap.add_argument("--busy-mib", type=float, default=1024.0)
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="4 structures spread over the size range, 2 warm-ups, 5 repeats")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in models if m not in MODEL_KEYS]
    if bad or not models:
        raise SystemExit(f"--models: unknown {bad}; choose from {MODEL_KEYS}")

    import torch as _torch
    torch = _torch
    from ase import Atoms

    HET = Path(args.het)
    PROJ = HET / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    script = args.c1b_script or REPO_ROOT / 'training/train_c1b_holdout40.py'
    paths = {
        "common": (args.common_checkpoint or PROJ / "runs/common_holdout40_ew10000_v1/best_checkpoint.pt"),
        "routing": (args.routing_checkpoint or PROJ / "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt"),
        "c1b": (args.c1b_checkpoint or PROJ / "runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt"),
        "soap": (args.torchsoap_module or PROJ / "data/expanded_holdout40_v1_features/src_snapshot/torchsoap_autograd_module_periodic_regnorm.py"),
    }
    paths = {k: Path(v).expanduser().resolve() for k, v in paths.items()}
    stage = (args.stage_root or PROJ / "data/expanded_holdout40_v1_c1b_stage").expanduser().resolve()
    ref_dir = (args.reference_dir or PROJ / "eval_parity_v1/c1b_holdout40_e10000_v1/raw_predictions/test").expanduser().resolve()
    for p in [script, *paths.values()]:
        if not Path(p).is_file():
            raise SystemExit(f"MISSING FILE: {p}")
    if not ref_dir.is_dir():
        raise SystemExit(f"MISSING DIRECTORY: {ref_dir}")
    out = (args.out or PROJ / "eval_cost_v1").expanduser().resolve()
    if args.smoke:
        out = out.parent / (out.name + "_smoke_" + time.strftime("%m%d_%H%M%S"))
        args.n_warmup, args.n_repeat, args.min_repeat, args.cap_s = 2, 5, 5, 1.0
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "computational_cost_raw.csv"
    raw = RawCsv(raw_path, args.resume)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    gpu_name = torch.cuda.get_device_name(device)
    free, total = torch.cuda.mem_get_info()
    used_mib = (total - free) / MIB
    print(f"GPU: {gpu_name} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
          f"memory in use before start: {used_mib:.0f} MiB of {total / MIB:.0f}")
    if used_mib > args.busy_mib and not args.allow_busy:
        raise SystemExit(f"GATE G3 FAILED: {used_mib:.0f} MiB already in use on this GPU (limit {args.busy_mib:.0f}); "
                         "another process would distort the timings")

    trainer = import_file(script, "bench_c1b_trainer")
    torch.set_default_dtype(torch.float32)

    # ------------------------------------------------------------ structures
    test = sorted(trainer.load_records(stage)["test"], key=lambda r: (r.n_atoms, r.source_file))
    test = [(i, r) for i, r in enumerate(test)]
    n_test = len(test)
    chosen = [(i, r) for i, r in test if r.n_atoms <= args.max_atoms]
    dropped = [r.n_atoms for i, r in test if r.n_atoms > args.max_atoms]
    print(f"test structures: {n_test}; with N <= {args.max_atoms}: {len(chosen)} (N from {chosen[0][1].n_atoms} to "
          f"{chosen[-1][1].n_atoms}); dropped sizes: {dropped}")
    if args.smoke:
        sel = sorted(set(int(round(x)) for x in np.linspace(0, len(chosen) - 1, 4)))
        chosen = [chosen[k] for k in sel]
        print(f"smoke: structures N = {[r.n_atoms for _, r in chosen]}")
    gate_set = chosen[:args.n_gate]

    def to_atoms(rec):
        return Atoms(numbers=np.asarray(rec.atomic_numbers, dtype=int), positions=np.asarray(rec.positions_np, float),
                     cell=np.asarray(rec.cell_np, float), pbc=True)

    # ----------------------------------------------------------------- gates
    if any(m in models for m in ("common", "full")):
        print("\nGATES")
        parts = build_local(trainer, paths, device, "gate")
        fn_common = make_common_fn(trainer, parts)
        fn_full = make_full_fn(trainer, parts)
        zero = copy.deepcopy(parts["residual_model"])
        with torch.no_grad():
            for p in zero.parameters():
                p.zero_()
        fn_zero = make_full_fn(trainer, parts, residual_override=zero)
        w1e = w1f = w2e = w2f = 0.0
        for _, rec in gate_set:
            at = to_atoms(rec)
            ec, fc = fn_common(at)
            ez, fz = fn_zero(at)
            w1e, w1f = max(w1e, abs(ec - ez)), max(w1f, float(np.abs(fc - fz).max()))
            ef, ff = fn_full(at)
            ref = load_reference(ref_dir, rec.source_file)
            w2e = max(w2e, abs(ef - float(ref["E_predicted_eV"])))
            w2f = max(w2f, float(np.abs(ff - ref["predicted_forces"]).max()))
        print(f"  G1 common-only vs zero-residual full model on N={[r.n_atoms for _, r in gate_set]}: "
              f"max |dE| = {w1e:.2e} eV, max |dF| = {w1f:.2e} eV/A (tolerance {args.tol_same:g})")
        print(f"  G2 full model vs stored parity-export predictions: max |dE| = {w2e:.2e} eV, "
              f"max |dF| = {w2f:.2e} eV/A (tolerance {args.tol_ref:g})")
        if w1e > args.tol_same or w1f > args.tol_same:
            raise SystemExit("GATE G1 FAILED: the common-only path is not the zero-residual full model; nothing timed")
        if w2e > args.tol_ref or w2f > args.tol_ref:
            raise SystemExit("GATE G2 FAILED: the benchmark route does not reproduce the stored predictions; nothing timed")
        del parts, fn_common, fn_full, fn_zero, zero
        cleanup()

    meta = {"script": Path(__file__).name, "script_sha256": sha256(__file__), "gpu": gpu_name,
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "models": models,
            "n_structures": len(chosen), "max_atoms": args.max_atoms, "n_warmup": args.n_warmup,
            "n_repeat": args.n_repeat, "min_repeat": args.min_repeat, "cap_s": args.cap_s,
            "c1b_checkpoint": str(paths["c1b"]), "common_checkpoint": str(paths["common"]),
            "routing_checkpoint": str(paths["routing"]), "smoke": bool(args.smoke),
            "torchsoap_module": str(paths["soap"]), "torchsoap_sha256": sha256(paths["soap"])}

    # ---------------------------------------------------------------- timing
    def run_block(key, fn):
        todo = [(i, r) for i, r in chosen if (key, r.source_file) not in raw.done]
        print(f"\n{'=' * 100}\n{LABEL[key]} | precision: {PRECISION[key]} | {len(todo)} of {len(chosen)} structures to time "
              f"| torch default dtype {torch.get_default_dtype()}\n{'=' * 100}", flush=True)
        for i, rec in todo:
            at = to_atoms(rec)
            row = {"model": LABEL[key], "model_key": key, "precision": PRECISION[key], "record_index": i,
                   "record_id": rec.source_file, "phase_pair": rec.phase_pair, "n_atoms": rec.n_atoms,
                   "images_enumerated": bool(plane_spacings(rec.cell_np).min() <= 2.0 * 5.0), "gpu": gpu_name}
            try:
                row.update(bench(fn, at, args.n_warmup, args.n_repeat, args.min_repeat, args.cap_s, gpu_name))
                row["status"] = "ok"
                print(f"  N={rec.n_atoms:5d}  median={row['median_s'] * 1e3:9.2f} ms  repeats={row['n_repeat']:3d}  "
                      f"peak={row['peak_gpu_allocated_MiB']:9.1f} MiB  {rec.phase_pair}", flush=True)
            except Exception as exc:  # noqa: BLE001
                if not is_oom(exc):
                    raise
                row["status"] = "OOM"
                print(f"  N={rec.n_atoms:5d}  OUT OF MEMORY  {rec.phase_pair}", flush=True)
                cleanup()
            raw.add(row)

    for key in models:
        if key in ("matpes64", "matpes32"):
            try:
                from mace.calculators import mace_mp
            except ImportError as exc:
                raise SystemExit(f"mace is not importable in this environment ({exc}); run the Local-MoE models here "
                                 "with --models common,full and MATPes in the mace environment with "
                                 "--models matpes64,matpes32 --resume") from exc
            cleanup()
            calc = mace_mp(model="mace-matpes-r2scan-0", device="cuda",
                           default_dtype="float64" if key == "matpes64" else "float32")
            fn = make_matpes_fn(calc)
            if key == "matpes64":
                w_e = w_f = 0.0
                for _, rec in gate_set:
                    e, f = fn(to_atoms(rec))
                    ref = load_reference(ref_dir, rec.source_file)
                    w_e = max(w_e, abs(e - float(ref["E_reference_eV"])))
                    w_f = max(w_f, float(np.abs(f - ref["reference_forces"]).max()))
                print(f"  info: MATPes float64 here vs stored labels on N={[r.n_atoms for _, r in gate_set]}: "
                      f"max |dE| = {w_e:.2e} eV, max |dF| = {w_f:.2e} eV/A" + ("   WARNING: above 1e-3" if max(w_e, w_f) > 1e-3 else ""))
            run_block(key, fn)
            del calc, fn
        else:
            torch.set_default_dtype(torch.float32)
            cleanup()
            parts = build_local(trainer, paths, device, key)
            fn = make_common_fn(trainer, parts) if key == "common" else make_full_fn(trainer, parts)
            run_block(key, fn)
            del parts, fn
        cleanup()

    summarize(raw_path, out, meta)
    print(f"\nsaved: {raw_path} | {out / 'computational_cost_summary.json'}")


if __name__ == "__main__":
    main()
