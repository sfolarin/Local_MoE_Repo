#!/usr/bin/env python3
"""
run_local_moe_long_nve.py

Long NVE runs of the holdout-40 Local-MoE potential for the supplementary information (default: 500 ps = 1,000,000
steps at dt = 0.5 fs), restartable.

Each structure starts from the state the 10 ps protocol already reached after its NPT stage
(<md-root>/case_<id>/npt_final_structure.extxyz: positions, cell and momenta), is rescaled exactly to 300 K as before,
and is then integrated with ASE VelocityVerlet. Same potential, same dtype (float64 networks), same dt, same drift
definition as run_local_moe_md.py, so the first 10 ps are directly comparable with the earlier result:

    drift(t) = 1000 * (E_total(t)/N - E_total(0)/N)   in meV/atom,   conserving if max |drift| <= 0.1 meV/atom

Output per structure, in <out-root>/case_<id>/ :
  nve_500ps_log.csv               one row per --log-every steps (rewritten at every checkpoint, so it can be plotted
                                  while the run is going)
  nve_500ps_frames.extxyz         one frame per --frame-every steps (positions, cell, momenta)
  restart.npz                     exact (float64) state every --checkpoint-every steps
  long_nve_summary.json           drift statistics, wall time, steps/s, energy_conserving
  nve_500ps_final_structure.extxyz   completion marker, written last

A structure whose final-structure marker exists is skipped; one with a restart.npz resumes from it, so a job that is
killed at the wall-time limit loses at most --checkpoint-every steps and can simply be resubmitted.

Which structures
  --manifest long_manifest.csv --slot S    the rows with slot == S, in the order of the 'order' column
  --records external_0110,own_0136         explicit list
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
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HET_DEFAULT = (_os.environ.get("MOE_HET", "."))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--het", default=HET_DEFAULT)
    p.add_argument("--md-root", type=Path, default=None, help="finished 10 ps runs; default <proj>/md_holdout40_v1")
    p.add_argument("--out-root", type=Path, default=None, help="default <proj>/md_long500ps_v1")
    p.add_argument("--common-checkpoint", type=Path, default=None)
    p.add_argument("--routing-checkpoint", type=Path, default=None)
    p.add_argument("--c1b-checkpoint", type=Path, default=None)
    p.add_argument("--torchsoap-module", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--slot", type=int, default=None)
    p.add_argument("--records", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--nve-ps", type=float, default=500.0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--frame-every", type=int, default=10000)
    p.add_argument("--checkpoint-every", type=int, default=50000)
    p.add_argument("--print-every", type=int, default=20000)
    p.add_argument("--conserve-limit-meV", type=float, default=0.1)
    p.add_argument("--stop-after-steps", type=int, default=0, help=argparse.SUPPRESS)   # test hook: simulate a kill
    return p.parse_args()


def kept_rows(path, step0):
    """Rows of an existing log with step <= step0 (a resumed run continues from the checkpoint)."""
    if not Path(path).is_file():
        return []
    with open(path, newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    return [r for r in rows if int(float(r["step"])) <= step0]


def truncate_frames(path, step0):
    """Keep only the frames written up to the checkpoint step (frames after it will be written again)."""
    from ase.io import iread, write
    path = Path(path)
    if not path.is_file():
        return set()
    keep = [a for a in iread(str(path), format="extxyz") if int(a.info.get("step", -1)) <= step0]
    path.unlink()
    for a in keep:
        write(str(path), a, format="extxyz", append=True)
    return {int(a.info["step"]) for a in keep}


def run_one(rid, calc, args, md_root, out_root, meta, D):
    from ase import units
    from ase.io import read, write
    from ase.md.verlet import VelocityVerlet

    out = out_root / f"case_{rid}"
    out.mkdir(parents=True, exist_ok=True)
    final_file = out / "nve_500ps_final_structure.extxyz"
    if final_file.is_file() and final_file.stat().st_size > 0:
        print(f"SKIP COMPLETE: {rid}", flush=True)
        return "skipped"
    src = md_root / f"case_{rid}" / "npt_final_structure.extxyz"
    if not src.is_file():
        raise FileNotFoundError(f"no NPT-final state for {rid}: {src}")

    dt = args.dt_fs
    n_total = int(round(args.nve_ps * 1000.0 / dt))
    log_path, frames_path, restart_path = out / "nve_500ps_log.csv", out / "nve_500ps_frames.extxyz", out / "restart.npz"

    atoms = read(str(src), index=-1)
    n = len(atoms)
    if not atoms.has("momenta"):
        raise RuntimeError(f"{src} carries no momenta")
    step0, resumed, e_ref_row = 0, False, None
    if restart_path.is_file():
        z = np.load(restart_path)
        if int(z["n_atoms"]) != n:
            raise RuntimeError("restart file does not belong to this structure")
        atoms.set_cell(z["cell"], scale_atoms=False)
        atoms.set_positions(z["positions"])
        atoms.set_momenta(z["momenta"])
        step0, resumed = int(z["step"]), True
        t_before = float(z["t_before_rescale"])
        scale = float(z["velocity_scale_factor"])
        wall_prev = float(z["wall_seconds"])
    else:
        t_before = float(atoms.get_temperature())
        scale = math.sqrt(args.temperature_K / t_before)
        atoms.set_momenta(atoms.get_momenta() * scale)
        wall_prev = 0.0
        for old in (log_path, frames_path):
            if Path(old).exists():
                Path(old).unlink()
    atoms.calc = calc
    rows = kept_rows(log_path, step0) if resumed else []
    rows = [{k: (v if k == "stage" else float(v)) for k, v in r.items()} for r in rows]
    for r in rows:
        r["step"] = int(r["step"])
    seen_log = {r["step"] for r in rows}
    seen_frame = truncate_frames(frames_path, step0) if resumed else set()
    print("=" * 110)
    print(f"{rid}  N={n}  NVE {args.nve_ps} ps = {n_total} steps (dt {dt} fs, {args.dtype})"
          + (f"  RESUMING at step {step0}" if resumed else f"  rescale {t_before:.3f} K -> "
             f"{float(atoms.get_temperature()):.3f} K (factor {scale:.6f})"), flush=True)

    t_start = time.time()
    state = {"last_print": step0, "last_ckpt": step0}

    def save_checkpoint(step):
        tmp = out / "restart_tmp.npz"
        np.savez(tmp, positions=atoms.get_positions(), momenta=atoms.get_momenta(), cell=atoms.cell.array,
                 step=step, n_atoms=n, t_before_rescale=t_before, velocity_scale_factor=scale,
                 wall_seconds=wall_prev + time.time() - t_start)
        os.replace(tmp, restart_path)
        D.write_log(log_path, rows)

    def observe():
        step = step0 + dyn.nsteps
        if step % args.log_every == 0 and step not in seen_log:
            rows.append(D.sample(atoms, "NVE", step, dt))
            seen_log.add(step)
        if step % args.frame_every == 0 and step not in seen_frame:
            fr = atoms.copy()
            fr.info["step"] = int(step)
            write(str(frames_path), fr, format="extxyz", append=True)
            seen_frame.add(step)
        if step - state["last_print"] >= args.print_every and rows:
            r = rows[-1]
            e0 = rows[0]["total_energy_per_atom_eV"]
            wall = wall_prev + time.time() - t_start
            done = step - step0
            rate = done / max(time.time() - t_start, 1e-9)
            eta_h = (n_total - step) / max(rate, 1e-9) / 3600.0
            print(f"  [{rid}] step {step:8d}/{n_total}  t={step * dt / 1000:7.2f} ps  T={r['temperature_K']:7.2f} K  "
                  f"drift={1000 * (r['total_energy_per_atom_eV'] - e0):+.5f} meV/atom  {rate:5.1f} steps/s  "
                  f"ETA {eta_h:5.2f} h", flush=True)
            state["last_print"] = step
        if step - state["last_ckpt"] >= args.checkpoint_every:
            save_checkpoint(step)
            state["last_ckpt"] = step
        if args.stop_after_steps and step >= args.stop_after_steps:
            raise SystemExit(f"TEST STOP at step {step} (no checkpoint written here: simulates a kill)")

    dyn = VelocityVerlet(atoms, timestep=dt * units.fs)
    observe()                                                    # the starting step (a no-op when resuming)
    dyn.attach(observe, interval=1)
    dyn.run(n_total - step0)
    wall = wall_prev + time.time() - t_start

    # ---- summary
    write_log = D.write_log
    write_log(log_path, rows)
    e = np.array([r["total_energy_per_atom_eV"] for r in rows])
    t = np.array([r["time_ps"] for r in rows])
    temp = np.array([r["temperature_K"] for r in rows])
    drift = (e - e[0]) * 1000.0
    max_abs = float(np.max(np.abs(drift)))
    slope, intercept = np.polyfit(t - t[0], drift, 1)
    resid = drift - (slope * (t - t[0]) + intercept)
    se = float(np.std(resid, ddof=2) / (np.std(t - t[0]) * math.sqrt(len(t))))
    first10 = drift[t - t[0] <= 10.0 + 1e-9]
    win = np.array_split(np.arange(len(drift)), 10)                      # ten equal windows (50 ps for 500 ps)
    summary = {
        "record_id": rid, "n_atoms": n, "dtype": args.dtype, "dt_fs": dt, "nve_ps": args.nve_ps, "n_steps": n_total,
        "start_state": str(src), "temperature_before_rescale_K": t_before, "velocity_scale_factor": scale,
        "log_every_steps": args.log_every, "frame_every_steps": args.frame_every, "n_log_samples": len(rows),
        "walltime_seconds": wall, "steps_per_second": n_total / max(wall, 1e-9), "resumed": bool(resumed),
        "final_drift_meV_per_atom": float(drift[-1]), "max_abs_drift_meV_per_atom": max_abs,
        "peak_to_peak_meV_per_atom": float(drift.max() - drift.min()),
        "rms_fluctuation_about_trend_meV_per_atom": float(np.std(resid)),
        "linear_drift_meV_per_atom_per_ps": float(slope), "linear_drift_standard_error": se,
        "linear_drift_meV_per_atom_per_ns": float(slope * 1000.0),
        "first_10ps_max_abs_drift_meV_per_atom": float(np.max(np.abs(first10))),
        "window_means_meV_per_atom": [float(drift[w].mean()) for w in win],
        "nve_mean_temperature_K": float(temp.mean()), "nve_min_temperature_K": float(temp.min()),
        "nve_max_temperature_K": float(temp.max()),
        "energy_conserving": bool(max_abs <= args.conserve_limit_meV),
        "conservation_limit_meV_per_atom": args.conserve_limit_meV, "provenance": meta,
    }
    (out / "long_nve_summary.json").write_text(json.dumps(summary, indent=2))
    write(str(out / "nve_final_structure_tmp.extxyz"), atoms.copy(), format="extxyz")
    os.replace(out / "nve_final_structure_tmp.extxyz", final_file)            # completion marker, written last
    if restart_path.exists():
        restart_path.unlink()
    print(f"  DONE {rid}: max |drift| {max_abs:.5f} meV/atom over {args.nve_ps:g} ps (first 10 ps "
          f"{summary['first_10ps_max_abs_drift_meV_per_atom']:.5f}), slope {slope * 1000:+.4f} +- {se * 1000:.4f} "
          f"meV/atom/ns, {summary['steps_per_second']:.1f} steps/s, "
          f"{'conserving' if summary['energy_conserving'] else 'NOT CONSERVING'}", flush=True)
    return "done"


def main():
    args = parse_args()
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_local_moe_md as D

    HET = Path(args.het)
    PROJ = HET / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    paths = {
        "common": args.common_checkpoint or PROJ / "runs/common_holdout40_ew10000_v1/best_checkpoint.pt",
        "routing": args.routing_checkpoint or PROJ / "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt",
        "c1b": args.c1b_checkpoint or PROJ / "runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt",
        "soap": args.torchsoap_module or REPO_ROOT / 'src/local_moe/torchsoap_autograd_module_periodic_regnorm_nl.py',
    }
    paths = {k: Path(v).expanduser().resolve() for k, v in paths.items()}
    md_root = (args.md_root or PROJ / "md_holdout40_v1").expanduser().resolve()
    out_root = (args.out_root or PROJ / "md_long500ps_v1").expanduser().resolve()
    for p in paths.values():
        if not p.is_file():
            raise SystemExit(f"MISSING FILE: {p}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device visible")
        print(f"GPU: {torch.cuda.get_device_name()}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
              flush=True)

    if args.manifest:
        if args.slot is None:
            raise SystemExit("--manifest needs --slot")
        with open(args.manifest, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["slot"]) == args.slot]
        rows.sort(key=lambda r: int(r.get("order", 0) or 0))
        wanted = [r["record_id"] for r in rows]
    else:
        wanted = [x.strip() for x in args.records.split(",") if x.strip()]
    if not wanted:
        print("nothing assigned to this slot; exiting")
        return
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoints": {k: str(v) for k, v in paths.items()}, "md_root": str(md_root),
            "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"), "torch": torch.__version__,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import ase
        meta["ase"] = ase.__version__
    except Exception:
        pass
    calc = D.make_calculator(args, paths)
    print(f"records for this process ({len(wanted)}): {wanted}", flush=True)
    status = {}
    for rid in wanted:
        try:
            status[rid] = run_one(rid, calc, args, md_root, out_root, meta, D)
        except SystemExit:
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            status[rid] = f"FAILED: {exc}"
            print(f"FAILED {rid}: {exc}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("\nSTATUS", json.dumps(status, indent=1))
    if any(str(v).startswith("FAILED") for v in status.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
