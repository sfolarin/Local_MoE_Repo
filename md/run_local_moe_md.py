#!/usr/bin/env python3
"""
run_local_moe_md.py

NVE-stability MD for the holdout-40 Local-MoE potential, following the protocol of the earlier
37-structure run (md_relaxed37_npt5_nve10_v1) case by case:

  1. fixed-cell relaxation with the model      FIRE to fmax 0.05 (<= 2000 steps, maxstep 0.05), then
                                                BFGS to fmax 0.01 (<= 1500 steps, maxstep 0.03);
                                                LBFGS instead of BFGS above --bfgs-max-atoms atoms
                                                (dense BFGS needs an eigendecomposition per step)
  2. NPT, ASE NPTBerendsen, isotropic          5 ps, dt 0.5 fs, 300 K, 1 bar, taut 100 fs, taup 1000 fs,
                                                compressibility 4.57e-5 / bar, Maxwell-Boltzmann start
  3. exact velocity rescale to 300 K
  4. NVE, ASE VelocityVerlet                   10 ps, dt 0.5 fs, logged every 20 steps (1001 samples)

Drift is 1000 * (E_total/N - E_total/N at the first NVE sample) in meV/atom; a structure counts as
energy-conserving if max |drift| <= 0.1 meV/atom (the criterion of the earlier figure).

Per case it writes, in <out-root>/case_<record_id>/, the same files as before:
  npt_5ps_log.csv  npt_5ps_frames.extxyz  npt_final_structure.extxyz
  nve_10ps_log.csv nve_10ps_frames.extxyz nve_final_structure.extxyz  final_protocol_summary.json
plus input_structure.extxyz, regions.npz (phase, interface_center from the dataset) and relax logs.
A case whose nve_final_structure.extxyz exists is skipped, so a job can simply be resubmitted.

Which structures
----------------
  --manifest md_manifest.csv --job J --slot S    rows of the manifest assigned to that job slot (in order)
  --records external_0090,own_0136               explicit list
  --check external_0090                          finite-difference check of forces and stress, then exit
  --parity external_0090                         energy / forces against the training code's own predict_energy_forces

Run one process per GPU; see run_md_job.slurm.
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
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HET_DEFAULT = (_os.environ.get("MOE_HET", "."))
REPO_DEFAULT = str(REPO_ROOT)
GPA_PER_EV_A3 = 160.21766208
LOG_COLUMNS = ["stage", "step", "time_fs", "time_ps", "temperature_K", "pressure_bar", "pressure_GPa",
               "stress_xx_GPa", "stress_yy_GPa", "stress_zz_GPa", "stress_yz_GPa", "stress_xz_GPa",
               "stress_xy_GPa", "potential_energy_eV", "kinetic_energy_eV", "total_energy_eV",
               "total_energy_per_atom_eV", "volume_A3", "cell_a_A", "cell_b_A", "cell_c_A", "cell_det_A3"]


def import_file(path, name):
    path = Path(path).expanduser().resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--het", default=HET_DEFAULT)
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--trainer-script", type=Path, default=None, help="default: <repo>/training/train_c1b_holdout40.py")
    p.add_argument("--stage-root", type=Path, default=None)
    p.add_argument("--common-checkpoint", type=Path, default=None)
    p.add_argument("--routing-checkpoint", type=Path, default=None)
    p.add_argument("--c1b-checkpoint", type=Path, default=None)
    p.add_argument("--torchsoap-module", type=Path, default=None, help="the neighbour-list descriptor module")
    p.add_argument("--out-root", type=Path, default=None, help="default: <proj>/md_holdout40_v1")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--job", type=int, default=None)
    p.add_argument("--slot", type=int, default=None)
    p.add_argument("--records", default="")
    p.add_argument("--check", default="", help="record id: finite-difference check of forces and stress, then exit")
    p.add_argument("--parity", default="", help="record id: compare this calculator with the trainer's own "
                                                "predict_energy_forces on that record, then exit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--pressure-bar", type=float, default=1.0)
    p.add_argument("--dt-fs", type=float, default=0.5)
    p.add_argument("--npt-ps", type=float, default=5.0)
    p.add_argument("--nve-ps", type=float, default=10.0)
    p.add_argument("--taut-fs", type=float, default=100.0)
    p.add_argument("--taup-fs", type=float, default=1000.0)
    p.add_argument("--compressibility-per-bar", type=float, default=4.57e-5)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--frame-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--fire-fmax", type=float, default=0.05)
    p.add_argument("--fire-steps", type=int, default=2000)
    p.add_argument("--bfgs-fmax", type=float, default=0.01)
    p.add_argument("--bfgs-steps", type=int, default=1500)
    p.add_argument("--bfgs-max-atoms", type=int, default=1500)
    p.add_argument("--conserve-limit-meV", type=float, default=0.1)
    return p.parse_args()


# --------------------------------------------------------------------------------- helpers
def record_number(name):
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else 0


def atoms_from_record(rec):
    from ase import Atoms
    return Atoms(numbers=np.asarray(rec.atomic_numbers, dtype=int), positions=np.asarray(rec.positions_np, float),
                 cell=np.asarray(rec.cell_np, float), pbc=True)


def make_calculator(args, paths):
    from local_moe_ase import LocalMoECalculator
    return LocalMoECalculator(common_checkpoint=paths["common"], routing_checkpoint=paths["routing"],
                              c1b_checkpoint=paths["c1b"], torchsoap_module=paths["soap"],
                              device=args.device, dtype=args.dtype)


def sample(atoms, stage, step, dt_fs):
    stress = np.asarray(atoms.get_stress(voigt=True, include_ideal_gas=True), dtype=float)
    pot = float(atoms.get_potential_energy())
    kin = float(atoms.get_kinetic_energy())
    cell = atoms.cell.array
    p_gpa = -float(np.mean(stress[:3])) * GPA_PER_EV_A3
    row = {
        "stage": stage, "step": int(step), "time_fs": step * dt_fs, "time_ps": step * dt_fs / 1000.0,
        "temperature_K": float(atoms.get_temperature()), "pressure_bar": p_gpa * 1.0e4, "pressure_GPa": p_gpa,
        "stress_xx_GPa": stress[0] * GPA_PER_EV_A3, "stress_yy_GPa": stress[1] * GPA_PER_EV_A3,
        "stress_zz_GPa": stress[2] * GPA_PER_EV_A3, "stress_yz_GPa": stress[3] * GPA_PER_EV_A3,
        "stress_xz_GPa": stress[4] * GPA_PER_EV_A3, "stress_xy_GPa": stress[5] * GPA_PER_EV_A3,
        "potential_energy_eV": pot, "kinetic_energy_eV": kin, "total_energy_eV": pot + kin,
        "total_energy_per_atom_eV": (pot + kin) / len(atoms), "volume_A3": float(atoms.get_volume()),
        "cell_a_A": float(np.linalg.norm(cell[0])), "cell_b_A": float(np.linalg.norm(cell[1])),
        "cell_c_A": float(np.linalg.norm(cell[2])), "cell_det_A3": float(abs(np.linalg.det(cell))),
    }
    if not all(math.isfinite(v) for k, v in row.items() if k != "stage"):
        raise RuntimeError(f"non-finite value in {stage} step {step}: {row}")
    return row


def write_log(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def run_stage(atoms, dyn, stage, n_steps, dt_fs, log_every, frame_every, frames_path, label):
    """Run `dyn` for n_steps, sampling every log_every steps and writing frames every frame_every steps."""
    from ase.io import write
    rows, seen = [], set()
    if Path(frames_path).exists():
        Path(frames_path).unlink()

    def observe():
        step = dyn.nsteps
        if step in seen:
            return
        if step % log_every == 0:
            rows.append(sample(atoms, stage, step, dt_fs))
        if step % frame_every == 0:
            write(str(frames_path), atoms.copy(), format="extxyz", append=True)
        seen.add(step)
        if step and step % 2000 == 0:
            r = rows[-1] if rows else None
            msg = f"  [{label}] step {step:6d}/{n_steps}"
            if r and r["step"] == step:
                msg += (f"  T={r['temperature_K']:7.2f} K  P={r['pressure_GPa']:+8.4f} GPa  "
                        f"E/N={r['total_energy_per_atom_eV']:.9f} eV")
            print(msg, flush=True)

    observe()                                  # step 0, before any dynamics
    dyn.attach(observe, interval=1)
    t0 = time.time()
    dyn.run(n_steps)
    wall = time.time() - t0
    return rows, wall


# --------------------------------------------------------------------------------- one case
def run_case(rec, calc, args, out_root, meta, region_masks_fn=None):
    from ase import units
    from ase.io import write
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase.optimize import BFGS, FIRE, LBFGS

    rid = str(rec.source_file)
    out = out_root / f"case_{rid}"
    out.mkdir(parents=True, exist_ok=True)
    final_file = out / "nve_final_structure.extxyz"
    if final_file.is_file() and final_file.stat().st_size > 0:
        print(f"SKIP COMPLETE: {rid}", flush=True)
        return "skipped"

    atoms = atoms_from_record(rec)
    n = len(atoms)
    write(str(out / "input_structure.extxyz"), atoms, format="extxyz")
    # Region masks exactly as the trainer defines them for the test metrics (atom order is preserved in MD).
    masks = {"all": np.ones(n, dtype=bool)}
    has_regions = False
    if region_masks_fn is not None:
        try:
            got = region_masks_fn(rec)
            masks = {k: np.asarray(got[k], dtype=bool) for k in ("all", "interface", "phase1_bulk", "phase2_bulk")}
            has_regions = bool(masks["interface"].any() and masks["phase1_bulk"].any() and masks["phase2_bulk"].any())
        except Exception as exc:                       # records without region metadata
            print(f"  (no region masks: {exc})", flush=True)
    np.savez(out / "regions.npz", phase=np.asarray(rec.phase), interface_center=np.asarray(rec.interface_center),
             numbers=np.asarray(rec.atomic_numbers), has_regions=np.asarray(has_regions),
             **{f"mask_{k}": v for k, v in masks.items()})
    atoms.calc = calc
    dt = args.dt_fs
    print("=" * 110)
    print(f"{rid}  N={n}  phase pair={getattr(rec, 'phase_pair', '')}  dtype={args.dtype}  dt={dt} fs", flush=True)

    # ------------------------------------------------ 1. fixed-cell relaxation
    cell0 = atoms.cell.array.copy()
    t0 = time.time()
    e_in = float(atoms.get_potential_energy())
    f_in = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    fire = FIRE(atoms, logfile=str(out / "relax_fire.log"), maxstep=0.05)
    fire.run(fmax=args.fire_fmax, steps=args.fire_steps)
    polish_name = "BFGS" if n <= args.bfgs_max_atoms else "LBFGS"
    polish = (BFGS if n <= args.bfgs_max_atoms else LBFGS)(atoms, logfile=str(out / "relax_polish.log"), maxstep=0.03)
    polish.run(fmax=args.bfgs_fmax, steps=args.bfgs_steps)
    f_out = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    if float(np.max(np.abs(atoms.cell.array - cell0))) > 1e-10:
        raise RuntimeError("cell changed during fixed-cell relaxation")
    relax = {"optimizers": ["FIRE", polish_name], "fire_steps": int(fire.nsteps), "polish_steps": int(polish.nsteps),
             "Fmax_initial_eV_A": f_in, "Fmax_final_eV_A": f_out, "E_initial_eV": e_in,
             "E_final_eV": float(atoms.get_potential_energy()), "walltime_seconds": time.time() - t0}
    print(f"  relaxed: FIRE {fire.nsteps} + {polish_name} {polish.nsteps} steps, Fmax {f_in:.3f} -> {f_out:.4f} eV/A",
          flush=True)
    write(str(out / "relaxed_structure.extxyz"), atoms.copy(), format="extxyz")

    # ------------------------------------------------ 2. NPT
    seed = args.seed + record_number(rid)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=np.random.RandomState(seed))
    p0 = -float(np.mean(atoms.get_stress(voigt=True, include_ideal_gas=False)[:3])) * GPA_PER_EV_A3
    v0 = float(atoms.get_volume())
    npt = NPTBerendsen(atoms, timestep=dt * units.fs, temperature_K=args.temperature_K,
                       taut=args.taut_fs * units.fs, pressure_au=args.pressure_bar * units.bar,
                       taup=args.taup_fs * units.fs,
                       compressibility_au=args.compressibility_per_bar / units.bar)
    n_npt = int(round(args.npt_ps * 1000.0 / dt))
    print(f"  NPT {args.npt_ps} ps ({n_npt} steps), initial pressure {p0:+.4f} GPa", flush=True)
    npt_rows, npt_wall = run_stage(atoms, npt, "NPT", n_npt, dt, args.log_every, args.frame_every,
                                   out / "npt_5ps_frames.extxyz", "NPT")
    write_log(out / "npt_5ps_log.csv", npt_rows)
    write(str(out / "npt_final_structure.extxyz"), atoms.copy(), format="extxyz")

    # ------------------------------------------------ 3. exact rescale, 4. NVE
    t_before = float(atoms.get_temperature())
    scale = math.sqrt(args.temperature_K / t_before)
    atoms.set_momenta(atoms.get_momenta() * scale)
    t_after = float(atoms.get_temperature())
    nve = VelocityVerlet(atoms, timestep=dt * units.fs)
    n_nve = int(round(args.nve_ps * 1000.0 / dt))
    print(f"  rescale {t_before:.3f} K -> {t_after:.3f} K (factor {scale:.6f}); NVE {args.nve_ps} ps ({n_nve} steps)",
          flush=True)
    nve_rows, nve_wall = run_stage(atoms, nve, "NVE", n_nve, dt, args.log_every, args.frame_every,
                                   out / "nve_10ps_frames.extxyz", "NVE")
    write_log(out / "nve_10ps_log.csv", nve_rows)
    write(str(out / "nve_final_structure_tmp.extxyz"), atoms.copy(), format="extxyz")

    # ------------------------------------------------ summary
    e = np.array([r["total_energy_per_atom_eV"] for r in nve_rows])
    t = np.array([r["time_ps"] for r in nve_rows])
    drift = (e - e[0]) * 1000.0
    temps = np.array([r["temperature_K"] for r in nve_rows])
    npt_t = np.array([r["temperature_K"] for r in npt_rows])
    npt_p = np.array([r["pressure_GPa"] for r in npt_rows])
    max_abs = float(np.max(np.abs(drift)))
    summary = {
        "record_id": rid, "phase_pair": str(getattr(rec, "phase_pair", "")), "split": "test", "n_atoms": n,
        "selected_model": "C1b frozen-routing conservative Local-MoE (holdout-40), neighbour-list TorchSOAP",
        "dtype": args.dtype, "seed": seed, "relaxation": relax,
        "initial_pressure_GPa": p0, "npt_initial_volume_A3": v0, "npt_final_volume_A3": float(atoms.get_volume()),
        "npt_volume_change_percent": 100.0 * (float(atoms.get_volume()) / v0 - 1.0),
        "npt_mean_temperature_K": float(npt_t.mean()), "npt_mean_pressure_GPa": float(npt_p.mean()),
        "npt_final_temperature_K": t_before, "npt_final_pressure_GPa": float(npt_p[-1]),
        "final_npt_cell_A": atoms.cell.array.tolist(),
        "npt_walltime_seconds": npt_wall, "npt_steps_per_second": n_npt / max(npt_wall, 1e-9),
        "temperature_before_velocity_rescale_K": t_before, "temperature_after_velocity_rescale_K": t_after,
        "velocity_scale_factor": scale,
        "nve_walltime_seconds": nve_wall, "nve_steps_per_second": n_nve / max(nve_wall, 1e-9),
        "nve_mean_temperature_K": float(temps.mean()), "nve_min_temperature_K": float(temps.min()),
        "nve_max_temperature_K": float(temps.max()), "nve_last50_mean_temperature_K": float(temps[-50:].mean()),
        "nve_final_energy_drift_meV_per_atom": float(drift[-1]),
        "nve_max_abs_energy_drift_meV_per_atom": max_abs,
        "nve_peak_to_peak_energy_meV_per_atom": float(drift.max() - drift.min()),
        "nve_linear_drift_meV_per_atom_per_ps": float(np.polyfit(t - t[0], drift, 1)[0]),
        "energy_conserving": bool(max_abs <= args.conserve_limit_meV),
        "protocol": {
            "temperature_K": args.temperature_K, "pressure_bar": args.pressure_bar, "dt_fs": dt,
            "npt_integrator": "ASE NPTBerendsen", "npt_cell_scaling": "isotropic default NPTBerendsen",
            "npt_ps": args.npt_ps, "taut_fs": args.taut_fs, "taup_fs": args.taup_fs,
            "compressibility_per_bar": args.compressibility_per_bar,
            "post_npt_velocity_rescale": "exact to target temperature",
            "nve_integrator": "ASE VelocityVerlet", "nve_ps": args.nve_ps, "log_every_steps": args.log_every,
            "frame_every_steps": args.frame_every, "conservation_limit_meV_per_atom": args.conserve_limit_meV},
        "provenance": meta,
    }
    (out / "final_protocol_summary.json").write_text(json.dumps(summary, indent=2))
    # the completion marker is written last, so an interrupted case is rerun
    os.replace(out / "nve_final_structure_tmp.extxyz", final_file)
    print(f"  DONE {rid}: max |drift| {max_abs:.5f} meV/atom, final {drift[-1]:+.5f}, "
          f"{'conserving' if summary['energy_conserving'] else 'NOT CONSERVING'}; "
          f"NPT {summary['npt_steps_per_second']:.1f} and NVE {summary['nve_steps_per_second']:.1f} steps/s", flush=True)
    return "done"


# --------------------------------------------------------------------------------- check mode
def run_check(rec, calc, args):
    """Central finite differences of the model's own energy: forces (3 atoms) and all six stress components."""
    from ase import Atoms
    atoms = atoms_from_record(rec)
    atoms.calc = calc
    n = len(atoms)
    f_an = atoms.get_forces()
    s_an = atoms.get_stress(voigt=False)
    h = 1.0e-4
    print(f"CHECK {rec.source_file}: N={n}, dtype={args.dtype}, energy {atoms.get_potential_energy():.6f} eV")
    worst_f = 0.0
    for a in sorted({0, n // 2, n - 1}):
        for k in range(3):
            def energy_at(d):
                q = atoms.copy()
                q.positions[a, k] += d
                q.calc = calc
                return q.get_potential_energy()
            f_fd = -(energy_at(+h) - energy_at(-h)) / (2 * h)
            worst_f = max(worst_f, abs(f_fd - f_an[a, k]))
    vol = atoms.get_volume()
    worst_s = 0.0
    for (i, j) in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        eps = np.zeros((3, 3))
        eps[i, j] += h
        eps[j, i] += h if i != j else 0.0
        energies = []
        for sgn in (+1, -1):
            q = atoms.copy()
            q.set_cell(atoms.cell.array @ (np.eye(3) + sgn * eps).T, scale_atoms=True)
            q.calc = calc
            energies.append(q.get_potential_energy())
        d = (energies[0] - energies[1]) / (2 * h)           # = V * sum(sigma_ab * eps_ab) / h
        sigma_fd = d / vol / (1.0 if i == j else 2.0)
        diff = abs(sigma_fd - s_an[i, j])
        worst_s = max(worst_s, diff)
        print(f"  stress[{i}{j}]  analytic {s_an[i, j]:+.8e}  FD {sigma_fd:+.8e}  eV/A^3   diff {diff:.2e}")
    ok_f = worst_f < 1e-5
    ok_s = worst_s < 2e-6
    print(f"  forces: worst |FD - analytic| = {worst_f:.3e} eV/A  ({'PASS' if ok_f else 'FAIL'}, limit 1e-5)")
    print(f"  stress: worst |FD - analytic| = {worst_s:.3e} eV/A^3 = {worst_s * GPA_PER_EV_A3:.3e} GPa  "
          f"({'PASS' if ok_s else 'FAIL'}, limit 2e-6 eV/A^3)")
    if args.dtype != "float64":
        print("  note: finite differences are only meaningful with --dtype float64")
    return ok_f and ok_s


def run_parity(rec, calc, args, trainer, paths):
    """Energy and forces of this calculator against the trainer's predict_energy_forces (float32 networks)."""
    import torch
    device = torch.device(args.device)
    common_model, cm, cs, e_ref, _ = trainer.load_common_checkpoint(paths["common"], device)
    routing_model, rm, rs, _ = trainer.load_routing_checkpoint(paths["routing"], device)
    bck = torch.load(paths["c1b"], map_location="cpu", weights_only=False)
    residual = trainer.MLP(input_dim=int(bck["input_dim"]), hidden_dims=[int(x) for x in bck["residual_hidden_dims"]],
                           output_dim=7, dropout=0.0).to(device)
    residual.load_state_dict(bck["residual_model_state_dict"])
    residual.eval()
    soap = trainer.import_torchsoap(paths["soap"])
    descriptor = soap.TorchSOAPDescriptor(soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    for prm in descriptor.parameters():
        prm.requires_grad_(False)
    with torch.enable_grad():
        terms, forces = trainer.predict_energy_forces(
            record=rec, descriptor=descriptor, common_model=common_model, common_mean=cm, common_std=cs,
            routing_model=routing_model, routing_mean=rm, routing_std=rs, residual_model=residual,
            residual_mean=cm, residual_std=cs, e_ref=e_ref, device=device, create_graph=False)
    e_ref_trainer = float(terms["total_energy"].detach().cpu())
    f_trainer = forces.detach().cpu().numpy().astype(np.float64)
    atoms = atoms_from_record(rec)
    atoms.calc = calc
    e_calc, f_calc = float(atoms.get_potential_energy()), atoms.get_forces()
    n = len(atoms)
    de = (e_calc - e_ref_trainer) / n
    df = np.abs(f_calc - f_trainer)
    print(f"PARITY {rec.source_file}: N={n}, calculator dtype={args.dtype}")
    print(f"  energy  calculator {e_calc:.6f} eV   trainer {e_ref_trainer:.6f} eV   diff {1000 * de:+.5f} meV/atom")
    print(f"  forces  max |diff| {df.max():.3e} eV/A   rms {np.sqrt((df ** 2).mean()):.3e} eV/A   "
          f"(trainer |F| max {np.abs(f_trainer).max():.3f})")
    ok = abs(de) < 1.0e-4 and df.max() < 2.0e-3
    print("  " + ("PASS (agreement at the level of float32 rounding)" if ok else
                  "FAIL: larger than float32 rounding; do not run MD"))
    return ok


# --------------------------------------------------------------------------------- main
def main():
    args = parse_args()
    import torch
    HET = Path(args.het)
    PROJ = HET / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    script = args.trainer_script or REPO_ROOT / 'training/train_c1b_holdout40.py'
    paths = {
        "common": args.common_checkpoint or PROJ / "runs/common_holdout40_ew10000_v1/best_checkpoint.pt",
        "routing": args.routing_checkpoint or PROJ / "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt",
        "c1b": args.c1b_checkpoint or PROJ / "runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt",
        "soap": args.torchsoap_module or REPO_ROOT / 'src/local_moe/torchsoap_autograd_module_periodic_regnorm_nl.py',
    }
    paths = {k: Path(v).expanduser().resolve() for k, v in paths.items()}
    stage = (args.stage_root or PROJ / "data/expanded_holdout40_v1_c1b_stage").expanduser().resolve()
    out_root = (args.out_root or PROJ / "md_holdout40_v1").expanduser().resolve()
    for p in [script, *paths.values()]:
        if not Path(p).is_file():
            raise SystemExit(f"MISSING FILE: {p}")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("no CUDA device visible")
        # one GPU per process (srun --gpus-per-task=1), so "cuda" is that GPU
        print(f"GPU: {torch.cuda.get_device_name()}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
              flush=True)

    trainer = import_file(script, "md_c1b_trainer")
    test = {str(r.source_file): r for r in trainer.load_records(stage)["test"]}
    print(f"test records available: {len(test)}", flush=True)

    # ---- which records
    if args.check or args.parity:
        wanted = [args.check or args.parity]
    elif args.manifest:
        if args.job is None or args.slot is None:
            raise SystemExit("--manifest needs --job and --slot")
        with open(args.manifest, newline="") as f:
            wanted = [r["record_id"] for r in csv.DictReader(f)
                      if int(r["job"]) == args.job and int(r["slot"]) == args.slot]
    else:
        wanted = [x.strip() for x in args.records.split(",") if x.strip()]
    missing = [w for w in wanted if w not in test]
    if missing:
        raise SystemExit(f"records not in the test split: {missing}")
    if not wanted:
        print("nothing assigned to this job/slot; exiting")
        return

    calc = make_calculator(args, paths)
    if args.check:
        sys.exit(0 if run_check(test[wanted[0]], calc, args) else 1)
    if args.parity:
        sys.exit(0 if run_parity(test[wanted[0]], calc, args, trainer, paths) else 1)

    out_root.mkdir(parents=True, exist_ok=True)
    meta = {"checkpoints": {k: str(v) for k, v in paths.items()}, "trainer_script": str(script),
            "host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import ase
        meta["ase"] = ase.__version__
    except Exception:
        pass
    print(f"records for this process ({len(wanted)}): {wanted}", flush=True)
    status = {}
    for rid in wanted:
        try:
            status[rid] = run_case(test[rid], calc, args, out_root, meta,
                                   region_masks_fn=getattr(trainer, "region_masks", None))
        except Exception as exc:                       # keep going: one bad case must not kill the job
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
