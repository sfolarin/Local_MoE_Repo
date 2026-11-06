#!/usr/bin/env python3
"""
test_rotational_invariance_local_moe.py

Rotation / translation / conservativity tests of the FINAL holdout-40 Local-MoE
checkpoints, with the descriptor file you point at.  READ-ONLY: no checkpoint,
trainer or descriptor file is modified, nothing is retrained, no basis is
normalised.  The descriptor is imported from --torchsoap-module exactly as the
trainer / MD calculator import it.

Three evaluation routes (same checkpoints, same descriptor file):

  trainer32  the trainer's own predict_energy_forces() (float32 nets, float64
             positions/energy sums, descriptor float64 -> float32 features).
             This is the code path that produced the reported accuracy numbers
             and the stored parity-export predictions.
  calc32     LocalMoECalculator, float32 nets  (MD calculator, benchmark precision)
  calc64     LocalMoECalculator, float64 nets  (MD calculator, float64 protocol;
             also used for the finite-difference and Hessian-symmetry checks)

What is reported (per route, per structure)
-------------------------------------------
  noise floor   |E(x)-E(x)|, max|F(x)-F(x)| for two evaluations of the SAME input
  rotations     proper rotations R (pos' = pos @ R.T, cell' = cell @ R.T):
                  dE   = (E(Rx) - E(x)) / N                     [meV/atom]
                  dF   = max_ia |F(Rx) - F(x) @ R.T|            [eV/A]  (covariance)
                  dF_rms / F_rms                                 [-]
                  dS   = max |sigma(Rx) - R sigma(x) R^T|       [GPa]   (calc routes)
  translation   rigid shift (no wrapping): dE, max|dF|, |sum F|
  conservativity (calc64): central-difference check F = -dE/dx and Hessian
                  symmetry dF_a/dx_b = dF_b/dx_a, on the unrotated and a rotated copy.
  stored predictions: identity evaluation vs the parity-export npz (real test records)

Conservativity and rotational invariance are DIFFERENT properties and are
reported separately.  Conservativity: F is the gradient of one scalar E.
Rotational invariance: E(Rx) = E(x) and F(Rx) = R F(x).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_torchsoap_rotation as D  # noqa: E402  (descriptor-test helpers: rotations, structures, control basis)
from local_moe_ase import LocalMoECalculator, import_file as _import_file  # noqa: E402

EV_A3_TO_GPA = 160.21766208


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def voigt_to_mat(s):
    s = np.asarray(s, float)
    return np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])


# ---------------------------------------------------------------------------
# trainer route (copied from benchmark_cost_holdout40.py: build_local, record_from_atoms, make_full_fn)
# ---------------------------------------------------------------------------
def record_from_atoms(trainer, atoms, name):
    n = len(atoms)
    z = np.asarray(atoms.numbers).astype(np.int16)
    symbols, masses = trainer.build_symbols_masses(z)
    return trainer.StructureRecord(
        source_file=name, phase_pair="", split="bench", n_atoms=n, feature_file=Path(""), source_shard=Path(""),
        atomic_numbers=z, positions_np=np.array(atoms.positions, dtype=np.float64),
        cell_np=np.array(atoms.cell, dtype=np.float64),
        phase=np.zeros(n, dtype=np.int16), interface_center=np.zeros(n, dtype=np.int16),
        target_energy_total_eV=0.0, target_forces_np=np.zeros((n, 3)), symbols=symbols, masses=masses)


def build_trainer_parts(trainer, paths, device, descriptor_override=None):
    common_model, common_mean, common_std, e_ref, _ = trainer.load_common_checkpoint(paths["common"], device)
    soap = trainer.import_torchsoap(paths["soap"])
    descriptor = soap.TorchSOAPDescriptor(
        soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    if descriptor_override is not None:
        descriptor = descriptor_override(soap).to(device)
    for p in descriptor.parameters():
        p.requires_grad_(False)
    routing_model, routing_mean, routing_std, _ = trainer.load_routing_checkpoint(paths["routing"], device)
    bck = torch.load(paths["c1b"], map_location="cpu", weights_only=False)
    if [str(x) for x in bck["expert_names"]] != list(trainer.EXPERTS):
        raise SystemExit("C1b expert order does not match the trainer")
    residual_model = trainer.MLP(input_dim=int(bck["input_dim"]),
                                 hidden_dims=[int(x) for x in bck["residual_hidden_dims"]],
                                 output_dim=7, dropout=0.0).to(device)
    residual_model.load_state_dict(bck["residual_model_state_dict"])
    residual_model.eval()
    return dict(common_model=common_model, common_mean=common_mean, common_std=common_std, e_ref=e_ref,
                descriptor=descriptor, device=device, routing_model=routing_model, routing_mean=routing_mean,
                routing_std=routing_std, residual_model=residual_model, residual_mean=common_mean,
                residual_std=common_std)


def make_trainer_fn(trainer, parts):
    device = parts["device"]

    def fn(atoms):
        rec = record_from_atoms(trainer, atoms, "rot")
        with torch.enable_grad():
            terms, forces = trainer.predict_energy_forces(
                record=rec, descriptor=parts["descriptor"], common_model=parts["common_model"],
                common_mean=parts["common_mean"], common_std=parts["common_std"],
                routing_model=parts["routing_model"], routing_mean=parts["routing_mean"],
                routing_std=parts["routing_std"], residual_model=parts["residual_model"],
                residual_mean=parts["residual_mean"], residual_std=parts["residual_std"],
                e_ref=parts["e_ref"], device=device, create_graph=False)
        return {"E": float(terms["total_energy"].detach().cpu()), "F": forces.detach().cpu().numpy(), "S": None}
    return fn


def make_calc_fn(calc):
    from ase.calculators.calculator import all_changes

    def fn(atoms):
        calc.calculate(atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
        r = calc.results
        return {"E": float(r["energy"]), "F": np.asarray(r["forces"], float), "S": np.asarray(r["stress"], float)}
    return fn


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compare(ref, out, R, n_atoms):
    dE = (out["E"] - ref["E"]) / n_atoms * 1e3
    cov = ref["F"] @ R.T
    dF = out["F"] - cov
    rec = {"dE_meV_atom": float(dE), "dF_max": float(np.abs(dF).max()),
           "dF_rms": float(np.sqrt(np.mean(dF ** 2))), "F_rms": float(np.sqrt(np.mean(ref["F"] ** 2))),
           "sumF_rot": float(np.abs(out["F"].sum(axis=0)).max())}
    rec["dF_rel"] = rec["dF_rms"] / max(rec["F_rms"], 1e-300)
    if ref["S"] is not None and out["S"] is not None:
        s0 = voigt_to_mat(ref["S"])
        s1 = voigt_to_mat(out["S"])
        rec["dS_GPa"] = float(np.abs(s1 - R @ s0 @ R.T).max() * EV_A3_TO_GPA)
    return rec


def fd_and_hessian(fn, atoms, rng, h=1e-4, n_fd=6):
    """Finite-difference force check and Hessian-symmetry (curl) check, float64 route."""
    n = len(atoms)
    base = fn(atoms)
    pairs = [(int(rng.integers(n)), int(rng.integers(3))) for _ in range(n_fd)]
    worst = 0.0
    worst_half = 0.0   # same check with h/2: a truncation error O(h^2) drops by ~4
    for a, c in pairs:
        for hh in (h, h / 2):
            ap, am = atoms.copy(), atoms.copy()
            ap.positions[a, c] += hh
            am.positions[a, c] -= hh
            f_fd = -(fn(ap)["E"] - fn(am)["E"]) / (2 * hh)
            err = abs(f_fd - base["F"][a, c])
            if hh == h:
                worst = max(worst, err)
            else:
                worst_half = max(worst_half, err)
    # Hessian symmetry between atom 0 and its nearest neighbour (non-zero coupling)
    d = atoms.get_distances(0, list(range(1, n)), mic=True)
    b = 1 + int(np.argmin(d))
    asym, scale = 0.0, 0.0
    for ia, ib in [(0, 1), (1, 2), (2, 0)]:
        def dF(atom_shift, comp_shift, atom_force, comp_force):
            ap, am = atoms.copy(), atoms.copy()
            ap.positions[atom_shift, comp_shift] += h
            am.positions[atom_shift, comp_shift] -= h
            return (fn(ap)["F"][atom_force, comp_force] - fn(am)["F"][atom_force, comp_force]) / (2 * h)
        h_ab = -dF(b, ib, 0, ia)   # d(-F_0,ia)/d x_b,ib
        h_ba = -dF(0, ia, b, ib)   # d(-F_b,ib)/d x_0,ia
        asym = max(asym, abs(h_ab - h_ba))
        scale = max(scale, abs(h_ab), abs(h_ba))
    return {"fd_force_max_abs_err": worst, "fd_force_max_abs_err_half_h": worst_half, "hessian_asym_max": asym, "hessian_scale": scale,
            "fd_h": h, "n_fd": n_fd, "hessian_neighbour": b}


# ---------------------------------------------------------------------------
def pick_test_records(trainer, stage, k, max_atoms):
    recs = trainer.load_records(stage)["test"]
    recs = sorted([r for r in recs if r.n_atoms <= max_atoms], key=lambda r: (r.n_atoms, r.source_file))
    if not recs:
        return []
    idx = sorted(set(int(round(x)) for x in np.linspace(0, len(recs) - 1, min(k, len(recs)))))
    return [recs[i] for i in idx]


def atoms_from_record(rec):
    from ase import Atoms
    return Atoms(numbers=np.asarray(rec.atomic_numbers, dtype=int), positions=np.asarray(rec.positions_np, float),
                 cell=np.asarray(rec.cell_np, float), pbc=True)


def load_reference(ref_dir, rid):
    p = Path(ref_dir) / (str(rid).replace("/", "__") + ".npz")
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as d:
        return {k: np.asarray(d[k], float) for k in ("E_predicted_eV", "predicted_forces")}


def describe_cell(cell):
    c = np.asarray(cell, float)
    lens = np.linalg.norm(c, axis=1)
    cosang = [float(np.dot(c[(i + 1) % 3], c[(i + 2) % 3]) / (lens[(i + 1) % 3] * lens[(i + 2) % 3])) for i in range(3)]
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    diag = np.allclose(c - np.diag(np.diag(c)), 0, atol=1e-8)
    return (f"a,b,c=({lens[0]:.2f},{lens[1]:.2f},{lens[2]:.2f}) A  alpha,beta,gamma=({ang[0]:.1f},{ang[1]:.1f},{ang[2]:.1f})"
            f"  axis-aligned-orthorhombic={diag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proj", default=os.environ.get("PROJ"), help="project dir (env PROJ) holding runs/, data/, eval_parity_v1/")
    ap.add_argument("--repo", default=os.environ.get("REPO"), help="repo dir (env REPO)")
    ap.add_argument("--c1b-script", type=Path, default=None, help="default <repo>/scripts/train_c1b_holdout40.py")
    ap.add_argument("--common-checkpoint", type=Path, default=None)
    ap.add_argument("--routing-checkpoint", type=Path, default=None)
    ap.add_argument("--c1b-checkpoint", type=Path, default=None)
    ap.add_argument("--torchsoap-module", type=Path, required=True, help="the descriptor file under test")
    ap.add_argument("--stage-root", type=Path, default=None)
    ap.add_argument("--reference-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--routes", default="trainer32,calc32,calc64")
    ap.add_argument("--n-test", type=int, default=6, help="real holdout-40 TEST records, spread over size (0 = none)")
    ap.add_argument("--max-atoms", type=int, default=1500, help="size cap for real test records")
    ap.add_argument("--no-builtin", action="store_true", help="skip the built-in quartz / cristobalite / random cells")
    ap.add_argument("--structures", nargs="*", default=None, help="extra ASE-readable files")
    ap.add_argument("--nrot", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-fd", action="store_true")
    ap.add_argument("--positive-control", action="store_true",
                    help="DIAGNOSTIC ONLY: replace the descriptor by a subclass with correctly weighted angular basis. "
                         "These are NOT the trained inputs; only shows the harness is exact.")
    ap.add_argument("--tol-E64", type=float, default=1e-6, help="meV/atom, float64 route")
    ap.add_argument("--tol-F64", type=float, default=1e-6, help="eV/A, float64 route")
    ap.add_argument("--tol-E32", type=float, default=1e-2, help="meV/atom, float32 routes")
    ap.add_argument("--tol-F32", type=float, default=1e-4, help="eV/A, float32 routes")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    proj = Path(args.proj).expanduser().resolve() if args.proj else None
    repo = Path(args.repo).expanduser().resolve() if args.repo else None

    def dflt(given, rel, base):
        if given is not None:
            return Path(given).expanduser().resolve()
        if base is None:
            raise SystemExit(f"need --proj/$PROJ (or an explicit path) for {rel}")
        return (base / rel).resolve()

    paths = {
        "common": dflt(args.common_checkpoint, "runs/common_holdout40_ew10000_v1/best_checkpoint.pt", proj),
        "routing": dflt(args.routing_checkpoint, "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt", proj),
        "c1b": dflt(args.c1b_checkpoint, "runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt", proj),
        "soap": Path(args.torchsoap_module).expanduser().resolve(),
    }
    script = dflt(args.c1b_script, "scripts/train_c1b_holdout40.py", repo)
    stage = dflt(args.stage_root, "data/expanded_holdout40_v1_c1b_stage", proj) if (args.stage_root or proj) else None
    ref_dir = dflt(args.reference_dir, "eval_parity_v1/c1b_holdout40_e10000_v1/raw_predictions/test", proj) if (args.reference_dir or proj) else None
    for k, p in {**paths, "trainer": script}.items():
        if not Path(p).is_file():
            raise SystemExit(f"MISSING FILE ({k}): {p}")

    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    routes_req = [r.strip() for r in args.routes.split(",") if r.strip()]

    report = {"script": Path(__file__).name, "script_sha256": sha256(__file__),
              "descriptor_test_sha256": sha256(HERE / "test_torchsoap_rotation.py"),
              "local_moe_ase_sha256": sha256(HERE / "local_moe_ase.py"),
              "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
              "host": platform.node(), "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "date": time.strftime("%Y-%m-%d %H:%M:%S"),
              "files": {k: {"path": str(v), "sha256": sha256(v)} for k, v in {**paths, "trainer_script": script}.items()},
              "positive_control": bool(args.positive_control), "routes": routes_req,
              "tolerances": {"E64_meV_atom": args.tol_E64, "F64_eV_A": args.tol_F64,
                             "E32_meV_atom": args.tol_E32, "F32_eV_A": args.tol_F32},
              "seed": args.seed, "nrot": args.nrot, "results": {}}
    print("=" * 110)
    print("LOCAL-MoE ROTATIONAL-INVARIANCE / CONSERVATIVITY TEST")
    print("=" * 110)
    for k, v in report["files"].items():
        print(f"{k:15s} {v['sha256']}  {v['path']}")
    print(f"{'this script':15s} {report['script_sha256']}")
    print(f"{'descriptor test':15s} {report['descriptor_test_sha256']}")
    print(f"{'local_moe_ase':15s} {report['local_moe_ase_sha256']}")
    print(f"torch {torch.__version__} | numpy {np.__version__} | device {device} | gpu {report['gpu']} | host {report['host']}")
    print(f"routes {routes_req} | seed {args.seed} | random rotations {args.nrot}")
    print(f"tolerances: float64 route dE<{args.tol_E64:g} meV/atom, dF<{args.tol_F64:g} eV/A ; "
          f"float32 routes dE<{args.tol_E32:g} meV/atom, dF<{args.tol_F32:g} eV/A")
    if args.positive_control:
        print("\n*** POSITIVE CONTROL: descriptor replaced by a correctly-weighted DIAGNOSTIC subclass. NOT the trained inputs. ***")

    sys.path.insert(0, str(script.parent))
    trainer = _import_file(script, "rot_c1b_trainer")   # also needed to read the real test records

    # ---------------- descriptor override (diagnostic) ----------------
    override = None
    if args.positive_control:
        soap_mod = _import_file(paths["soap"], "rot_soap_for_weights")
        W = D.control_weights(soap_mod)
        Ctl = D.make_control_class(soap_mod, W)

        def override(soap):
            return Ctl(soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4))

    # ---------------- routes ----------------
    fns = {}
    if "trainer32" in routes_req:
        parts = build_trainer_parts(trainer, paths, device, override)
        fns["trainer32"] = make_trainer_fn(trainer, parts)
    for name, dt in (("calc32", "float32"), ("calc64", "float64")):
        if name in routes_req:
            calc = LocalMoECalculator(paths["common"], paths["routing"], paths["c1b"], paths["soap"],
                                      device=str(device), dtype=dt)
            if override is not None:
                calc.descriptor = override(_import_file(paths["soap"], "rot_soap_calc_" + name)).to(device)
                for p in calc.descriptor.parameters():
                    p.requires_grad_(False)
            fns[name] = make_calc_fn(calc)

    # ---------------- structures ----------------
    structs = []  # (name, atoms, record_id or None)
    if args.n_test > 0 and stage is not None and Path(stage).exists():
        for rec in pick_test_records(trainer, stage, args.n_test, args.max_atoms):
            structs.append((f"TEST {rec.source_file} [{rec.phase_pair}] N={rec.n_atoms}", atoms_from_record(rec), rec.source_file,
                            (float(rec.target_energy_total_eV), np.asarray(rec.target_forces_np, float))))
    elif args.n_test > 0:
        print(f"(stage root not found: {stage}; real test records skipped)")
    if not args.no_builtin:
        for nm, at in D.build_structures(args.seed):
            structs.append((f"BUILT-IN {nm}", at, None, None))
    if args.structures:
        for nm, at in D.load_structures(args.structures, args.seed):
            structs.append((f"FILE {nm}", at, None, None))
    print(f"\n{len(structs)} structures")

    rng = np.random.default_rng(args.seed)
    LRs = [(f"random{k:02d}", D.random_rotation(rng)) for k in range(args.nrot)]
    LRs += [("Rz(30)", D.rot_axis("z", 30)), ("Rz(90)", D.rot_axis("z", 90)), ("Rx(90)", D.rot_axis("x", 90)),
            ("Ry(90)", D.rot_axis("y", 90)), ("Rz(180)", D.rot_axis("z", 180)), ("Rx(180)", D.rot_axis("x", 180))]
    t_shift = np.array([0.37, -1.91, 2.43])

    summary_rows = []
    for route, fn in fns.items():
        is64 = route == "calc64"
        tolE, tolF = (args.tol_E64, args.tol_F64) if is64 else (args.tol_E32, args.tol_F32)
        print("\n" + "#" * 110)
        print(f"# ROUTE {route}   (tolerances: dE < {tolE:g} meV/atom, dF < {tolF:g} eV/A)")
        print("#" * 110)
        acc = {}   # orientation class -> list of (dE_meV_atom, F_mse_struct, F_sumsq, F_count) vs stored MatPES labels
        for name, atoms, rid, target in structs:
            n = len(atoms)
            print(f"\n- {name}\n    {describe_cell(atoms.cell.array)}")
            ref = fn(atoms)
            ref2 = fn(atoms)
            ident = compare(ref, ref2, np.eye(3), n)
            row = {"structure": name, "route": route, "n_atoms": n, "noise_floor": ident, "rotations": {}}
            # stored parity-export predictions
            if rid is not None and ref_dir is not None:
                st = load_reference(ref_dir, rid)
                if st is not None:
                    row["vs_stored"] = {"dE_meV_atom": (ref["E"] - float(st["E_predicted_eV"])) / n * 1e3,
                                        "dF_max": float(np.abs(ref["F"] - st["predicted_forces"]).max())}
                    print(f"    identity vs stored parity-export prediction: dE = {row['vs_stored']['dE_meV_atom']:+.3e} meV/atom,"
                          f" max|dF| = {row['vs_stored']['dF_max']:.3e} eV/A")
            print(f"    noise floor (same input twice): dE = {ident['dE_meV_atom']:+.2e} meV/atom, max|dF| = {ident['dF_max']:.2e} eV/A;"
                  f"  rms|F| = {ident['F_rms']:.3f} eV/A")
            print(f"    {'rotation':10s} {'dE meV/atom':>13s} {'max|dF| eV/A':>14s} {'rms dF/rms F':>13s} {'|sum F| eV/A':>13s} {'dSigma GPa':>11s}")
            rand = []
            if target is not None:
                tE, tF = target
                def _acc(cls, o, R):
                    e = (o["E"] - tE) / n * 1e3
                    d = o["F"] - tF @ R.T
                    acc.setdefault(cls, []).append((e, float(np.mean(d ** 2)), float(np.sum(d ** 2)), d.size))
                _acc("identity (native orientation)", ref, np.eye(3))
            for lab, R in LRs:
                out = fn(D.rotated(atoms, R))
                m = compare(ref, out, R, n)
                row["rotations"][lab] = m
                if target is not None:
                    _acc("random SO(3) (pooled)" if lab.startswith("random") else lab, out, R)
                if lab.startswith("random"):
                    rand.append(m)
                    continue
                print(f"    {lab:10s} {m['dE_meV_atom']:+13.4e} {m['dF_max']:14.4e} {m['dF_rel']:13.3e} {m['sumF_rot']:13.3e} "
                      f"{m.get('dS_GPa', float('nan')):11.3e}")
            dEr = np.array([m["dE_meV_atom"] for m in rand])
            dFr = np.array([m["dF_max"] for m in rand])
            dFrel = np.array([m["dF_rel"] for m in rand])
            row["random"] = {"max_abs_dE": float(np.abs(dEr).max()), "rms_dE": float(np.sqrt(np.mean(dEr ** 2))),
                             "max_dF": float(dFr.max()), "max_dF_rel": float(dFrel.max()),
                             "max_dS_GPa": float(max(m.get("dS_GPa", 0.0) for m in rand))}
            print(f"    random SO(3) (n={len(rand)}): max|dE| = {row['random']['max_abs_dE']:.4e} meV/atom, rms dE = {row['random']['rms_dE']:.4e}, "
                  f"max|dF| = {row['random']['max_dF']:.4e} eV/A, max rms dF/rms F = {row['random']['max_dF_rel']:.3e}")
            # translation
            at_t = atoms.copy()
            at_t.positions += t_shift
            mt = compare(ref, fn(at_t), np.eye(3), n)
            row["translation"] = mt
            print(f"    translation: dE = {mt['dE_meV_atom']:+.2e} meV/atom, max|dF| = {mt['dF_max']:.2e} eV/A, |sum F| = {mt['sumF_rot']:.2e}")
            ok = (abs(row["random"]["max_abs_dE"]) < tolE) and (row["random"]["max_dF"] < tolF)
            row["rotation_invariance_pass"] = bool(ok)
            row["translation_pass"] = bool(abs(mt["dE_meV_atom"]) < tolE and mt["dF_max"] < tolF)
            print(f"    -> rotation invariance within tolerance: {'PASS' if ok else 'FAIL'};  translation: {'PASS' if row['translation_pass'] else 'FAIL'}")
            summary_rows.append(row)
            report["results"].setdefault(route, []).append(row)

        if acc:
            print(f"\n  ACCURACY vs ORIENTATION [{route}] on {len(acc['identity (native orientation)'])} real holdout-40 TEST records "
                  f"(labels = stored MatPES r2SCAN energies/forces, forces rotated consistently: F_ref' = F_ref R^T)")
            print(f"    {'orientation':32s} {'E RMSE meV/atom':>16s} {'E MAE meV/atom':>15s} {'F SB-RMSE eV/A':>15s} {'F pooled RMSE':>14s} {'n':>4s}")
            report.setdefault("accuracy_vs_orientation", {})[route] = {}
            for cls, v in acc.items():
                e = np.array([x[0] for x in v])
                sb = float(np.sqrt(np.mean([x[1] for x in v])))
                pooled = float(np.sqrt(sum(x[2] for x in v) / sum(x[3] for x in v)))
                rec_ = {"E_RMSE": float(np.sqrt(np.mean(e ** 2))), "E_MAE": float(np.mean(np.abs(e))),
                        "F_SB_RMSE": sb, "F_pooled_RMSE": pooled, "n": len(v)}
                report["accuracy_vs_orientation"][route][cls] = rec_
                print(f"    {cls:32s} {rec_['E_RMSE']:16.4f} {rec_['E_MAE']:15.4f} {sb:15.5f} {pooled:14.5f} {len(v):4d}")
        # conservativity (float64 route only)
        if is64 and not args.no_fd:
            print("\n  Conservativity (calc64): F = -dE/dx by central differences, and Hessian symmetry (curl-free test)")
            name, atoms, _, _ = min(structs, key=lambda t: len(t[1]))
            Rc = D.random_rotation(np.random.default_rng(args.seed + 1))
            cons = {}
            for tag, at in (("unrotated", atoms), ("rotated (random R)", D.rotated(atoms, Rc))):
                c = fd_and_hessian(fn, at, np.random.default_rng(args.seed + 2))
                cons[tag] = c
                print(f"    {tag:20s} on '{name}': max|F_autograd - F_fd| = {c['fd_force_max_abs_err']:.3e} eV/A (h={c['fd_h']:g} A; {c['fd_force_max_abs_err_half_h']:.3e} at h/2; {c['n_fd']} components);"
                      f" Hessian asymmetry max|H_ab-H_ba| = {c['hessian_asym_max']:.3e} (scale {c['hessian_scale']:.3e} eV/A^2)")
            report["conservativity"] = {"structure": name, **cons}

    # ---------------- final summary ----------------
    print("\n" + "=" * 110)
    print("SUMMARY (worst case over the structures listed above)")
    print("=" * 110)
    print(f"{'route':10s} {'max noise dE':>13s} {'max noise dF':>13s} | {'max|dE| rot':>13s} {'max|dF| rot':>13s} {'max rel dF':>11s} {'max dS GPa':>11s} | {'trans dE':>10s} {'trans dF':>10s} | rot-inv")
    for route in fns:
        rows = report["results"][route]
        nd = max(abs(r["noise_floor"]["dE_meV_atom"]) for r in rows)
        nf = max(r["noise_floor"]["dF_max"] for r in rows)
        rd = max(r["random"]["max_abs_dE"] for r in rows)
        rf = max(r["random"]["max_dF"] for r in rows)
        rr = max(r["random"]["max_dF_rel"] for r in rows)
        rs = max(r["random"]["max_dS_GPa"] for r in rows)
        td = max(abs(r["translation"]["dE_meV_atom"]) for r in rows)
        tf = max(r["translation"]["dF_max"] for r in rows)
        ok = all(r["rotation_invariance_pass"] for r in rows)
        print(f"{route:10s} {nd:13.2e} {nf:13.2e} | {rd:13.4e} {rf:13.4e} {rr:11.3e} {rs:11.3e} | {td:10.2e} {tf:10.2e} | {'PASS' if ok else 'FAIL'}")
        report.setdefault("summary", {})[route] = {"noise_dE": nd, "noise_dF": nf, "rot_dE": rd, "rot_dF": rf,
                                                   "rot_dF_rel": rr, "rot_dS_GPa": rs, "trans_dE": td, "trans_dF": tf,
                                                   "rotation_invariance_pass": bool(ok)}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
