#!/usr/bin/env python3
"""
eval_gamma66_fullmoe.py

Interface energies gamma = (E_AB - E_A - E_B) / (2 area) * 16.02176634  [J/m^2] from the FULL trained Local-MoE
(common conservative network + router-weighted C1b residual), for every row of gamma66_manifest.csv, on the
identical MATPes-relaxed AB / A / B geometries used for the MATPes values.

The trainer's own model_terms() returns the total energy together with its two parts
    total_energy = N * e_ref + common_residual_total + moe_residual_total
so a single pass gives, for every row, the full-model gamma AND the common-network-only gamma AND the
residual's exact contribution (the N * e_ref terms cancel because N_AB = N_A + N_B). The identity
    gamma_full = gamma_common_part + gamma_residual_part
is checked and the script stops if it fails.

Nothing is re-implemented: models are loaded with the trainer's own loaders, structures are wrapped into the
trainer's StructureRecord, and trainer.predict_energy_forces (the function that produced the validation/test
numbers) is called.

Gates (all on by default; the script stops if one fails):
  * the C1b checkpoint's recorded common/routing checkpoint paths must match the ones passed (if recorded)
  * ROUTE CHECK: a few real validation structures are evaluated both through the trainer's own record loader
    and through the ASE route used here; energies and forces must agree within --route-tol
  * the gamma identity above

Also compared (information only): the common part here against the stand-alone common-network evaluation
(gamma_common_J_m2 in --common-gamma-csv), which should agree to float precision.

Needs one GPU. Run from External_Benchmark_Oishi2025/ inside a GPU allocation:
    python eval_gamma66_fullmoe.py --c1b-checkpoint <PROJ>/runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt
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
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CONV = 16.02176634
HET_DEFAULT = (_os.environ.get("MOE_HET", "."))
REPO_DEFAULT = str(REPO_ROOT)
PARTS = (("total_energy", "full"), ("common_residual_total", "common_part"), ("moe_residual_total", "residual_part"))


def import_file(path, name):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def record_from_atoms(trainer, atoms, name):
    """Wrap an ASE structure into the trainer's StructureRecord (targets are unused dummies)."""
    n = len(atoms)
    z = np.asarray(atoms.numbers).astype(np.int16)
    symbols, masses = trainer.build_symbols_masses(z)
    return trainer.StructureRecord(
        source_file=name, phase_pair="", split="gamma", n_atoms=n, feature_file=Path(""), source_shard=Path(""),
        atomic_numbers=z, positions_np=np.array(atoms.positions, dtype=np.float64),
        cell_np=np.array(atoms.cell, dtype=np.float64),
        phase=np.zeros(n, dtype=np.int16), interface_center=np.zeros(n, dtype=np.int16),
        target_energy_total_eV=0.0, target_forces_np=np.zeros((n, 3)), symbols=symbols, masses=masses)


def stats(x, y):
    e = y - x
    d = {"N": len(x), "MAE": float(np.abs(e).mean()), "bias": float(e.mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}
    if len(x) >= 8:
        d["Pearson"] = float(np.corrcoef(x, y)[0, 1])
        d["slope"] = float(np.polyfit(x, y, 1)[0])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--het", default=HET_DEFAULT)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--c1b-script", type=Path, default=None, help="default: <repo>/training/train_c1b_holdout40.py")
    ap.add_argument("--c1b-checkpoint", type=Path, required=True, help="best_c1b_residual_model.pt of the run")
    ap.add_argument("--common-checkpoint", type=Path, default=None)
    ap.add_argument("--routing-checkpoint", type=Path, default=None)
    ap.add_argument("--torchsoap-module", type=Path, default=None)
    ap.add_argument("--stage-root", type=Path, default=None, help="C1b stage root (for the route check)")
    ap.add_argument("--manifest", default="gamma66_manifest.csv")
    ap.add_argument("--common-gamma-csv", default="gamma66_common_holdout40_ew10000_v1.csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-n", type=int, default=3)
    ap.add_argument("--route-tol", type=float, default=1e-9)
    ap.add_argument("--skip-route-check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from ase import Atoms
    from ase.io import read as ase_read

    HET = Path(args.het)
    EXT = HET / "External_Benchmark_Oishi2025"
    PROJ = HET / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    script = args.c1b_script or REPO_ROOT / 'training/train_c1b_holdout40.py'
    common_p = (args.common_checkpoint or PROJ / "runs/common_holdout40_ew10000_v1/best_checkpoint.pt").expanduser().resolve()
    routing_p = (args.routing_checkpoint or PROJ / "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt").expanduser().resolve()
    soap_p = (args.torchsoap_module or PROJ / "data/expanded_holdout40_v1_features/src_snapshot/torchsoap_autograd_module_periodic_regnorm.py").expanduser().resolve()
    stage = (args.stage_root or PROJ / "data/expanded_holdout40_v1_c1b_stage").expanduser().resolve()
    c1b_p = args.c1b_checkpoint.expanduser().resolve()
    for p in (script, common_p, routing_p, soap_p, c1b_p):
        if not Path(p).is_file():
            raise SystemExit(f"MISSING FILE: {p}")

    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    trainer = import_file(script, "gamma66_c1b_trainer")
    common_model, common_mean, common_std, e_ref, _ = trainer.load_common_checkpoint(common_p, device)
    routing_model, routing_mean, routing_std, _ = trainer.load_routing_checkpoint(routing_p, device)
    bck = torch.load(c1b_p, map_location="cpu", weights_only=False)
    if [str(x) for x in bck["expert_names"]] != list(trainer.EXPERTS):
        raise SystemExit("C1b expert order does not match the trainer")
    for key, passed in (("common_checkpoint", common_p), ("routing_checkpoint", routing_p)):
        rec = bck.get(key)
        if rec is not None and Path(str(rec)).resolve() != passed:
            raise SystemExit(f"GATE FAILED: the C1b checkpoint was trained with {key}\n  {rec}\nbut you passed\n  {passed}")

    residual_model = trainer.MLP(input_dim=int(bck["input_dim"]), hidden_dims=[int(x) for x in bck["residual_hidden_dims"]],
                                 output_dim=7, dropout=0.0).to(device)
    residual_model.load_state_dict(bck["residual_model_state_dict"])
    residual_model.eval()
    residual_mean, residual_std = common_mean, common_std     # as in the trainer: the residual head reuses the common normalization

    soap = trainer.import_torchsoap(soap_p)
    descriptor = soap.TorchSOAPDescriptor(soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    for p in descriptor.parameters():
        p.requires_grad_(False)

    run_name = c1b_p.parent.name
    print(f"model: {run_name} | energy weight {bck.get('energy_weight')} / force weight {bck.get('force_weight')} | "
          f"best epoch {bck.get('best_epoch')} (selected on validation) | e_ref {e_ref:.6f} eV/atom")

    def predict(record):
        with torch.enable_grad():
            return trainer.predict_energy_forces(
                record=record, descriptor=descriptor, common_model=common_model, common_mean=common_mean,
                common_std=common_std, routing_model=routing_model, routing_mean=routing_mean, routing_std=routing_std,
                residual_model=residual_model, residual_mean=residual_mean, residual_std=residual_std,
                e_ref=e_ref, device=device, create_graph=False)

    def terms_fn(atoms, name):
        at = atoms.copy()
        at.pbc = True
        terms, forces = predict(record_from_atoms(trainer, at, name))
        out = {k: float(terms[k].detach().cpu()) for k, _ in PARTS}
        del terms, forces
        if not all(np.isfinite(v) for v in out.values()):
            raise FloatingPointError("nonfinite energy for " + name)
        return out

    if not args.skip_route_check:
        recs = sorted(trainer.load_records(stage)["val"], key=lambda r: r.n_atoms)[:args.check_n]
        worst_e = worst_f = 0.0
        for rec in recs:
            t1, f1 = predict(rec)
            atoms = Atoms(numbers=rec.atomic_numbers, positions=rec.positions_np, cell=rec.cell_np, pbc=True)
            t2, f2 = predict(record_from_atoms(trainer, atoms, rec.source_file))
            worst_e = max(worst_e, abs(float(t1["total_energy"].detach().cpu()) - float(t2["total_energy"].detach().cpu())))
            worst_f = max(worst_f, float(np.abs(f1.detach().cpu().numpy() - f2.detach().cpu().numpy()).max()))
        print(f"route check on {len(recs)} validation structures (N={[r.n_atoms for r in recs]}): "
              f"max |dE| = {worst_e:.2e} eV, max |dF| = {worst_f:.2e} eV/A (tolerance {args.route_tol:g})")
        if worst_e > args.route_tol or worst_f > args.route_tol:
            raise SystemExit("ROUTE CHECK FAILED: the ASE route does not reproduce the training route; "
                             "do not use these gammas (or raise --route-tol only if the difference is float noise)")

    manifest = pd.read_csv(EXT / args.manifest)
    rows = []
    for i, r in enumerate(manifest.itertuples(), 1):
        ab, a, b = (ase_read(EXT / p, index=-1) for p in (r.ab_path, r.a_path, r.b_path))
        if len(a) + len(b) != len(ab):
            raise ValueError(f"{r.case}: atom counts do not close ({len(a)} + {len(b)} != {len(ab)})")
        area = float(np.linalg.norm(np.cross(ab.cell[0], ab.cell[1])))
        t_ab, t_a, t_b = terms_fn(ab, r.case + ":AB"), terms_fn(a, r.case + ":A"), terms_fn(b, r.case + ":B")
        row = {"case": r.case, "N_AB": len(ab), "area_A2": area}
        for key, short in PARTS:
            row[f"{short}_AB_eV"], row[f"{short}_A_eV"], row[f"{short}_B_eV"] = t_ab[key], t_a[key], t_b[key]
            row[f"gamma_{short}_J_m2"] = (t_ab[key] - t_a[key] - t_b[key]) / (2.0 * area) * CONV
        rows.append(row)
        if i % 10 == 0 or i == len(manifest):
            print(f"  {i}/{len(manifest)} rows done", flush=True)
    t = pd.DataFrame(rows)

    ident = float((t.gamma_full_J_m2 - t.gamma_common_part_J_m2 - t.gamma_residual_part_J_m2).abs().max())
    print(f"identity gamma_full = gamma_common_part + gamma_residual_part: max |difference| = {ident:.2e} J/m2")
    if ident > 1e-7:
        raise SystemExit("GATE FAILED: the energy parts do not add up; do not use these gammas")

    cols = ["case", "origin", "pair", "new_split", "old_pool_split", "gamma_MATPES_J_m2"]
    t = manifest[cols].merge(t, on="case")
    t["err_full"] = t.gamma_full_J_m2 - t.gamma_MATPES_J_m2
    t["err_common_part"] = t.gamma_common_part_J_m2 - t.gamma_MATPES_J_m2
    t["abs_err_change"] = t.err_full.abs() - t.err_common_part.abs()      # negative: the residual helped this row

    cg = EXT / args.common_gamma_csv
    if cg.is_file():
        c = pd.read_csv(cg)[["case", "gamma_common_J_m2"]].merge(t[["case", "gamma_common_part_J_m2"]], on="case")
        d = float((c.gamma_common_J_m2 - c.gamma_common_part_J_m2).abs().max())
        flag = "" if d < 1e-4 else "   <-- WARNING: larger than float noise; the two common-network evaluations disagree"
        print(f"common part vs stand-alone common evaluation ({len(c)} rows): max |difference| = {d:.2e} J/m2{flag}")
    else:
        print(f"(no stand-alone common evaluation found at {cg}; comparison skipped)")

    out = args.out or f"gamma66_fullmoe_{run_name}.csv"
    t.to_csv(out, index=False)

    print(f"\nGAMMA (J/m2), {len(t)} rows. Full Local-MoE and its common-network part, vs MATPes")
    summ = []
    for name, g in [("all", t)] + [(k, v) for k, v in t.groupby("new_split")]:
        x = g.gamma_MATPES_J_m2.to_numpy()
        summ.append({"set": name, "model": "full Local-MoE", **stats(x, g.gamma_full_J_m2.to_numpy())})
        summ.append({"set": name, "model": "common part only", **stats(x, g.gamma_common_part_J_m2.to_numpy())})
    print(pd.DataFrame(summ).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    te = t[t.new_split == "test"].copy()
    ch = te.abs_err_change.to_numpy()
    tol = 1e-4
    print(f"\nHELD-OUT families (new split = test), N = {len(te)}")
    print(f"  residual part of gamma: mean {te.gamma_residual_part_J_m2.mean():+.4f}, mean |.| {te.gamma_residual_part_J_m2.abs().mean():.4f}, "
          f"max |.| {te.gamma_residual_part_J_m2.abs().max():.4f} J/m2")
    print(f"  rows where the residual reduced |error| by > {tol:g}: {int((ch < -tol).sum())}, "
          f"increased it by > {tol:g}: {int((ch > tol).sum())}, within +/-{tol:g}: {int((np.abs(ch) <= tol).sum())}")
    print(f"  mean change of |error| from adding the residual: {ch.mean():+.5f} J/m2 (negative = residual helps)")
    print(f"  errors positive on {int((te.err_full > 0).sum())} of {len(te)}; max predicted gamma {te.gamma_full_J_m2.max():.4f} "
          f"vs MATPes max {te.gamma_MATPES_J_m2.max():.4f}")
    print("\nlargest absolute errors among held-out families (full Local-MoE):")
    print(te.reindex(te.err_full.abs().sort_values(ascending=False).index).head(8)[
        ["case", "gamma_MATPES_J_m2", "gamma_full_J_m2", "err_full", "err_common_part", "gamma_residual_part_J_m2"]
    ].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
