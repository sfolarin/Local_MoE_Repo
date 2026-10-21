#!/usr/bin/env python3
"""
eval_gamma66_common_only.py

Interface energies gamma = (E_AB - E_A - E_B) / (2 area) * 16.02176634  [J/m^2] from the COMMON
conservative energy network of a retrained run (no router, no C1b residual), for every row of
gamma66_manifest.csv, on the identical MATPes-relaxed AB / A / B geometries used for the MATPes values.

How the energies are computed (nothing is re-implemented):
  * the common network is loaded with the saved C1b loader (load_common_checkpoint), exactly as the
    training script's startup check does
  * the descriptor and trainer functions are the ones snapshotted in the run's src_snapshot
  * each ASE structure is wrapped into the trainer's StructureRecord and passed to
    trainer.predict_energy_forces, the function that produced the validation/test numbers
  * ROUTE CHECK (on by default): a few real validation structures are evaluated both through the
    training script's own record loader and through the ASE route used here; energies and forces
    must agree to 1e-9, otherwise the script stops

Compared with MATPes (from the manifest) and with the previous expanded full-MoE gamma stored in
oishi_gamma_candidates.csv (context only: that model was fitted on a different split).

Needs one GPU. Run from External_Benchmark_Oishi2025/ inside a GPU allocation:
    python eval_gamma66_common_only.py --run <PROJ>/runs/common_holdout40_ew10000_v1
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
from types import SimpleNamespace

import numpy as np
import pandas as pd

CONV = 16.02176634
HET_DEFAULT = (_os.environ.get("MOE_HET", "."))
COMMON = "train_conservative_common_energy_baseline_augmented.py"
C1B = "train_c1b_frozen_routing_conservative_residual_v3_196train.py"
SOAP = "torchsoap_autograd_module_periodic_regnorm.py"


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def record_from_atoms(trainer, atoms, name):
    """Wrap an ASE structure into the trainer's StructureRecord (targets are unused dummies)."""
    n = len(atoms)
    z = np.asarray(atoms.numbers)
    symbols, masses = trainer.build_symbols_masses(z)
    return trainer.StructureRecord(
        source_file=name, phase_pair="", split="gamma", n_atoms=n, feature_file=Path(""), source_shard=Path(""),
        atomic_numbers=z, positions_np=np.array(atoms.positions, dtype=np.float64), cell_np=np.array(atoms.cell, dtype=np.float64),
        phase=None, interface_center=None, target_energy_total_eV=0.0, target_forces_np=np.zeros((n, 3)),
        symbols=symbols, masses=masses)


def compute_gamma_table(manifest, ext, energy_fn, read):
    """gamma for every manifest row; energy_fn(atoms, name) -> eV; read(path) -> ASE atoms."""
    rows = []
    for r in manifest.itertuples():
        ab, a, b = (read(ext / p) for p in (r.ab_path, r.a_path, r.b_path))
        if len(a) + len(b) != len(ab):
            raise ValueError(f"{r.case}: atom counts do not close ({len(a)} + {len(b)} != {len(ab)})")
        area = float(np.linalg.norm(np.cross(ab.cell[0], ab.cell[1])))
        e_ab, e_a, e_b = energy_fn(ab, r.case + ":AB"), energy_fn(a, r.case + ":A"), energy_fn(b, r.case + ":B")
        rows.append({"case": r.case, "N_AB": len(ab), "area_A2": area, "E_AB_eV": e_ab, "E_A_eV": e_a, "E_B_eV": e_b,
                     "gamma_common_J_m2": (e_ab - e_a - e_b) / (2.0 * area) * CONV})
    return pd.DataFrame(rows)


def stats(x, y):
    e = y - x
    d = {"N": len(x), "MAE": float(np.abs(e).mean()), "bias": float(e.mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}
    if len(x) >= 8:
        d["Pearson"] = float(np.corrcoef(x, y)[0, 1])
        d["slope"] = float(np.polyfit(x, y, 1)[0])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--het", default=HET_DEFAULT)
    ap.add_argument("--run", type=Path, required=True, help="common-model run directory (contains best_checkpoint.pt and src_snapshot)")
    ap.add_argument("--features", type=Path, default=None, help="feature root used for training (for the route check)")
    ap.add_argument("--manifest", default="gamma66_manifest.csv")
    ap.add_argument("--candidates", default="oishi_gamma_candidates.csv")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--check-n", type=int, default=3)
    ap.add_argument("--skip-route-check", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from ase.io import read as ase_read

    HET = Path(args.het)
    EXT = HET / "External_Benchmark_Oishi2025"
    PROJ = HET / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    run = args.run.expanduser().resolve()
    features = (args.features or PROJ / "data/expanded_holdout40_v1_features").expanduser().resolve()
    _sys.path.extend(_p for _p in _REPO_CODE_DIRS if _p not in _sys.path)
    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    snap = run / "src_snapshot"
    trainer = import_file(snap / COMMON, "gamma66_common_trainer")
    c1b = import_file(snap / C1B, "gamma66_c1b_loader")
    soap = trainer.import_torchsoap(snap / SOAP)
    descriptor = soap.TorchSOAPDescriptor(soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    descriptor.eval()
    for p in descriptor.parameters():
        p.requires_grad_(False)

    ckpt = run / "best_checkpoint.pt"
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert meta.get("stage") == "expanded_common_energy" and meta.get("selection_split") == "val", "unexpected checkpoint"
    model, mean, std, e_ref, _ = c1b.load_common_checkpoint(ckpt, device)
    model.eval()
    print(f"model: {run.name} | energy weight {meta['energy_weight']} | best epoch {meta['best_epoch']} "
          f"(selected on validation) | e_ref {e_ref:.6f} eV/atom | contract {str(meta['training_contract_sha256'])[:12]}")

    ctx = SimpleNamespace(torch=torch, trainer=trainer, descriptor=descriptor, device=device,
                          mean=mean.to(device), std=std.to(device), e_ref=e_ref)

    def energy_fn(atoms, name):
        at = atoms.copy()
        at.pbc = True
        rec = record_from_atoms(trainer, at, name)
        with torch.enable_grad():
            e, f = trainer.predict_energy_forces(model=model, descriptor=descriptor, record=rec, feature_mean=ctx.mean,
                                                 feature_std=ctx.std, e_ref=e_ref, device=device, create_graph=False)
        value = float(e.detach().cpu())
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite energy for " + name)
        del e, f
        return value

    if not args.skip_route_check:
        tm = import_file(REPO_ROOT / 'training/train_expanded_moe_common.py', "gamma66_training_script")
        inputs = tm.verify_inputs(features)
        recs = sorted(tm.load_records(inputs, trainer, ("val",))["val"], key=lambda r: r.n_atoms)[:args.check_n]
        from ase import Atoms
        worst_e = worst_f = 0.0
        for rec in recs:
            with torch.enable_grad():
                e1, f1 = tm.prediction(ctx, model, rec, create_graph=False)
            atoms = Atoms(numbers=rec.atomic_numbers, positions=rec.positions_np, cell=rec.cell_np, pbc=True)
            rec2 = record_from_atoms(trainer, atoms, rec.source_file)
            with torch.enable_grad():
                e2, f2 = trainer.predict_energy_forces(model=model, descriptor=descriptor, record=rec2, feature_mean=ctx.mean,
                                                       feature_std=ctx.std, e_ref=e_ref, device=device, create_graph=False)
            worst_e = max(worst_e, abs(float(e1) - float(e2)))
            worst_f = max(worst_f, float((f1 - f2).abs().max()))
        print(f"route check on {len(recs)} validation structures (N={[r.n_atoms for r in recs]}): "
              f"max |dE| = {worst_e:.2e} eV, max |dF| = {worst_f:.2e} eV/A")
        if worst_e > 1e-9 or worst_f > 1e-9:
            raise SystemExit("ROUTE CHECK FAILED: the ASE route does not reproduce the training route; do not use these gammas")

    manifest = pd.read_csv(EXT / args.manifest)
    t = compute_gamma_table(manifest, EXT, energy_fn, lambda p: ase_read(p, index=-1))
    cand = pd.read_csv(EXT / args.candidates, keep_default_na=False)
    old = cand[cand.drop_reason.astype(str) == ""][["case", "gamma_LocalMoE_J_m2"]].rename(columns={"gamma_LocalMoE_J_m2": "gamma_prev_expanded_J_m2"})
    t = manifest[["case", "origin", "pair", "new_split", "old_pool_split", "gamma_MATPES_J_m2"]].merge(t, on="case").merge(old, on="case", how="left")
    t["err_common"] = t.gamma_common_J_m2 - t.gamma_MATPES_J_m2
    t["err_prev_expanded"] = t.gamma_prev_expanded_J_m2 - t.gamma_MATPES_J_m2
    out = args.out or f"gamma66_{run.name}.csv"
    t.to_csv(out, index=False)

    print(f"\nGAMMA (J/m2), {len(t)} rows. Common network of {run.name} vs MATPes; the previous expanded full MoE is context only")
    rows = []
    for name, g in [("all", t)] + [(k, v) for k, v in t.groupby("new_split")]:
        rows.append({"set": name, "model": "new common (this run)", **stats(g.gamma_MATPES_J_m2.to_numpy(), g.gamma_common_J_m2.to_numpy())})
        rows.append({"set": name, "model": "previous expanded (context)", **stats(g.gamma_MATPES_J_m2.to_numpy(), g.gamma_prev_expanded_J_m2.to_numpy())})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    te = t[t.new_split == "test"].copy()
    print(f"\nHELD-OUT families (new split = test): errors positive on {int((te.err_common > 0).sum())} of {len(te)}; "
          f"max predicted gamma {te.gamma_common_J_m2.max():.4f} vs MATPes max {te.gamma_MATPES_J_m2.max():.4f}")
    print("\nlargest absolute errors among held-out families:")
    print(te.reindex(te.err_common.abs().sort_values(ascending=False).index).head(8)[
        ["case", "gamma_MATPES_J_m2", "gamma_common_J_m2", "err_common", "old_pool_split"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
