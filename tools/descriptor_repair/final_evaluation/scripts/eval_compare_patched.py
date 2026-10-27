#!/usr/bin/env python3
# adapted from: <scratch>/my_project/MoE-Project-New/database/MoE-v6/Without-Descriptor/RUN_3x3x3_Angle/HETEROSTRUCTURES/Local_MoE_MATPES_R2SCAN_Expanded_v1/scripts/eval_test_regions_common_vs_full.py
# adapter: patch_compare.py -- tolerate a C1b run with no recorded test
#   numbers (0-byte test_region_force_metrics.csv, test-less
#   final_split_metrics.csv).  Reproduction of those values is skipped;
#   the comparison itself and the common-side reproduction are unchanged.
# original sha256: 42845f3c68fdc51189bb09a33af4c207b097ea11c7068f0b3128404e56cae8e1
"""
eval_test_regions_common_vs_full.py

Test-split energy / force errors and the region force breakdown, for
  * the COMMON NETWORK ALONE  (the C1b residual head's parameters are set to zero, so the model is exactly
    N*e_ref + common network; moe_residual_total is checked to be 0), and
  * the FULL Local-MoE of every C1b checkpoint given,
all through the trainer's own record loader, region_masks and predict_energy_forces.

Before anything is reported the script REPRODUCES numbers that already exist:
  * for every C1b run: test_region_force_metrics.csv (all four regions, atom-weighted and structure-balanced
    force RMSE) and the test row of final_split_metrics.csv (energy RMSE / MAE per atom)
  * for the common network: the test row (source ALL, family ALL) of the stand-alone evaluation's metrics.csv
    (energy RMSE / MAE / bias, structure-balanced and atom-weighted force RMSE)
If any of them is not reproduced (relative tolerance --check-rtol) the script writes its per-structure file and
stops without printing a comparison.  The structure-balanced definition (RMS over structures of the per-structure
MSE, or mean over structures of the per-structure RMSE) is not assumed: both are computed and the one that
reproduces the trainer's files is used.

Needs one GPU.  Run from anywhere; outputs go to the current directory:
    python eval_test_regions_common_vs_full.py --c1b-checkpoint RUN_A/best_c1b_residual_model.pt RUN_B/best_c1b_residual_model.pt
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HET_DEFAULT = ("<scratch>/my_project/MoE-Project-New/database/MoE-v6/"
               "Without-Descriptor/RUN_3x3x3_Angle/HETEROSTRUCTURES")
REPO_DEFAULT = "<scratch>/GitHub-Repositories/local-moe-silica-interfaces"
REGIONS = ("all", "interface", "phase1_bulk", "phase2_bulk")
COMMON = "common_only"
SB_DEFS = ("sb_rms", "sb_mean")


def import_file(path, name):
    path = Path(path)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def agg(rows, region):
    """rows: per-structure dicts {dE, sumsq{region}, n{region}}.  Force RMSEs are per Cartesian component."""
    s = [r for r in rows if r["n"][region] > 0]
    n_atoms = sum(r["n"][region] for r in s)
    mse = np.array([r["sumsq"][region] / (3.0 * r["n"][region]) for r in s])
    de = np.array([r["dE"] for r in rows])
    return {"n_struct": len(s), "n_atoms": n_atoms,
            "aw": float(np.sqrt(sum(r["sumsq"][region] for r in s) / (3.0 * n_atoms))),
            "sb_rms": float(np.sqrt(mse.mean())), "sb_mean": float(np.sqrt(mse).mean()),
            "e_rmse": float(np.sqrt((de ** 2).mean())), "e_mae": float(np.abs(de).mean()), "e_bias": float(de.mean())}


def close(a, b, rtol):
    return abs(a - b) <= 1e-9 + rtol * abs(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--het", default=HET_DEFAULT)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--c1b-script", type=Path, default=None, help="default: <repo>/scripts/train_c1b_holdout40.py")
    ap.add_argument("--c1b-checkpoint", type=Path, nargs="+", required=True, help="best_c1b_residual_model.pt of each run")
    ap.add_argument("--common-checkpoint", type=Path, default=None)
    ap.add_argument("--routing-checkpoint", type=Path, default=None)
    ap.add_argument("--torchsoap-module", type=Path, default=None)
    ap.add_argument("--stage-root", type=Path, default=None)
    ap.add_argument("--common-eval-dir", type=Path, default=None, help="evaluation_train_val_test of the common run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--expect-n-test", type=int, default=105)
    ap.add_argument("--check-rtol", type=float, default=1e-5)
    ap.add_argument("--out-prefix", default="test_regions_common_vs_full")
    args = ap.parse_args()

    import torch

    HET = Path(args.het)
    PROJ = HET / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    script = args.c1b_script or Path(args.repo) / "scripts/train_c1b_holdout40.py"
    common_p = (args.common_checkpoint or PROJ / "runs/common_holdout40_ew10000_v1/best_checkpoint.pt").expanduser().resolve()
    routing_p = (args.routing_checkpoint or PROJ / "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt").expanduser().resolve()
    soap_p = (args.torchsoap_module or PROJ / "data/expanded_holdout40_v1_features/src_snapshot/torchsoap_autograd_module_periodic_regnorm.py").expanduser().resolve()
    stage = (args.stage_root or PROJ / "data/expanded_holdout40_v1_c1b_stage").expanduser().resolve()
    cdir = (args.common_eval_dir or PROJ / "runs/common_holdout40_ew10000_v1/evaluation_train_val_test").expanduser().resolve()
    c1b_ps = [p.expanduser().resolve() for p in args.c1b_checkpoint]
    for p in [script, common_p, routing_p, soap_p, cdir / "metrics.csv"] + c1b_ps:
        if not Path(p).is_file():
            raise SystemExit(f"MISSING FILE: {p}")

    torch.set_default_dtype(torch.float32)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    trainer = import_file(script, "regions_c1b_trainer")
    common_model, common_mean, common_std, e_ref, _ = trainer.load_common_checkpoint(common_p, device)
    routing_model, routing_mean, routing_std, _ = trainer.load_routing_checkpoint(routing_p, device)

    def build_residual(bck):
        m = trainer.MLP(input_dim=int(bck["input_dim"]), hidden_dims=[int(x) for x in bck["residual_hidden_dims"]],
                        output_dim=7, dropout=0.0).to(device)
        m.load_state_dict(bck["residual_model_state_dict"])
        return m

    models, bcks = {}, {}
    for cp in c1b_ps:
        bck = torch.load(cp, map_location="cpu", weights_only=False)
        if [str(x) for x in bck["expert_names"]] != list(trainer.EXPERTS):
            raise SystemExit(f"{cp}: C1b expert order does not match the trainer")
        for key, passed in (("common_checkpoint", common_p), ("routing_checkpoint", routing_p)):
            rec = bck.get(key)
            if rec is not None and Path(str(rec)).resolve() != passed:
                raise SystemExit(f"GATE FAILED: {cp.parent.name} was trained with {key}\n  {rec}\nbut you passed\n  {passed}")
        name = cp.parent.name
        if name in models or name == COMMON:
            raise SystemExit(f"duplicate run name {name}")
        m = build_residual(bck)
        m.eval()
        models[name], bcks[name] = m, bck
    first = bcks[next(iter(bcks))]
    zero = build_residual(first)
    with torch.no_grad():
        for p in zero.parameters():
            p.zero_()
    zero.eval()
    models = {COMMON: zero, **models}

    soap = trainer.import_torchsoap(soap_p)
    descriptor = soap.TorchSOAPDescriptor(soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)).to(device)
    for p in descriptor.parameters():
        p.requires_grad_(False)

    def predict(record, residual):
        with torch.enable_grad():
            return trainer.predict_energy_forces(
                record=record, descriptor=descriptor, common_model=common_model, common_mean=common_mean,
                common_std=common_std, routing_model=routing_model, routing_mean=routing_mean, routing_std=routing_std,
                residual_model=residual, residual_mean=common_mean, residual_std=common_std,
                e_ref=e_ref, device=device, create_graph=False)

    recs = trainer.load_records(stage)["test"]
    if len(recs) != args.expect_n_test:
        raise SystemExit(f"test split has {len(recs)} structures, expected {args.expect_n_test}")
    print(f"models: {list(models)} | test structures {len(recs)} | e_ref {e_ref:.6f} eV/atom", flush=True)

    data = {k: [] for k in models}
    meta = []
    max_zero_moe = 0.0
    for i, rec in enumerate(recs, 1):
        masks = trainer.region_masks(rec)
        nreg = {r: int(np.asarray(masks[r]).sum()) for r in REGIONS}
        meta.append({"source_file": rec.source_file, "n_atoms": rec.n_atoms, **{f"n_{r}": nreg[r] for r in REGIONS}})
        tf = np.asarray(rec.target_forces_np, dtype=np.float64)
        for name, res in models.items():
            terms, forces = predict(rec, res)
            e = float(terms["total_energy"].detach().cpu())
            if name == COMMON:
                max_zero_moe = max(max_zero_moe, abs(float(terms["moe_residual_total"].detach().cpu())))
            f = forces.detach().cpu().numpy().astype(np.float64)
            sq = ((f - tf) ** 2).sum(axis=1)
            if not (np.isfinite(e) and np.isfinite(sq).all()):
                raise FloatingPointError(f"nonfinite prediction for {rec.source_file} / {name}")
            data[name].append({"dE": (e - float(rec.target_energy_total_eV)) / rec.n_atoms,
                               "sumsq": {r: float(sq[np.asarray(masks[r])].sum()) for r in REGIONS}, "n": nreg})
            del terms, forces
        if i % 20 == 0 or i == len(recs):
            print(f"  {i}/{len(recs)} structures done", flush=True)
    print(f"common-only model: max |moe_residual_total| over the test split = {max_zero_moe:.2e} eV (must be 0)")
    if max_zero_moe > 1e-12:
        raise SystemExit("GATE FAILED: zeroing the residual head did not remove the residual; no baseline can be reported")

    # per-structure file (written before the reproduction check so a failure can be inspected)
    long_rows = []
    for name, rows in data.items():
        for m, r in zip(meta, rows):
            long_rows.append({"model": name, **m, "dE_eV_per_atom": r["dE"], **{f"sumsq_{k}": v for k, v in r["sumsq"].items()}})
    pd.DataFrame(long_rows).to_csv(f"{args.out_prefix}_per_structure.csv", index=False)

    # ---------------- reproduction of existing numbers ----------------
    refs = []        # (label, model, region, quantity, reference value)
    cm = pd.read_csv(cdir / "metrics.csv")
    cr = cm[(cm.split == "test") & (cm.source == "ALL") & (cm.family == "ALL")]
    if len(cr) != 1:
        raise SystemExit(f"{cdir / 'metrics.csv'}: expected one test/ALL/ALL row, found {len(cr)}")
    cr = cr.iloc[0]
    for q, col, scale in (("e_rmse", "energy_RMSE_meV_per_atom", 1e-3), ("e_mae", "energy_MAE_meV_per_atom", 1e-3),
                          ("sb", "force_RMSE_structure_balanced_eV_A", 1.0), ("aw", "force_RMSE_atom_weighted_eV_A", 1.0)):
        refs.append((f"common metrics.csv {col}", COMMON, "all", q, float(cr[col]) * scale))
    for cp in c1b_ps:
        name, rd = cp.parent.name, cp.parent
        rf = rd / "test_region_force_metrics.csv"
        ff = rd / "final_split_metrics.csv"
        if rf.is_file() and rf.stat().st_size > 0:
            for row in pd.read_csv(rf).itertuples():
                if row.split == "test" and row.region in REGIONS:
                    refs.append((f"{name} test_region_force_metrics aw[{row.region}]", name, row.region, "aw", float(row.atom_weighted_rmse_component_eV_A)))
                    refs.append((f"{name} test_region_force_metrics sb[{row.region}]", name, row.region, "sb", float(row.structure_balanced_rmse_component_eV_A)))
        else:
            print(f"(no usable {rf}: absent or empty -- its reproduction is skipped)")
        _ff_has_test = False
        if ff.is_file() and ff.stat().st_size > 0:
            _ff_has_test = "test" in set(pd.read_csv(ff)["split"].astype(str))
        if _ff_has_test:
            fs = pd.read_csv(ff)
            fs = fs[fs.split == "test"].iloc[0]
            refs.append((f"{name} final_split_metrics energy_rmse", name, "all", "e_rmse", float(fs.energy_rmse_eV_per_atom)))
            refs.append((f"{name} final_split_metrics energy_mae", name, "all", "e_mae", float(fs.energy_mae_eV_per_atom)))
        else:
            print(f"(no usable {ff}: absent, empty, or no test row -- its reproduction is skipped)")

    def mine(model, region, q, sbdef):
        a = agg(data[model], region)
        return a["sb_rms" if q == "sb" and sbdef == "sb_rms" else "sb_mean" if q == "sb" else q]

    def worst(sbdef):
        return max(abs(mine(m, r, q, sbdef) - v) / max(abs(v), 1e-12) for _, m, r, q, v in refs)

    sbdef = min(SB_DEFS, key=worst)
    bad = []
    chk = []
    for label, m, r, q, v in refs:
        x = mine(m, r, q, sbdef)
        ok = close(x, v, args.check_rtol)
        chk.append({"quantity": label, "this script": x, "existing file": v, "rel diff": abs(x - v) / max(abs(v), 1e-12), "ok": ok})
        if not ok:
            bad.append(label)
    chk = pd.DataFrame(chk)
    print(f"\nREPRODUCTION OF EXISTING NUMBERS (structure-balanced definition that matches the trainer: {sbdef}; rtol {args.check_rtol:g})")
    print(f"  {len(chk)} numbers compared, {int(chk.ok.sum())} reproduced, max relative difference {chk['rel diff'].max():.2e}")
    if bad:
        print(chk[~chk.ok].to_string(index=False, float_format=lambda v: f"{v:.8g}"))
        raise SystemExit("REPRODUCTION CHECK FAILED: the numbers above are not reproduced, so no comparison is reported "
                         "(per-structure file written for inspection)")

    # ---------------- comparison ----------------
    names = list(models)
    lab = {COMMON: "common network"}
    lab.update({n: ("full MoE " + n.replace("c1b_holdout40_", "")) for n in names if n != COMMON})
    print(f"\nTEST SPLIT, all {len(recs)} structures, {sum(r['n_atoms'] for r in meta)} atoms (energy per atom; forces per component)")
    rows = []
    for n in names:
        a = agg(data[n], "all")
        rows.append({"model": lab[n], "E RMSE meV/atom": 1e3 * a["e_rmse"], "E MAE": 1e3 * a["e_mae"], "E bias": 1e3 * a["e_bias"],
                     "F SB RMSE eV/A": a[sbdef], "F AW RMSE eV/A": a["aw"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    lab_idx = [i for i, m in enumerate(meta) if m["n_interface"] > 0]
    lab_atoms = sum(meta[i]["n_atoms"] for i in lab_idx)
    print(f"\nREGION FORCE BREAKDOWN: only {len(lab_idx)} of {len(recs)} test structures carry region labels ({lab_atoms} atoms)")
    pc = cdir / "per_configuration.csv"
    if pc.is_file():
        pcd = pd.read_csv(pc)
        src = dict(zip(pcd.record_id.astype(str), pcd.source.astype(str)))
        got = [src.get(Path(str(meta[i]["source_file"])).stem) for i in lab_idx]
        if all(g is not None for g in got):
            print("  labelled structures by source: " + ", ".join(f"{k} {v}" for k, v in pd.Series(got).value_counts().items()))
        else:
            print(f"  (source of the labelled structures could not be matched to {pc.name}: "
                  f"{sum(g is None for g in got)} of {len(got)} ids missing)")
    reg_rows = []
    for r in REGIONS:
        base = agg(data[COMMON], r)
        row = {"region": r, "n_struct": base["n_struct"], "n_atoms": base["n_atoms"],
               "common AW": base["aw"], "common SB": base[sbdef]}
        for n in names[1:]:
            a = agg(data[n], r)
            row[f"{lab[n]} AW"] = a["aw"]
            row[f"{lab[n]} AW %"] = 100 * (a["aw"] - base["aw"]) / base["aw"]
            row[f"{lab[n]} SB"] = a[sbdef]
            row[f"{lab[n]} SB %"] = 100 * (a[sbdef] - base[sbdef]) / base[sbdef]
        reg_rows.append(row)
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(f"{args.out_prefix}_regions.csv", index=False)
    pd.set_option("display.width", 250)
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("  (AW = atom-weighted, SB = structure-balanced force RMSE in eV/A; % = change relative to the common network, negative = better)")

    print(f"\nPAIRED, interface atoms, {len(lab_idx)} labelled structures: per-structure interface force RMSE, full MoE vs common network")
    for n in names[1:]:
        d = []
        for i in lab_idx:
            c, f = data[COMMON][i], data[n][i]
            d.append(np.sqrt(f["sumsq"]["interface"] / (3.0 * f["n"]["interface"])) - np.sqrt(c["sumsq"]["interface"] / (3.0 * c["n"]["interface"])))
        d = np.array(d)
        print(f"  {lab[n]}: lower on {int((d < 0).sum())}, higher on {int((d > 0).sum())} of {len(d)}; "
              f"mean change {d.mean():+.5f} eV/A, largest decrease {d.min():+.5f}, largest increase {d.max():+.5f}")
    print(f"\nsaved {args.out_prefix}_per_structure.csv, {args.out_prefix}_regions.csv")


if __name__ == "__main__":
    main()
