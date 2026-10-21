#!/usr/bin/env python3
"""
export_local_moe_spec.py

Exports everything needed to re-implement the Local-MoE holdout-40 model outside
PyTorch (for the LAMMPS port), and a set of REAL reference cases with the
PyTorch model's own predictions to check the port against.

Written to --out (nothing else on disk is touched, no checkpoint is modified):

  spec/local_moe_spec.npz    all network weights, normalisation vectors, e_ref,
                             radial centres, the 78 channel-pair indices
  spec/local_moe_spec.json   layer-by-layer op lists, descriptor constants,
                             species order, checkpoint / module checksums
  reference/case_XX_N<atoms>.extxyz   the structure
  reference/case_XX_N<atoms>.npz      E and forces of the full model and of the
                             common network alone, per-atom descriptor (float64),
                             per-atom e0, routing weights w_ik, centred residual
                             delta_ik (and the 392-dim feature vector for small cases)
  reference/index.json       list of cases, and how each agrees with the stored
                             parity-export predictions

Gates (the export stops if one fails):
  E1  the standalone recomputation of the energy from the per-atom pieces equals
      the trainer's own total energy (tolerance --tol-self)
  E2  the full model reproduces the stored parity-export predictions of the same
      structure (tolerance --tol-ref, needs --reference-dir; skipped with a
      warning if the stored file is missing)

Run inside a GPU allocation (or on CPU for a quick check):
    python export_local_moe_spec.py --out $PROJ/port_spec_v1 \
        --torchsoap-module <the neighbour-list module>
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
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np

HET_DEFAULT = (_os.environ.get("MOE_HET", "."))
REPO_DEFAULT = str(REPO_ROOT)
DEFAULT_SIZES = "60,90,216,432,648,1134,2160,4608"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def plane_spacings(cell):
    c = np.asarray(cell, float)
    vol = abs(np.linalg.det(c))
    return np.array([vol / np.linalg.norm(np.cross(c[(k + 1) % 3], c[(k + 2) % 3])) for k in range(3)])


def net_ops(torch, model, prefix, arrays):
    """Layer-by-layer description of an MLP, with the weights copied into `arrays`."""
    nn = torch.nn
    ops, n_lin, n_ln = [], 0, 0
    for m in model.network:
        if isinstance(m, nn.Linear):
            arrays[f"{prefix}.L{n_lin}.W"] = m.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"{prefix}.L{n_lin}.b"] = m.bias.detach().cpu().numpy().astype(np.float32)
            ops.append({"op": "linear", "W": f"{prefix}.L{n_lin}.W", "b": f"{prefix}.L{n_lin}.b",
                        "in": int(m.in_features), "out": int(m.out_features)})
            n_lin += 1
        elif isinstance(m, nn.LayerNorm):
            if not m.elementwise_affine or len(m.normalized_shape) != 1:
                raise SystemExit("unsupported LayerNorm configuration")
            arrays[f"{prefix}.LN{n_ln}.w"] = m.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"{prefix}.LN{n_ln}.b"] = m.bias.detach().cpu().numpy().astype(np.float32)
            ops.append({"op": "layernorm", "w": f"{prefix}.LN{n_ln}.w", "b": f"{prefix}.LN{n_ln}.b",
                        "eps": float(m.eps), "dim": int(m.normalized_shape[0])})
            n_ln += 1
        elif isinstance(m, nn.SiLU):
            ops.append({"op": "silu"})
        elif isinstance(m, nn.Dropout):
            ops.append({"op": "dropout", "p": float(m.p), "note": "identity at inference (model.eval())"})
        else:
            raise SystemExit(f"unsupported layer type in {prefix}: {type(m).__name__}")
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--het", default=HET_DEFAULT)
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--c1b-script", type=Path, default=None)
    ap.add_argument("--stage-root", type=Path, default=None)
    ap.add_argument("--common-checkpoint", type=Path, default=None)
    ap.add_argument("--routing-checkpoint", type=Path, default=None)
    ap.add_argument("--c1b-checkpoint", type=Path, default=None)
    ap.add_argument("--torchsoap-module", type=Path, default=None)
    ap.add_argument("--reference-dir", type=Path, default=None,
                    help="raw_predictions/test of the parity export of the same C1b run")
    ap.add_argument("--sizes", default=DEFAULT_SIZES,
                    help="comma list of target atom counts; the nearest distinct test structure is used for each")
    ap.add_argument("--max-feature-atoms", type=int, default=700,
                    help="store the full 392-dim feature matrix only for cases up to this size")
    ap.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    ap.add_argument("--tol-self", type=float, default=1e-6)
    ap.add_argument("--tol-ref", type=float, default=1e-4)
    args = ap.parse_args()

    import torch

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    PROJ = Path(args.het) / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    script = args.c1b_script or REPO_ROOT / 'training/train_c1b_holdout40.py'
    paths = {
        "common": args.common_checkpoint or PROJ / "runs/common_holdout40_ew10000_v1/best_checkpoint.pt",
        "routing": args.routing_checkpoint or PROJ / "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt",
        "c1b": args.c1b_checkpoint or PROJ / "runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt",
        "soap": args.torchsoap_module or PROJ / "data/expanded_holdout40_v1_features/src_snapshot/torchsoap_autograd_module_periodic_regnorm.py",
    }
    paths = {k: Path(v).expanduser().resolve() for k, v in paths.items()}
    stage = (args.stage_root or PROJ / "data/expanded_holdout40_v1_c1b_stage").expanduser().resolve()
    ref_dir = (args.reference_dir or PROJ / "eval_parity_v1/c1b_holdout40_e10000_v1/raw_predictions/test").expanduser().resolve()
    for p in [script, *paths.values()]:
        if not Path(p).is_file():
            raise SystemExit(f"MISSING FILE: {p}")

    out = args.out.expanduser().resolve()
    if (out / "spec" / "local_moe_spec.npz").exists():
        raise SystemExit(f"{out} already holds an export; choose a new --out (nothing is overwritten)")
    (out / "spec").mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(parents=True, exist_ok=True)

    torch.set_default_dtype(torch.float32)
    trainer = import_file(script, "export_c1b_trainer")
    torch.set_default_dtype(torch.float32)

    # ------------------------------------------------------------------ models
    common_model, common_mean, common_std, e_ref, common_ck = trainer.load_common_checkpoint(paths["common"], device)
    routing_model, routing_mean, routing_std, routing_ck = trainer.load_routing_checkpoint(paths["routing"], device)
    bck = torch.load(paths["c1b"], map_location="cpu", weights_only=False)
    if [str(x) for x in bck["expert_names"]] != list(trainer.EXPERTS):
        raise SystemExit("C1b expert order does not match the trainer")
    if abs(float(bck["e_ref_eV_per_atom"]) - e_ref) > 1e-12:
        raise SystemExit("e_ref differs between the common and the C1b checkpoint")
    residual_model = trainer.MLP(input_dim=int(bck["input_dim"]),
                                 hidden_dims=[int(x) for x in bck["residual_hidden_dims"]],
                                 output_dim=7, dropout=0.0).to(device)
    residual_model.load_state_dict(bck["residual_model_state_dict"])
    residual_model.eval()
    residual_mean, residual_std = common_mean, common_std     # as in the trainer / benchmark: the C1b residual head uses the COMMON normalisation

    soap = trainer.import_torchsoap(paths["soap"])
    cfg = soap.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)
    descriptor = soap.TorchSOAPDescriptor(cfg).to(device)
    for p in descriptor.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------ spec
    arrays = {}
    nets = {
        "common": {"role": "per-atom common energy e0_i = H0((x_i - mean_c)/std_c)", "normalisation": "common",
                   "ops": net_ops(torch, common_model, "common", arrays)},
        "routing": {"role": "logits g_i = G((x_i - mean_r)/std_r); w_ik = softmax_k(g_ik)", "normalisation": "routing",
                    "ops": net_ops(torch, routing_model, "routing", arrays)},
        "residual": {"role": "raw_delta_i = Hres((x_i - mean_c)/std_c); delta_ik = raw_delta_ik - mean_k(raw_delta_ik)",
                     "normalisation": "common", "ops": net_ops(torch, residual_model, "residual", arrays)},
    }
    arrays["common.feature_mean"] = common_mean.detach().cpu().numpy().astype(np.float32)
    arrays["common.feature_std"] = common_std.detach().cpu().numpy().astype(np.float32)
    arrays["routing.feature_mean"] = routing_mean.detach().cpu().numpy().astype(np.float32)
    arrays["routing.feature_std"] = routing_std.detach().cpu().numpy().astype(np.float32)
    arrays["radial_centers"] = descriptor.radial_centers.detach().cpu().numpy().astype(np.float64)
    arrays["pair_i"] = descriptor.pair_i.detach().cpu().numpy().astype(np.int64)
    arrays["pair_j"] = descriptor.pair_j.detach().cpu().numpy().astype(np.int64)
    arrays["e_ref_eV_per_atom"] = np.asarray(e_ref, dtype=np.float64)

    d_mean = float(np.max(np.abs(arrays["routing.feature_mean"] - arrays["common.feature_mean"])))
    d_std = float(np.max(np.abs(arrays["routing.feature_std"] - arrays["common.feature_std"])))
    print(f"routing vs common normalisation: max|d mean| = {d_mean:.3e}, max|d std| = {d_std:.3e}")
    print(f"smallest std entry: {float(arrays['common.feature_std'].min()):.3e} "
          f"(count at <= 1e-5: {int((arrays['common.feature_std'] <= 1e-5).sum())})")

    meta = {
        "model": "Local-MoE holdout-40, stage C1b (frozen routing parameters, conservative residual)",
        "energy": "E = N*e_ref + sum_i e0_i + sum_i sum_k w_ik * delta_ik",
        "forces": "F = -dE/dR, including dw_ik/dR (routing parameters are frozen, the weights are NOT detached from R)",
        "feature_vector": "x_i = [TorchSOAP power spectrum (390, float64 -> cast to float32), isSi, isO] (392, float32)",
        "species_order": ["Si", "O"], "atomic_numbers": [14, 8],
        "expert_names": list(trainer.EXPERTS),
        "expert_note": "the 7 channels carry the names of the classical potentials the router was distilled from; "
                       "no classical potential is evaluated at inference",
        "descriptor": {
            "r_cut": cfg.r_cut, "n_max": cfg.n_max, "l_max": cfg.l_max, "radial_sigma": cfg.radial_sigma,
            "eps": cfg.eps, "normalize_power": cfg.normalize_power, "power_norm_lambda": cfg.power_norm_lambda,
            "n_channels": len(cfg.species) * cfg.n_max, "soap_dim": int(descriptor.soap_dim),
            "pbc": "fully periodic in all three directions, every image within r_cut is a neighbour; only the "
                   "home-cell self pair (i==i, shift 0) is excluded",
            "precision": "positions, cell and descriptor in float64",
        },
        "networks": nets,
        "normalisation_difference_routing_vs_common": {"max_abs_mean": d_mean, "max_abs_std": d_std},
        "precision": "networks and normalisation float32; positions/cell/descriptor and the final sums float64",
        "provenance": {
            "checkpoints": {k: {"path": str(paths[k]), "sha256": sha256(paths[k])} for k in ("common", "routing", "c1b")},
            "torchsoap_module": {"path": str(paths["soap"]), "sha256": sha256(paths["soap"])},
            "trainer_script": {"path": str(script), "sha256": sha256(script)},
            "torch": torch.__version__, "python": platform.python_version(), "device": str(device),
        },
    }
    np.savez(out / "spec" / "local_moe_spec.npz", **arrays)
    (out / "spec" / "local_moe_spec.json").write_text(json.dumps(meta, indent=2))
    shutil.copy2(paths["soap"], out / "spec" / paths["soap"].name)      # the descriptor source, for reading
    print(f"spec written: {len(arrays)} arrays; descriptor source copied: {paths['soap'].name}")

    # ------------------------------------------------------------------ reference cases
    test = sorted(trainer.load_records(stage)["test"], key=lambda r: (r.n_atoms, r.source_file))
    chosen, used = [], set()
    for target in [int(s) for s in args.sizes.split(",") if s.strip()]:
        order = sorted(range(len(test)), key=lambda k: (abs(test[k].n_atoms - target), k))
        for k in order:
            if k not in used:
                used.add(k)
                chosen.append(test[k])
                break
    chosen.sort(key=lambda r: (r.n_atoms, r.source_file))

    from ase import Atoms
    from ase.io import write

    def pieces(rec):
        """Everything the trainer computes, plus the pieces, for one record."""
        pos = torch.tensor(rec.positions_np, dtype=torch.float64, device=device, requires_grad=True)
        cell = torch.tensor(rec.cell_np, dtype=torch.float64, device=device)
        with torch.enable_grad():
            terms, forces = trainer.predict_energy_forces(
                record=rec, descriptor=descriptor, common_model=common_model, common_mean=common_mean,
                common_std=common_std, routing_model=routing_model, routing_mean=routing_mean,
                routing_std=routing_std, residual_model=residual_model, residual_mean=residual_mean,
                residual_std=residual_std, e_ref=e_ref, device=device, create_graph=False)
            # common network alone (no routing, no residual)
            x = trainer.live_raw_features(record=rec, positions=pos, cell=cell, descriptor=descriptor)
            e0 = common_model((x - common_mean) / common_std).squeeze(-1)
            e_common = pos.new_tensor(rec.n_atoms * e_ref) + e0.sum().to(dtype=torch.float64)
            f_common = -torch.autograd.grad(e_common, pos, allow_unused=False)[0]
        with torch.no_grad():
            w = torch.softmax(routing_model((x - routing_mean) / routing_std), dim=1)
            raw = residual_model((x - residual_mean) / residual_std)
            delta = raw - raw.mean(dim=1, keepdim=True)
            soap64 = trainer.call_compute_per_center(descriptor, pos.detach(), cell, rec.symbols, rec.masses)
        # recompute the total from the pieces (gate E1)
        e_pieces = (rec.n_atoms * e_ref + float(e0.detach().sum().to(torch.float64))
                    + float(torch.sum(w * delta).to(torch.float64)))
        return dict(E_full=float(terms["total_energy"].detach().cpu()), F_full=forces.detach().cpu().numpy(),
                    E_common=float(e_common.detach().cpu()), F_common=f_common.detach().cpu().numpy(),
                    e0=e0.detach().cpu().numpy(), w=w.detach().cpu().numpy(), delta=delta.detach().cpu().numpy(),
                    x=x.detach().cpu().numpy(), soap64=soap64.detach().cpu().numpy(), E_pieces=e_pieces,
                    E_moe_term=float(terms["moe_residual_total"].detach().cpu()))

    index = []
    worst_self = worst_e = worst_f = 0.0
    n_ref_missing = 0
    for ci, rec in enumerate(chosen):
        r = pieces(rec)
        d_self = abs(r["E_pieces"] - r["E_full"])
        worst_self = max(worst_self, d_self)
        stem = f"case_{ci:02d}_N{rec.n_atoms}"
        atoms = Atoms(numbers=np.asarray(rec.atomic_numbers).astype(int), positions=np.asarray(rec.positions_np, float),
                      cell=np.asarray(rec.cell_np, float), pbc=True)
        write(out / "reference" / f"{stem}.extxyz", atoms, format="extxyz")
        save = dict(record_id=np.asarray(rec.source_file), numbers=np.asarray(rec.atomic_numbers).astype(np.int64),
                    positions=np.asarray(rec.positions_np, float), cell=np.asarray(rec.cell_np, float),
                    E_full=np.asarray(r["E_full"]), F_full=r["F_full"], E_common=np.asarray(r["E_common"]),
                    F_common=r["F_common"], E_moe_term=np.asarray(r["E_moe_term"]),
                    e0=r["e0"], w=r["w"], delta=r["delta"], soap64=r["soap64"])
        if rec.n_atoms <= args.max_feature_atoms:
            save["x"] = r["x"]
        np.savez(out / "reference" / f"{stem}.npz", **save)

        entry = {"case": stem, "record_id": str(rec.source_file), "n_atoms": int(rec.n_atoms),
                 "min_plane_spacing_A": float(plane_spacings(rec.cell_np).min()),
                 "E_full_eV": r["E_full"], "E_common_eV": r["E_common"], "E_moe_term_eV": r["E_moe_term"],
                 "max_abs_F_full": float(np.abs(r["F_full"]).max()),
                 "self_consistency_dE_eV": d_self, "features_stored": bool(rec.n_atoms <= args.max_feature_atoms)}
        refp = ref_dir / (str(rec.source_file).replace("/", "__") + ".npz")
        if refp.is_file():
            with np.load(refp, allow_pickle=False) as z:
                dE = abs(r["E_full"] - float(z["E_predicted_eV"]))
                dF = float(np.abs(r["F_full"] - np.asarray(z["predicted_forces"], float)).max())
            entry.update(vs_stored_parity_dE_eV=dE, vs_stored_parity_dF_eV_per_A=dF)
            worst_e, worst_f = max(worst_e, dE), max(worst_f, dF)
        else:
            n_ref_missing += 1
            entry["vs_stored_parity"] = "stored file missing"
        index.append(entry)
        print(f"  {stem:22s} E_full={r['E_full']:+.6f} eV  self dE={d_self:.2e}"
              + (f"  vs stored: dE={entry['vs_stored_parity_dE_eV']:.2e} dF={entry['vs_stored_parity_dF_eV_per_A']:.2e}"
                 if "vs_stored_parity_dE_eV" in entry else "  (no stored prediction)"))

    (out / "reference" / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nE1 standalone recomputation of the total from per-atom pieces: worst |dE| = {worst_self:.2e} eV (tolerance {args.tol_self:g})")
    if n_ref_missing < len(chosen):
        print(f"E2 full model vs stored parity-export predictions: worst |dE| = {worst_e:.2e} eV, "
              f"worst |dF| = {worst_f:.2e} eV/A (tolerance {args.tol_ref:g})")
    else:
        print("E2 skipped: no stored parity-export prediction found for any case")
    bad = []
    if worst_self > args.tol_self:
        bad.append("E1")
    if n_ref_missing < len(chosen) and (worst_e > args.tol_ref or worst_f > args.tol_ref):
        bad.append("E2")
    if bad:
        raise SystemExit(f"GATE {'/'.join(bad)} FAILED - the files above are NOT a valid reference; do not hand them over")
    print(f"\nexport complete: {out}")


if __name__ == "__main__":
    main()
