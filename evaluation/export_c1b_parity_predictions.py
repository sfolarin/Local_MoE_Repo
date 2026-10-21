#!/usr/bin/env python3
"""
export_c1b_parity_predictions.py

Exports raw per-structure energy/force parity data for the FULL, trained
C1b model (frozen router + frozen common C1a + trained residual) -- not
just the standalone common-energy model. Reuses the real loading and
prediction functions from train_c1b_own_oishi_252.py directly (via
import, not reimplemented): load_records, load_common_checkpoint,
load_routing_checkpoint, import_torchsoap, predict_energy_forces, MLP.

Every name/class used here was confirmed directly against the real
source (not guessed): residual_mean/residual_std == common_mean/
common_std (the residual head reuses the common model's own
normalization, confirmed at lines 1277-1278 of the trainer); the
descriptor is built via import_torchsoap() -> TorchSOAPConfig ->
TorchSOAPDescriptor (lines 1279-1291); the residual network is the
plain MLP class with input_dim=392, output_dim=7 (line 1296).

Writes raw_predictions/<split>/<record_id>.npz in the exact schema
export_common_parity_predictions.py already uses, so
plot_relaxed37_c1b_publication.py can read it with zero changes.

Usage:
    python3 export_c1b_parity_predictions.py \
        --c1b-script training/train_c1b_holdout40.py \
        --stage_b1_root <C1b_Stage_Root_v1> \
        --common_checkpoint <C1a_v1/best_checkpoint.pt> \
        --routing_checkpoint <TorchSOAP_Gate_v1/best_torchsoap_routing_gate.pt> \
        --c1b_checkpoint <C1b_v1/best_c1b_residual_model.pt> \
        --torchsoap_module <path/to/torchsoap_autograd_module_periodic_regnorm.py> \
        --residual-hidden-dims 256 128 \
        --out <C1b_Parity_Export_v1>
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c1b-script", required=True, type=Path)
    ap.add_argument("--stage_b1_root", required=True, type=Path)
    ap.add_argument("--common_checkpoint", required=True, type=Path)
    ap.add_argument("--routing_checkpoint", required=True, type=Path)
    ap.add_argument("--c1b_checkpoint", required=True, type=Path,
                     help="best_c1b_residual_model.pt -- the FULL trained C1b model")
    ap.add_argument("--torchsoap_module", required=True, type=Path)
    ap.add_argument("--residual-hidden-dims", type=int, nargs="+", default=[256, 128])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    sys.path.insert(0, str(args.c1b_script.parent))
    c1b = import_module(args.c1b_script, "c1b_own_oishi_252")

    import torch

    device = torch.device("cuda" if (args.device in ("cuda", "auto") and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    records = c1b.load_records(args.stage_b1_root)
    common_model, common_mean, common_std, e_ref, _ = c1b.load_common_checkpoint(
        args.common_checkpoint, device
    )
    # Confirmed signature from source: load_routing_checkpoint(path, device) -- only two args.
    # Confirmed from source: returns (model, mean, std, checkpoint) -- 4 values.
    routing_model, routing_mean, routing_std, _ = c1b.load_routing_checkpoint(
        args.routing_checkpoint, device
    )

    # Confirmed directly from source: residual head reuses common normalization.
    residual_mean, residual_std = common_mean, common_std

    module = c1b.import_torchsoap(args.torchsoap_module)
    config = module.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)
    descriptor = module.TorchSOAPDescriptor(config).to(device)
    for p in descriptor.parameters():
        p.requires_grad_(False)

    residual_model = c1b.MLP(
        input_dim=392, hidden_dims=args.residual_hidden_dims, output_dim=7, dropout=0.0,
    ).to(device)

    best = torch.load(args.c1b_checkpoint, map_location="cpu", weights_only=False)
    residual_model.load_state_dict(best["residual_model_state_dict"])
    residual_model.eval()
    print(f"Loaded C1b checkpoint: best_epoch={best.get('best_epoch')}, "
          f"best_val_loss={best.get('best_val_loss')}")

    out = args.out.expanduser().resolve()
    raw_root = out / "raw_predictions"
    per_config_rows = []

    with torch.enable_grad():
        for split_name, split_records in [("train", records["train"]), ("val", records["val"]), ("test", records["test"])]:
            raw_dir = raw_root / split_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{split_name}: {len(split_records)} structures")

            for record in split_records:
                terms, forces = c1b.predict_energy_forces(
                    record=record, descriptor=descriptor,
                    common_model=common_model, common_mean=common_mean, common_std=common_std,
                    routing_model=routing_model, routing_mean=routing_mean, routing_std=routing_std,
                    residual_model=residual_model, residual_mean=residual_mean, residual_std=residual_std,
                    e_ref=e_ref, device=device, create_graph=False,
                )
                pred_energy = float(terms["total_energy"].detach().cpu())
                ref_energy = float(record.target_energy_total_eV)
                pred_forces = forces.detach().cpu().numpy().astype(np.float64, copy=False)
                ref_forces = record.target_forces_np

                safe_id = str(record.source_file).replace("/", "__")
                np.savez_compressed(
                    raw_dir / f"{safe_id}.npz",
                    record_id=np.asarray(str(record.source_file)),
                    n_atoms=np.asarray(record.n_atoms, dtype=np.int64),
                    E_reference_eV=np.asarray(ref_energy, dtype=np.float64),
                    E_predicted_eV=np.asarray(pred_energy, dtype=np.float64),
                    reference_forces=np.asarray(ref_forces, dtype=np.float64),
                    predicted_forces=np.asarray(pred_forces, dtype=np.float64),
                )
                per_config_rows.append(dict(
                    record_id=record.source_file, split=split_name, n_atoms=record.n_atoms,
                    E_reference_eV=ref_energy, E_predicted_eV=pred_energy,
                    dE_eV_per_atom=(pred_energy - ref_energy) / record.n_atoms,
                ))
                del terms, forces

    config_csv = out / "per_configuration.csv"
    with open(config_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_config_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_config_rows)

    print(f"\nWrote {raw_root} and {config_csv}")
    print(f"Total structures exported: {len(per_config_rows)}")


if __name__ == "__main__":
    main()
