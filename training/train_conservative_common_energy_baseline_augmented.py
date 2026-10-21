#!/usr/bin/env python3
"""
train_conservative_common_energy_baseline.py

Conservative Local MoE v1 -- Stage C1a.

Train the COMMON conservative scalar-energy baseline before adding any MoE
residual heads.

Live model:
    R
      -> differentiable TorchSOAP.compute_per_center()
      -> [390 local TorchSOAP + 2 species one-hot]
      -> fixed train-only feature normalization
      -> atomic residual-energy MLP (392 -> 256 -> 128 -> 1)
      -> E_total = N * e_ref + sum_i e_i^res
      -> F = -dE_total/dR

Targets:
    E_target = MACE total energy
    F_target = MACE forces

Loss (one complete structure per optimizer step):
    L = energy_weight * ((E_pred - E_target)/N)^2
      + force_weight  * mean_components((F_pred - F_target)^2)

The validation/test split remains fixed at 22/22; the training split may include approved short-range augmentations. Validation/test reporting
is structure-balanced.

Important:
  * No routing is used in C1a.
  * No phase/interface labels are inputs.
  * The force loss requires second-order autograd through TorchSOAP.
  * latest_checkpoint.pt is written every epoch and can be resumed with
    --restart_latest.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


EXPERTS = [
    "ML-BKS",
    "Original_BKS",
    "Chik_BKS",
    "PSO1_BKS",
    "PSO2_BKS",
    "Tersoff",
    "Vashishta",
]

SI_Z = 14
O_Z = 8
SI_MASS = 28.0855
O_MASS = 15.9994

REGIONS = ["all", "interface", "phase1_bulk", "phase2_bulk"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--stage_b1_root", type=Path, required=True)
    p.add_argument("--stage_c0_audit", type=Path, required=True)
    p.add_argument("--torchsoap_module", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)

    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])

    p.add_argument("--energy_weight", type=float, default=10.0)
    p.add_argument("--force_weight", type=float, default=1000.0)

    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-7)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--scheduler_patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1.0e-5)

    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--torch_threads", type=int, default=16)

    p.add_argument("--restart_latest", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def import_torchsoap(module_path: Path):
    module_path = module_path.expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(module_path)

    spec = importlib.util.spec_from_file_location(
        "local_moe_torchsoap_c1a_training",
        str(module_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")

    module = importlib.util.module_from_spec(spec)
    # Needed by Python 3.9 dataclasses during dynamic import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_symbols_masses(z: np.ndarray):
    z = np.asarray(z, dtype=np.int64).reshape(-1)
    unknown = sorted(set(z.tolist()) - {SI_Z, O_Z})
    if unknown:
        raise ValueError(f"Unexpected atomic numbers: {unknown}")

    symbols = ["Si" if zi == SI_Z else "O" for zi in z.tolist()]
    masses = np.where(z == SI_Z, SI_MASS, O_MASS).astype(np.float64)
    return symbols, masses


def species_one_hot_torch(
    z: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    zt = torch.as_tensor(
        np.asarray(z, dtype=np.int64),
        device=device,
    )
    return torch.stack(
        [
            (zt == SI_Z).to(dtype=dtype),
            (zt == O_Z).to(dtype=dtype),
        ],
        dim=1,
    )


def call_compute_per_center(
    descriptor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    symbols,
    masses,
):
    method = descriptor.compute_per_center
    sig = inspect.signature(method)

    available = {
        "positions": positions,
        "cell": cell,
        "symbols": symbols,
        "masses": masses,
    }

    kwargs = {}
    missing = []

    for name, param in sig.parameters.items():
        if name in available:
            kwargs[name] = available[name]
        elif param.default is inspect._empty:
            missing.append(name)

    if missing:
        raise TypeError(
            "Unsupported required compute_per_center arguments: "
            f"{missing}; signature={sig}"
        )

    result = method(**kwargs)
    if not torch.is_tensor(result):
        result = torch.as_tensor(result, device=positions.device)

    return result


@dataclass
class StructureRecord:
    source_file: str
    phase_pair: str
    split: str
    n_atoms: int
    feature_file: Path
    source_shard: Path

    atomic_numbers: np.ndarray
    positions_np: np.ndarray
    cell_np: np.ndarray
    phase: np.ndarray
    interface_center: np.ndarray

    target_energy_total_eV: float
    target_forces_np: np.ndarray

    symbols: List[str]
    masses: np.ndarray


class AtomicEnergyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        d = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(d, h),
                nn.LayerNorm(h),
                nn.SiLU(),
            ])
            d = h

        layers.append(nn.Linear(d, 1))
        self.network = nn.Sequential(*layers)

        # Start near the fixed N*e_ref energy baseline.
        last = self.network[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def load_records(stage_b1_root: Path) -> Dict[str, List[StructureRecord]]:
    manifest = read_csv(stage_b1_root / "manifest.csv")

    records: Dict[str, List[StructureRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for row in manifest:
        split = str(row["split"]).strip()
        if split not in records:
            raise RuntimeError(f"Unexpected split {split!r}")

        feature_file = (stage_b1_root / row["output_file"]).resolve()
        source_shard = Path(row["source_shard"]).expanduser().resolve()

        if not feature_file.is_file():
            raise FileNotFoundError(feature_file)
        if not source_shard.is_file():
            raise FileNotFoundError(source_shard)

        with np.load(feature_file, allow_pickle=False) as d:
            z = np.asarray(d["atomic_numbers"], dtype=np.int16).reshape(-1)
            positions = np.asarray(d["positions"], dtype=np.float64)
            cell = np.asarray(d["cell"], dtype=np.float64)
            phase = np.asarray(d["phase"], dtype=np.int16).reshape(-1)
            center = np.asarray(
                d["interface_center"],
                dtype=np.int16,
            ).reshape(-1)

        with np.load(source_shard, allow_pickle=False) as d:
            target_energy = float(d["mace_energy_total_eV"])
            target_forces = np.asarray(
                d["mace_forces"],
                dtype=np.float64,
            )

        n = len(z)
        if n != int(row["n_atoms"]):
            raise RuntimeError(
                f"{row['source_file']}: N mismatch "
                f"{n} != {row['n_atoms']}"
            )
        if positions.shape != (n, 3):
            raise RuntimeError(
                f"{row['source_file']}: bad positions shape {positions.shape}"
            )
        if target_forces.shape != (n, 3):
            raise RuntimeError(
                f"{row['source_file']}: bad forces shape {target_forces.shape}"
            )
        if len(phase) != n or len(center) != n:
            raise RuntimeError(
                f"{row['source_file']}: region-label length mismatch"
            )

        symbols, masses = build_symbols_masses(z)

        records[split].append(
            StructureRecord(
                source_file=str(row["source_file"]),
                phase_pair=str(row["phase_pair"]),
                split=split,
                n_atoms=n,
                feature_file=feature_file,
                source_shard=source_shard,
                atomic_numbers=z,
                positions_np=positions,
                cell_np=cell,
                phase=phase,
                interface_center=center,
                target_energy_total_eV=target_energy,
                target_forces_np=target_forces,
                symbols=symbols,
                masses=masses,
            )
        )

    return records


def compute_feature_normalization(
    train_records: Sequence[StructureRecord],
) -> Tuple[np.ndarray, np.ndarray]:
    n_total = 0
    sx = np.zeros(392, dtype=np.float64)
    sx2 = np.zeros(392, dtype=np.float64)

    for record in train_records:
        with np.load(record.feature_file, allow_pickle=False) as d:
            x = np.asarray(
                d["X_local_torchsoap"],
                dtype=np.float64,
            )

        sx += np.sum(x, axis=0, dtype=np.float64)
        sx2 += np.sum(x * x, axis=0, dtype=np.float64)
        n_total += len(x)

    mean = sx / n_total
    var = np.maximum(
        sx2 / n_total - mean * mean,
        1.0e-12,
    )
    std = np.sqrt(var)
    std = np.where(std < 1.0e-6, 1.0, std)

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def live_features(
    record: StructureRecord,
    positions: torch.Tensor,
    cell: torch.Tensor,
    descriptor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    local_soap = call_compute_per_center(
        descriptor=descriptor,
        positions=positions,
        cell=cell,
        symbols=record.symbols,
        masses=record.masses,
    ).to(dtype=torch.float32)

    onehot = species_one_hot_torch(
        record.atomic_numbers,
        device=positions.device,
        dtype=torch.float32,
    )

    x = torch.cat([local_soap, onehot], dim=1)

    if x.shape != (record.n_atoms, 392):
        raise RuntimeError(
            f"{record.source_file}: live feature shape={x.shape}, "
            f"expected {(record.n_atoms, 392)}"
        )

    return (x - feature_mean) / feature_std


def predict_energy_forces(
    model: nn.Module,
    descriptor,
    record: StructureRecord,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    e_ref: float,
    device: torch.device,
    create_graph: bool,
):
    positions = torch.tensor(
        record.positions_np,
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    cell = torch.tensor(
        record.cell_np,
        dtype=torch.float64,
        device=device,
    )

    x = live_features(
        record=record,
        positions=positions,
        cell=cell,
        descriptor=descriptor,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    atomic_residual = model(x)
    residual_total = atomic_residual.sum().to(dtype=torch.float64)

    energy_total = (
        positions.new_tensor(record.n_atoms * e_ref)
        + residual_total
    )

    forces = -torch.autograd.grad(
        energy_total,
        positions,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )[0]

    return energy_total, forces


def structure_loss(
    energy_total: torch.Tensor,
    forces: torch.Tensor,
    record: StructureRecord,
    energy_weight: float,
    force_weight: float,
    device: torch.device,
):
    target_energy = energy_total.new_tensor(
        record.target_energy_total_eV
    )
    target_forces = torch.tensor(
        record.target_forces_np,
        dtype=torch.float64,
        device=device,
    )

    de_pa = (
        energy_total - target_energy
    ) / record.n_atoms

    force_mse = torch.mean(
        (forces - target_forces) ** 2
    )

    loss = (
        energy_weight * de_pa * de_pa
        + force_weight * force_mse
    )

    return loss, de_pa, force_mse


class ForceRegionAccumulator:
    def __init__(self) -> None:
        self.sse = 0.0
        self.n_components = 0
        self.structure_mse_sum = 0.0
        self.n_structures = 0
        self.n_atoms = 0

    def update(
        self,
        prediction: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return

        diff = (
            np.asarray(prediction[mask], dtype=np.float64)
            - np.asarray(reference[mask], dtype=np.float64)
        )

        self.sse += float(np.sum(diff * diff))
        self.n_components += int(diff.size)
        self.structure_mse_sum += float(np.mean(diff * diff))
        self.n_structures += 1
        self.n_atoms += int(mask.sum())

    def finalize(self) -> Dict[str, float]:
        if self.n_structures == 0:
            return {}

        return {
            "n_structures": self.n_structures,
            "n_atoms": self.n_atoms,
            "atom_weighted_rmse_component_eV_A":
                math.sqrt(self.sse / self.n_components),
            "structure_balanced_rmse_component_eV_A":
                math.sqrt(
                    self.structure_mse_sum / self.n_structures
                ),
        }


def region_masks(record: StructureRecord) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones(record.n_atoms, dtype=bool),
        "interface": record.interface_center == 1,
        "phase1_bulk":
            (record.phase == 1)
            & (record.interface_center == 0),
        "phase2_bulk":
            (record.phase == 2)
            & (record.interface_center == 0),
    }


def train_one_epoch(
    model: nn.Module,
    descriptor,
    train_records: Sequence[StructureRecord],
    optimizer: torch.optim.Optimizer,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    e_ref: float,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
    grad_clip: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    model.train()

    order = rng.permutation(len(train_records))

    loss_values = []
    de_pa_values = []
    force_mse_values = []

    start = time.perf_counter()

    for idx in order:
        record = train_records[int(idx)]

        optimizer.zero_grad(set_to_none=True)

        energy, forces = predict_energy_forces(
            model=model,
            descriptor=descriptor,
            record=record,
            feature_mean=feature_mean,
            feature_std=feature_std,
            e_ref=e_ref,
            device=device,
            create_graph=True,
        )

        loss, de_pa, force_mse = structure_loss(
            energy_total=energy,
            forces=forces,
            record=record,
            energy_weight=energy_weight,
            force_weight=force_weight,
            device=device,
        )

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        loss_values.append(float(loss.detach().cpu()))
        de_pa_values.append(float(de_pa.detach().cpu()))
        force_mse_values.append(
            float(force_mse.detach().cpu())
        )

        # Drop references to large second-order graphs promptly.
        del energy, forces, loss, de_pa, force_mse

    elapsed = time.perf_counter() - start

    return {
        "mean_loss": float(np.mean(loss_values)),
        "energy_rmse_eV_per_atom":
            float(np.sqrt(np.mean(np.square(de_pa_values)))),
        "force_structure_balanced_rmse_eV_A":
            float(np.sqrt(np.mean(force_mse_values))),
        "elapsed_seconds": elapsed,
    }


def evaluate_split(
    model: nn.Module,
    descriptor,
    records: Sequence[StructureRecord],
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    e_ref: float,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
    include_regions: bool,
) -> Tuple[Dict[str, float], List[Dict]]:
    model.eval()

    losses = []
    de_pa_values = []
    abs_de_pa_values = []
    force_mse_values = []
    net_force_norms = []

    region_acc = {
        region: ForceRegionAccumulator()
        for region in REGIONS
    }

    start = time.perf_counter()

    # We need gradient tracking with respect to positions to obtain forces,
    # even during validation/test. create_graph=False prevents second-order
    # graph retention.
    with torch.enable_grad():
        for record in records:
            energy, forces = predict_energy_forces(
                model=model,
                descriptor=descriptor,
                record=record,
                feature_mean=feature_mean,
                feature_std=feature_std,
                e_ref=e_ref,
                device=device,
                create_graph=False,
            )

            target_energy = float(
                record.target_energy_total_eV
            )
            pred_energy = float(
                energy.detach().cpu()
            )
            de_pa = (
                pred_energy - target_energy
            ) / record.n_atoms

            pred_f = (
                forces.detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            ref_f = record.target_forces_np

            force_mse = float(
                np.mean((pred_f - ref_f) ** 2)
            )

            losses.append(
                energy_weight * de_pa * de_pa
                + force_weight * force_mse
            )
            de_pa_values.append(de_pa)
            abs_de_pa_values.append(abs(de_pa))
            force_mse_values.append(force_mse)
            net_force_norms.append(
                float(np.linalg.norm(np.sum(pred_f, axis=0)))
            )

            if include_regions:
                masks = region_masks(record)
                for region, mask in masks.items():
                    region_acc[region].update(
                        pred_f,
                        ref_f,
                        mask,
                    )

            del energy, forces

    elapsed = time.perf_counter() - start

    summary = {
        "n_structures": len(records),
        "n_atoms": int(sum(r.n_atoms for r in records)),
        "mean_combined_loss":
            float(np.mean(losses)),
        "energy_rmse_eV_per_atom":
            float(np.sqrt(
                np.mean(np.square(de_pa_values))
            )),
        "energy_mae_eV_per_atom":
            float(np.mean(abs_de_pa_values)),
        "force_structure_balanced_rmse_eV_A":
            float(np.sqrt(
                np.mean(force_mse_values)
            )),
        "mean_pred_net_force_norm_eV_A":
            float(np.mean(net_force_norms)),
        "max_pred_net_force_norm_eV_A":
            float(np.max(net_force_norms)),
        "elapsed_seconds": elapsed,
    }

    region_rows = []

    if include_regions:
        for region in REGIONS:
            result = region_acc[region].finalize()
            if result:
                row = {"region": region}
                row.update(result)
                region_rows.append(row)

    return summary, region_rows


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    bad_epochs: int,
    feature_mean_np: np.ndarray,
    feature_std_np: np.ndarray,
    args,
    e_ref: float,
    rng: np.random.Generator,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "feature_mean": torch.from_numpy(
                feature_mean_np
            ),
            "feature_std": torch.from_numpy(
                feature_std_np
            ),
            "input_dim": 392,
            "hidden_dims": list(args.hidden_dims),
            "e_ref_eV_per_atom": e_ref,
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "seed": args.seed,
            "rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all":
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            "model_role":
                "C1a common conservative energy baseline",
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    rng: np.random.Generator,
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "rng_state" in checkpoint:
        rng.bit_generator.state = checkpoint["rng_state"]

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(
            checkpoint["torch_rng_state"]
        )

    if (
        torch.cuda.is_available()
        and checkpoint.get("cuda_rng_state_all") is not None
    ):
        torch.cuda.set_rng_state_all(
            checkpoint["cuda_rng_state_all"]
        )

    return checkpoint


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(
        max(1, args.torch_threads)
    )

    device = choose_device(args.device)

    stage_b1_root = (
        args.stage_b1_root
        .expanduser()
        .resolve()
    )
    stage_c0_audit = (
        args.stage_c0_audit
        .expanduser()
        .resolve()
    )
    torchsoap_module = (
        args.torchsoap_module
        .expanduser()
        .resolve()
    )
    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    if not stage_b1_root.is_dir():
        raise FileNotFoundError(stage_b1_root)
    if not stage_c0_audit.is_file():
        raise FileNotFoundError(stage_c0_audit)

    latest_path = (
        output_root / "latest_checkpoint.pt"
    )

    if (
        output_root.exists()
        and any(output_root.iterdir())
        and not args.restart_latest
        and not args.overwrite
    ):
        raise SystemExit(
            f"Output directory is not empty: {output_root}\n"
            "Use --restart_latest to resume or --overwrite to start fresh."
        )

    if (
        args.overwrite
        and output_root.exists()
        and not args.restart_latest
    ):
        import shutil
        shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    c0 = json.loads(
        stage_c0_audit.read_text()
    )
    e_ref = float(
        c0[
            "recommended_train_only_energy_reference_eV_per_atom"
        ]
    )

    records = load_records(stage_b1_root)

    train_records = records["train"]
    val_records = records["val"]
    test_records = records["test"]

    if (
        len(train_records) < 112
        or len(val_records) != 22
        or len(test_records) != 22
    ):
        raise RuntimeError(
            "Unexpected split sizes: "
            f"{len(train_records)}/"
            f"{len(val_records)}/"
            f"{len(test_records)}"
        )

    feature_mean_np, feature_std_np = (
        compute_feature_normalization(
            train_records
        )
    )

    feature_mean = torch.from_numpy(
        feature_mean_np
    ).to(device)
    feature_std = torch.from_numpy(
        feature_std_np
    ).to(device)

    module = import_torchsoap(
        torchsoap_module
    )
    config = module.TorchSOAPConfig(
        species=("Si", "O"),
        r_cut=5.0,
        n_max=6,
        l_max=4,
    )
    descriptor = (
        module.TorchSOAPDescriptor(config)
        .to(device)
    )

    # Descriptor is fixed. It must remain differentiable with respect
    # to positions, but no descriptor parameters are optimized.
    for p in descriptor.parameters():
        p.requires_grad_(False)

    model = AtomicEnergyMLP(
        input_dim=392,
        hidden_dims=args.hidden_dims,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.scheduler_patience,
            min_lr=1.0e-6,
        )
    )

    rng = np.random.default_rng(args.seed)

    start_epoch = 1
    best_epoch = -1
    best_val_loss = float("inf")
    bad_epochs = 0

    history_path = (
        output_root / "training_history.csv"
    )
    history = (
        read_csv(history_path)
        if history_path.is_file()
        else []
    )

    if args.restart_latest:
        if not latest_path.is_file():
            raise FileNotFoundError(
                f"--restart_latest requested but missing {latest_path}"
            )

        checkpoint = restore_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            rng=rng,
        )

        start_epoch = int(
            checkpoint["epoch"]
        ) + 1
        best_epoch = int(
            checkpoint["best_epoch"]
        )
        best_val_loss = float(
            checkpoint["best_val_loss"]
        )
        bad_epochs = int(
            checkpoint["bad_epochs"]
        )

        print(
            f"Restarting from epoch "
            f"{checkpoint['epoch']}; "
            f"next epoch={start_epoch}"
        )

    print("=" * 118)
    print("STAGE C1a: COMMON CONSERVATIVE ENERGY BASELINE")
    print("=" * 118)
    print(f"Stage B1 root:      {stage_b1_root}")
    print(f"Stage C0 audit:     {stage_c0_audit}")
    print(f"TorchSOAP module:   {torchsoap_module}")
    print(f"Output root:        {output_root}")
    print(f"Device:             {device}")
    print(
        "Splits:             "
        f"train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"test={len(test_records)}"
    )
    print(
        f"N atoms range:      "
        f"{min(r.n_atoms for r in train_records + val_records + test_records)}"
        f" - "
        f"{max(r.n_atoms for r in train_records + val_records + test_records)}"
    )
    print(f"e_ref:              {e_ref:+.12f} eV/atom")
    print(f"Architecture:       392 -> {' -> '.join(map(str,args.hidden_dims))} -> 1")
    print(f"Loss weights:       E={args.energy_weight}, F={args.force_weight}")
    print(f"LR:                 {args.lr}")
    print(f"Grad clip:          {args.grad_clip}")
    print(f"Max epochs:         {args.max_epochs}")
    print(f"Patience:           {args.patience}")
    print(f"Start epoch:        {start_epoch}")
    print("=" * 118)

    best_model_path = (
        output_root / "best_common_energy_model.pt"
    )

    for epoch in range(
        start_epoch,
        args.max_epochs + 1,
    ):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_stats = train_one_epoch(
            model=model,
            descriptor=descriptor,
            train_records=train_records,
            optimizer=optimizer,
            feature_mean=feature_mean,
            feature_std=feature_std,
            e_ref=e_ref,
            device=device,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            grad_clip=args.grad_clip,
            rng=rng,
        )

        val_stats, _ = evaluate_split(
            model=model,
            descriptor=descriptor,
            records=val_records,
            feature_mean=feature_mean,
            feature_std=feature_std,
            e_ref=e_ref,
            device=device,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            include_regions=False,
        )

        val_loss = float(
            val_stats["mean_combined_loss"]
        )

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = (
            val_loss
            < best_val_loss - args.min_delta
        )

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "feature_mean":
                        torch.from_numpy(
                            feature_mean_np
                        ),
                    "feature_std":
                        torch.from_numpy(
                            feature_std_np
                        ),
                    "input_dim": 392,
                    "hidden_dims":
                        list(args.hidden_dims),
                    "e_ref_eV_per_atom":
                        e_ref,
                    "energy_weight":
                        args.energy_weight,
                    "force_weight":
                        args.force_weight,
                    "seed":
                        args.seed,
                    "best_epoch":
                        best_epoch,
                    "best_val_loss":
                        best_val_loss,
                    "model_role":
                        "C1a common conservative energy baseline",
                    "torchsoap":
                        {
                            "species": ["Si", "O"],
                            "r_cut_A": 5.0,
                            "n_max": 6,
                            "l_max": 4,
                            "feature_dim": 392,
                        },
                },
                best_model_path,
            )
        else:
            bad_epochs += 1

        peak_mem_gib = (
            torch.cuda.max_memory_allocated()
            / 1024**3
            if device.type == "cuda"
            else float("nan")
        )

        epoch_row = {
            "epoch": epoch,
            "train_mean_loss":
                train_stats["mean_loss"],
            "train_energy_rmse_eV_per_atom":
                train_stats[
                    "energy_rmse_eV_per_atom"
                ],
            "train_force_SB_rmse_eV_A":
                train_stats[
                    "force_structure_balanced_rmse_eV_A"
                ],
            "val_mean_loss":
                val_stats["mean_combined_loss"],
            "val_energy_rmse_eV_per_atom":
                val_stats[
                    "energy_rmse_eV_per_atom"
                ],
            "val_force_SB_rmse_eV_A":
                val_stats[
                    "force_structure_balanced_rmse_eV_A"
                ],
            "lr": lr_now,
            "train_seconds":
                train_stats["elapsed_seconds"],
            "val_seconds":
                val_stats["elapsed_seconds"],
            "peak_cuda_allocated_GiB":
                peak_mem_gib,
            "best_epoch":
                best_epoch,
            "best_val_loss":
                best_val_loss,
        }
        history.append(epoch_row)
        write_csv(history_path, history)

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            bad_epochs=bad_epochs,
            feature_mean_np=feature_mean_np,
            feature_std_np=feature_std_np,
            args=args,
            e_ref=e_ref,
            rng=rng,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train E={train_stats['energy_rmse_eV_per_atom']*1000:8.3f} meV/atom "
            f"F={train_stats['force_structure_balanced_rmse_eV_A']:8.5f} eV/A | "
            f"val E={val_stats['energy_rmse_eV_per_atom']*1000:8.3f} meV/atom "
            f"F={val_stats['force_structure_balanced_rmse_eV_A']:8.5f} eV/A | "
            f"valL={val_loss:10.5f} "
            f"lr={lr_now:.2e} "
            f"mem={peak_mem_gib:.2f} GiB "
            f"{'*' if improved else ''}"
        )

        if bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, "
                f"best val loss={best_val_loss:.8f}"
            )
            break

    if not best_model_path.is_file():
        raise RuntimeError(
            f"Best model was never saved: {best_model_path}"
        )

    best = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        best["model_state_dict"]
    )
    model.eval()

    print("\n" + "=" * 118)
    print("FINAL EVALUATION USING BEST C1a CHECKPOINT")
    print("=" * 118)
    print(
        f"best epoch: {best['best_epoch']}, "
        f"best val loss: {best['best_val_loss']:.8f}"
    )

    split_summaries = []
    test_region_rows = []

    for split_name, split_records in [
        ("train", train_records),
        ("val", val_records),
        ("test", test_records),
    ]:
        summary, region_rows = evaluate_split(
            model=model,
            descriptor=descriptor,
            records=split_records,
            feature_mean=feature_mean,
            feature_std=feature_std,
            e_ref=e_ref,
            device=device,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            include_regions=(split_name == "test"),
        )

        split_summaries.append(
            {"split": split_name, **summary}
        )

        if split_name == "test":
            test_region_rows = [
                {"split": split_name, **row}
                for row in region_rows
            ]

    write_csv(
        output_root / "final_split_metrics.csv",
        split_summaries,
    )
    write_csv(
        output_root / "test_region_force_metrics.csv",
        test_region_rows,
    )

    info = {
        "stage": "C1a",
        "model_role":
            "common conservative energy baseline",
        "best_checkpoint":
            str(best_model_path),
        "best_epoch":
            int(best["best_epoch"]),
        "best_val_loss":
            float(best["best_val_loss"]),
        "e_ref_eV_per_atom":
            e_ref,
        "architecture":
            [392] + list(args.hidden_dims) + [1],
        "energy_weight":
            args.energy_weight,
        "force_weight":
            args.force_weight,
        "important_note":
            (
                "This baseline contains no routing/MoE residual heads. "
                "It establishes what the shared differentiable TorchSOAP "
                "representation can achieve as a single conservative PES."
            ),
    }

    (
        output_root / "training_info.json"
    ).write_text(
        json.dumps(
            info,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        f"\n{'split':<7} {'E_RMSE(meV/atom)':>18} "
        f"{'E_MAE(meV/atom)':>17} {'F_SB_RMSE(eV/A)':>18} "
        f"{'mean |Fnet|':>14} {'max |Fnet|':>14}"
    )
    print("-" * 96)

    for row in split_summaries:
        print(
            f"{row['split']:<7} "
            f"{1000*row['energy_rmse_eV_per_atom']:18.6f} "
            f"{1000*row['energy_mae_eV_per_atom']:17.6f} "
            f"{row['force_structure_balanced_rmse_eV_A']:18.6f} "
            f"{row['mean_pred_net_force_norm_eV_A']:14.3e} "
            f"{row['max_pred_net_force_norm_eV_A']:14.3e}"
        )

    print("\nTEST REGION FORCE METRICS")
    print(
        f"{'region':<13} {'SB_RMSE':>12} "
        f"{'pooled_RMSE':>14} {'Nstruct':>9} {'Natoms':>9}"
    )
    print("-" * 66)

    for row in test_region_rows:
        print(
            f"{row['region']:<13} "
            f"{row['structure_balanced_rmse_component_eV_A']:12.6f} "
            f"{row['atom_weighted_rmse_component_eV_A']:14.6f} "
            f"{row['n_structures']:9d} "
            f"{row['n_atoms']:9d}"
        )

    print("\nSaved:")
    for name in [
        "best_common_energy_model.pt",
        "latest_checkpoint.pt",
        "training_history.csv",
        "final_split_metrics.csv",
        "test_region_force_metrics.csv",
        "training_info.json",
    ]:
        print(f"  {output_root / name}")


if __name__ == "__main__":
    main()
