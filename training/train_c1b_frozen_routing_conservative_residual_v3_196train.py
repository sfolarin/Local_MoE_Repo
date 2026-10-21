#!/usr/bin/env python3
"""
train_c1b_frozen_routing_conservative_residual.py

Conservative Local MoE v1 -- Stage C1b.

Purpose
-------
Train the first conservative Local-MoE correction on top of the validated C1a
common PES while keeping both:
  * the C1a common-energy parameters frozen, and
  * the B2 distilled routing-gate parameters frozen.

IMPORTANT: "frozen routing" means frozen PARAMETERS only. The routing weights
remain live functions of coordinates:

    w_ik = softmax(G(X_i(R)))

and are NOT detached from R. Therefore the conservative force contains the
required gate-gradient contribution d w_ik / dR.

Model
-----
Raw local differentiable features:
    X_i(R) = [TorchSOAP_390, species_onehot_2]

Frozen common energy:
    e_i^0 = H0(X_i)

Frozen local routing:
    w_ik = softmax(G(X_i))

Trainable seven residual heads:
    raw_delta_e_ik = Hres(X_i)

Per-atom expert centering removes the additive expert-direction gauge:
    delta_e_ik = raw_delta_e_ik - mean_k(raw_delta_e_ik)

Total energy:
    E(R) = N * e_ref
           + sum_i e_i^0
           + sum_i sum_k w_ik(R) delta_e_ik(R)

Forces:
    F = -dE/dR

Targets:
    MACE total energies and MACE forces.

Training
--------
One complete structure per optimizer step => structure-balanced training.

Loss:
    L = energy_weight * ((E_pred - E_MACE)/N)^2
      + force_weight  * mean_components((F_pred - F_MACE)^2)

The residual output layer is initialized EXACTLY to zero. Therefore the initial
C1b model is exactly the validated C1a model:
    E_C1b(t=0) == E_C1a
    F_C1b(t=0) == F_C1a

Checkpointing
-------------
latest_checkpoint.pt is saved every epoch.
Use --restart_latest after an allocation timeout.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import random
import shutil
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
    p.add_argument("--common_checkpoint", type=Path, required=True)
    p.add_argument("--routing_checkpoint", type=Path, required=True)
    p.add_argument("--torchsoap_module", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)

    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--residual_hidden_dims", type=int, nargs="+", default=[256, 128])

    p.add_argument("--energy_weight", type=float, default=10.0)
    p.add_argument("--force_weight", type=float, default=1000.0)

    p.add_argument("--lr", type=float, default=2.5e-4)
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
            raise RuntimeError("--device cuda requested but CUDA unavailable.")
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
        "local_moe_torchsoap_c1b_training",
        str(module_path),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")

    module = importlib.util.module_from_spec(spec)
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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
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
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h

        layers.append(nn.Linear(d, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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
                f"{row['source_file']}: N mismatch {n} != {row['n_atoms']}"
            )
        if positions.shape != (n, 3):
            raise RuntimeError(
                f"{row['source_file']}: bad positions shape {positions.shape}"
            )
        if target_forces.shape != (n, 3):
            raise RuntimeError(
                f"{row['source_file']}: bad forces shape {target_forces.shape}"
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


def load_common_checkpoint(
    path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )

    model = MLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=[int(x) for x in checkpoint["hidden_dims"]],
        output_dim=1,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    mean = checkpoint["feature_mean"].to(
        device=device,
        dtype=torch.float32,
    )
    std = checkpoint["feature_std"].to(
        device=device,
        dtype=torch.float32,
    )

    e_ref = float(checkpoint["e_ref_eV_per_atom"])

    return model, mean, std, e_ref, checkpoint


def load_routing_checkpoint(
    path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )

    names = [str(x) for x in checkpoint["expert_names"]]
    if names != EXPERTS:
        raise RuntimeError(
            f"Routing expert-order mismatch: {names}"
        )

    model = MLP(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=[int(x) for x in checkpoint["hidden_dims"]],
        output_dim=7,
        dropout=float(checkpoint["dropout"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    mean = checkpoint["feature_mean"].to(
        device=device,
        dtype=torch.float32,
    )
    std = checkpoint["feature_std"].to(
        device=device,
        dtype=torch.float32,
    )

    return model, mean, std, checkpoint


def live_raw_features(
    record: StructureRecord,
    positions: torch.Tensor,
    cell: torch.Tensor,
    descriptor,
) -> torch.Tensor:
    soap = call_compute_per_center(
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

    x = torch.cat([soap, onehot], dim=1)

    if x.shape != (record.n_atoms, 392):
        raise RuntimeError(
            f"{record.source_file}: live feature shape={x.shape}, "
            f"expected {(record.n_atoms, 392)}"
        )

    return x


def model_terms(
    *,
    record: StructureRecord,
    positions: torch.Tensor,
    cell: torch.Tensor,
    descriptor,
    common_model: nn.Module,
    common_mean: torch.Tensor,
    common_std: torch.Tensor,
    routing_model: nn.Module,
    routing_mean: torch.Tensor,
    routing_std: torch.Tensor,
    residual_model: nn.Module,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    e_ref: float,
):
    x = live_raw_features(
        record=record,
        positions=positions,
        cell=cell,
        descriptor=descriptor,
    )

    common_atomic = common_model(
        (x - common_mean) / common_std
    ).squeeze(-1)

    routing_logits = routing_model(
        (x - routing_mean) / routing_std
    )
    weights = torch.softmax(routing_logits, dim=1)

    raw_delta = residual_model(
        (x - residual_mean) / residual_std
    )

    # Remove the per-atom additive expert-direction gauge.
    centered_delta = (
        raw_delta
        - raw_delta.mean(dim=1, keepdim=True)
    )

    common_residual_total = (
        common_atomic.sum().to(dtype=torch.float64)
    )

    moe_residual_total = (
        torch.sum(weights * centered_delta)
        .to(dtype=torch.float64)
    )

    total_energy = (
        positions.new_tensor(record.n_atoms * e_ref)
        + common_residual_total
        + moe_residual_total
    )

    return {
        "total_energy": total_energy,
        "common_residual_total": common_residual_total,
        "moe_residual_total": moe_residual_total,
        "weights": weights,
        "centered_delta": centered_delta,
    }


def predict_energy_forces(
    *,
    record: StructureRecord,
    descriptor,
    common_model: nn.Module,
    common_mean: torch.Tensor,
    common_std: torch.Tensor,
    routing_model: nn.Module,
    routing_mean: torch.Tensor,
    routing_std: torch.Tensor,
    residual_model: nn.Module,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
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

    terms = model_terms(
        record=record,
        positions=positions,
        cell=cell,
        descriptor=descriptor,
        common_model=common_model,
        common_mean=common_mean,
        common_std=common_std,
        routing_model=routing_model,
        routing_mean=routing_mean,
        routing_std=routing_std,
        residual_model=residual_model,
        residual_mean=residual_mean,
        residual_std=residual_std,
        e_ref=e_ref,
    )

    forces = -torch.autograd.grad(
        terms["total_energy"],
        positions,
        create_graph=create_graph,
        retain_graph=create_graph,
        allow_unused=False,
    )[0]

    return terms, forces


def structure_loss(
    *,
    terms,
    forces: torch.Tensor,
    record: StructureRecord,
    energy_weight: float,
    force_weight: float,
    device: torch.device,
):
    target_energy = terms["total_energy"].new_tensor(
        record.target_energy_total_eV
    )

    target_forces = torch.tensor(
        record.target_forces_np,
        dtype=torch.float64,
        device=device,
    )

    de_pa = (
        terms["total_energy"] - target_energy
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
        self.structure_mse_sum += float(
            np.mean(diff * diff)
        )
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
    *,
    residual_model: nn.Module,
    descriptor,
    common_model: nn.Module,
    common_mean: torch.Tensor,
    common_std: torch.Tensor,
    routing_model: nn.Module,
    routing_mean: torch.Tensor,
    routing_std: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    train_records: Sequence[StructureRecord],
    optimizer: torch.optim.Optimizer,
    e_ref: float,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
    grad_clip: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    residual_model.train()
    common_model.eval()
    routing_model.eval()

    order = rng.permutation(len(train_records))

    loss_values = []
    de_pa_values = []
    force_mse_values = []
    moe_residual_epa_values = []

    start = time.perf_counter()

    for idx in order:
        record = train_records[int(idx)]

        optimizer.zero_grad(set_to_none=True)

        terms, forces = predict_energy_forces(
            record=record,
            descriptor=descriptor,
            common_model=common_model,
            common_mean=common_mean,
            common_std=common_std,
            routing_model=routing_model,
            routing_mean=routing_mean,
            routing_std=routing_std,
            residual_model=residual_model,
            residual_mean=residual_mean,
            residual_std=residual_std,
            e_ref=e_ref,
            device=device,
            create_graph=True,
        )

        loss, de_pa, force_mse = structure_loss(
            terms=terms,
            forces=forces,
            record=record,
            energy_weight=energy_weight,
            force_weight=force_weight,
            device=device,
        )

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                residual_model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        loss_values.append(
            float(loss.detach().cpu())
        )
        de_pa_values.append(
            float(de_pa.detach().cpu())
        )
        force_mse_values.append(
            float(force_mse.detach().cpu())
        )
        moe_residual_epa_values.append(
            float(
                (
                    terms["moe_residual_total"]
                    / record.n_atoms
                )
                .detach()
                .cpu()
            )
        )

        del terms, forces, loss, de_pa, force_mse

    elapsed = time.perf_counter() - start

    return {
        "mean_loss":
            float(np.mean(loss_values)),
        "energy_rmse_eV_per_atom":
            float(
                np.sqrt(
                    np.mean(np.square(de_pa_values))
                )
            ),
        "force_structure_balanced_rmse_eV_A":
            float(
                np.sqrt(
                    np.mean(force_mse_values)
                )
            ),
        "moe_residual_energy_per_atom_rms_eV":
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            moe_residual_epa_values
                        )
                    )
                )
            ),
        "elapsed_seconds": elapsed,
    }


def evaluate_split(
    *,
    residual_model: nn.Module,
    descriptor,
    common_model: nn.Module,
    common_mean: torch.Tensor,
    common_std: torch.Tensor,
    routing_model: nn.Module,
    routing_mean: torch.Tensor,
    routing_std: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    records: Sequence[StructureRecord],
    e_ref: float,
    device: torch.device,
    energy_weight: float,
    force_weight: float,
    include_regions: bool,
) -> Tuple[Dict[str, float], List[Dict]]:
    residual_model.eval()
    common_model.eval()
    routing_model.eval()

    losses = []
    de_pa_values = []
    abs_de_pa_values = []
    force_mse_values = []
    net_force_norms = []
    moe_residual_epa_values = []

    region_acc = {
        region: ForceRegionAccumulator()
        for region in REGIONS
    }

    start = time.perf_counter()

    with torch.enable_grad():
        for record in records:
            terms, forces = predict_energy_forces(
                record=record,
                descriptor=descriptor,
                common_model=common_model,
                common_mean=common_mean,
                common_std=common_std,
                routing_model=routing_model,
                routing_mean=routing_mean,
                routing_std=routing_std,
                residual_model=residual_model,
                residual_mean=residual_mean,
                residual_std=residual_std,
                e_ref=e_ref,
                device=device,
                create_graph=False,
            )

            pred_energy = float(
                terms["total_energy"]
                .detach()
                .cpu()
            )
            target_energy = float(
                record.target_energy_total_eV
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
                np.mean(
                    (pred_f - ref_f) ** 2
                )
            )

            losses.append(
                energy_weight * de_pa * de_pa
                + force_weight * force_mse
            )
            de_pa_values.append(de_pa)
            abs_de_pa_values.append(abs(de_pa))
            force_mse_values.append(force_mse)
            net_force_norms.append(
                float(
                    np.linalg.norm(
                        np.sum(pred_f, axis=0)
                    )
                )
            )
            moe_residual_epa_values.append(
                float(
                    (
                        terms["moe_residual_total"]
                        / record.n_atoms
                    )
                    .detach()
                    .cpu()
                )
            )

            if include_regions:
                masks = region_masks(record)
                for region, mask in masks.items():
                    region_acc[region].update(
                        pred_f,
                        ref_f,
                        mask,
                    )

            del terms, forces

    elapsed = time.perf_counter() - start

    summary = {
        "n_structures": len(records),
        "n_atoms":
            int(sum(r.n_atoms for r in records)),
        "mean_combined_loss":
            float(np.mean(losses)),
        "energy_rmse_eV_per_atom":
            float(
                np.sqrt(
                    np.mean(
                        np.square(de_pa_values)
                    )
                )
            ),
        "energy_mae_eV_per_atom":
            float(np.mean(abs_de_pa_values)),
        "force_structure_balanced_rmse_eV_A":
            float(
                np.sqrt(
                    np.mean(force_mse_values)
                )
            ),
        "moe_residual_energy_per_atom_rms_eV":
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            moe_residual_epa_values
                        )
                    )
                )
            ),
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
    residual_model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    bad_epochs: int,
    args,
    e_ref: float,
    rng: np.random.Generator,
) -> None:
    torch.save(
        {
            "residual_model_state_dict":
                residual_model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "scheduler_state_dict":
                scheduler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "input_dim": 392,
            "residual_hidden_dims":
                list(args.residual_hidden_dims),
            "expert_names": EXPERTS,
            "e_ref_eV_per_atom": e_ref,
            "energy_weight":
                args.energy_weight,
            "force_weight":
                args.force_weight,
            "seed": args.seed,
            "rng_state":
                rng.bit_generator.state,
            "torch_rng_state":
                torch.get_rng_state(),
            "cuda_rng_state_all":
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            "common_checkpoint":
                str(
                    args.common_checkpoint
                    .expanduser()
                    .resolve()
                ),
            "routing_checkpoint":
                str(
                    args.routing_checkpoint
                    .expanduser()
                    .resolve()
                ),
            "model_role":
                "C1b frozen-routing conservative Local-MoE residual",
        },
        path,
    )


def restore_checkpoint(
    path: Path,
    *,
    residual_model: nn.Module,
    optimizer,
    scheduler,
    rng: np.random.Generator,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    residual_model.load_state_dict(
        checkpoint["residual_model_state_dict"]
    )
    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "rng_state" in checkpoint:
        rng.bit_generator.state = (
            checkpoint["rng_state"]
        )

    if "torch_rng_state" in checkpoint:
        state = checkpoint["torch_rng_state"]
        if torch.is_tensor(state):
            state = (
                state.detach()
                .cpu()
                .to(torch.uint8)
            )
        torch.set_rng_state(state)

    if (
        torch.cuda.is_available()
        and checkpoint.get(
            "cuda_rng_state_all"
        ) is not None
    ):
        states = []
        for state in checkpoint[
            "cuda_rng_state_all"
        ]:
            if torch.is_tensor(state):
                state = (
                    state.detach()
                    .cpu()
                    .to(torch.uint8)
                )
            states.append(state)

        ndev = torch.cuda.device_count()
        if ndev > 0:
            torch.cuda.set_rng_state_all(
                states[:ndev]
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
    common_checkpoint_path = (
        args.common_checkpoint
        .expanduser()
        .resolve()
    )
    routing_checkpoint_path = (
        args.routing_checkpoint
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
        shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_records(stage_b1_root)

    train_records = records["train"]
    val_records = records["val"]
    test_records = records["test"]

    if (
        len(train_records),
        len(val_records),
        len(test_records),
    ) != (196, 22, 22):
        raise RuntimeError(
            "Unexpected split sizes: "
            f"{len(train_records)}/"
            f"{len(val_records)}/"
            f"{len(test_records)}"
        )

    common_model, common_mean, common_std, e_ref, common_ck = (
        load_common_checkpoint(
            common_checkpoint_path,
            device,
        )
    )

    routing_model, routing_mean, routing_std, routing_ck = (
        load_routing_checkpoint(
            routing_checkpoint_path,
            device,
        )
    )

    # Both were normalized on the same Stage-B1 training split. Keep the
    # common normalization for the residual head and report any discrepancy.
    mean_diff = float(
        torch.max(
            torch.abs(
                common_mean - routing_mean
            )
        ).detach().cpu()
    )
    std_diff = float(
        torch.max(
            torch.abs(
                common_std - routing_std
            )
        ).detach().cpu()
    )

    residual_mean = common_mean
    residual_std = common_std

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
    for p in descriptor.parameters():
        p.requires_grad_(False)

    residual_model = MLP(
        input_dim=392,
        hidden_dims=args.residual_hidden_dims,
        output_dim=7,
        dropout=0.0,
    ).to(device)

    # Critical: C1b starts EXACTLY from C1a.
    with torch.no_grad():
        final_layer = residual_model.network[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    optimizer = torch.optim.AdamW(
        residual_model.parameters(),
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
            residual_model=residual_model,
            optimizer=optimizer,
            scheduler=scheduler,
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

    print("=" * 122)
    print("STAGE C1b: FROZEN-ROUTING CONSERVATIVE LOCAL-MoE RESIDUAL")
    print("=" * 122)
    print(f"Stage B1 root:        {stage_b1_root}")
    print(f"C1a checkpoint:       {common_checkpoint_path}")
    print(f"B2 routing checkpoint:{routing_checkpoint_path}")
    print(f"TorchSOAP module:     {torchsoap_module}")
    print(f"Output root:          {output_root}")
    print(f"Device:               {device}")
    print(
        "Splits:               "
        f"train={len(train_records)}, "
        f"val={len(val_records)}, "
        f"test={len(test_records)}"
    )
    print(
        f"N atoms range:        "
        f"{min(r.n_atoms for r in train_records + val_records + test_records)}"
        f" - "
        f"{max(r.n_atoms for r in train_records + val_records + test_records)}"
    )
    print(f"e_ref:                {e_ref:+.12f} eV/atom")
    print(
        "Residual architecture:"
        f" 392 -> "
        f"{' -> '.join(map(str,args.residual_hidden_dims))}"
        " -> 7"
    )
    print("Residual centering:   per atom across 7 expert channels")
    print("C1a parameters:       frozen")
    print("Routing parameters:   frozen")
    print("Routing vs R:         LIVE (NOT detached)")
    print(f"Loss weights:         E={args.energy_weight}, F={args.force_weight}")
    print(f"Initial LR:           {args.lr}")
    print(f"Grad clip:            {args.grad_clip}")
    print(f"Max epochs:           {args.max_epochs}")
    print(f"Patience:             {args.patience}")
    print(f"Start epoch:          {start_epoch}")
    print(f"max |common_mean-route_mean|: {mean_diff:.3e}")
    print(f"max |common_std-route_std|:   {std_diff:.3e}")
    print("=" * 122)

    # Exact-zero initialization audit before training only.
    if start_epoch == 1:
        sample = train_records[0]

        with torch.enable_grad():
            terms0, forces0 = predict_energy_forces(
                record=sample,
                descriptor=descriptor,
                common_model=common_model,
                common_mean=common_mean,
                common_std=common_std,
                routing_model=routing_model,
                routing_mean=routing_mean,
                routing_std=routing_std,
                residual_model=residual_model,
                residual_mean=residual_mean,
                residual_std=residual_std,
                e_ref=e_ref,
                device=device,
                create_graph=False,
            )

        zero_corr = float(
            abs(
                terms0["moe_residual_total"]
                .detach()
                .cpu()
            )
        )

        print(
            "\nZERO-INITIALIZATION AUDIT: "
            f"|MoE residual total energy|={zero_corr:.6e} eV"
        )

        if zero_corr != 0.0:
            raise RuntimeError(
                "Residual zero initialization did not reproduce exact C1a."
            )

        del terms0, forces0

    best_model_path = (
        output_root
        / "best_c1b_residual_model.pt"
    )

    # On a fresh run, register the exact C1a-equivalent zero-residual model
    # as epoch 0. This makes C1a a hard validation baseline: C1b is allowed to
    # replace it only when the validation loss genuinely improves.
    if start_epoch == 1:
        baseline_val, _ = evaluate_split(
            residual_model=residual_model,
            descriptor=descriptor,
            common_model=common_model,
            common_mean=common_mean,
            common_std=common_std,
            routing_model=routing_model,
            routing_mean=routing_mean,
            routing_std=routing_std,
            residual_mean=residual_mean,
            residual_std=residual_std,
            records=val_records,
            e_ref=e_ref,
            device=device,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            include_regions=False,
        )

        best_val_loss = float(baseline_val["mean_combined_loss"])
        best_epoch = 0
        bad_epochs = 0

        torch.save(
            {
                "residual_model_state_dict":
                    residual_model.state_dict(),
                "input_dim": 392,
                "residual_hidden_dims":
                    list(args.residual_hidden_dims),
                "expert_names": EXPERTS,
                "e_ref_eV_per_atom": e_ref,
                "energy_weight":
                    args.energy_weight,
                "force_weight":
                    args.force_weight,
                "seed": args.seed,
                "best_epoch": 0,
                "best_val_loss": best_val_loss,
                "common_checkpoint":
                    str(common_checkpoint_path),
                "routing_checkpoint":
                    str(routing_checkpoint_path),
                "model_role":
                    "C1b frozen-routing conservative Local-MoE residual",
                "residual_centering":
                    "delta_ik = raw_delta_ik - mean_k(raw_delta_ik)",
                "routing_coordinate_gradient":
                    "enabled; routing parameters frozen, weights not detached",
                "note":
                    "Epoch 0 is the exact C1a baseline (zero residual correction).",
            },
            best_model_path,
        )

        baseline_row = {
            "epoch": 0,
            "train_mean_loss": float("nan"),
            "train_energy_rmse_eV_per_atom": float("nan"),
            "train_force_SB_rmse_eV_A": float("nan"),
            "train_moe_residual_Epa_rms_eV": 0.0,
            "val_mean_loss": baseline_val["mean_combined_loss"],
            "val_energy_rmse_eV_per_atom":
                baseline_val["energy_rmse_eV_per_atom"],
            "val_force_SB_rmse_eV_A":
                baseline_val["force_structure_balanced_rmse_eV_A"],
            "val_moe_residual_Epa_rms_eV": 0.0,
            "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": 0.0,
            "val_seconds": baseline_val["elapsed_seconds"],
            "peak_cuda_allocated_GiB": float("nan"),
            "best_epoch": 0,
            "best_val_loss": best_val_loss,
        }
        history.append(baseline_row)
        write_csv(history_path, history)

        print(
            "\nEPOCH-0 C1a VALIDATION BASELINE | "
            f"E={baseline_val['energy_rmse_eV_per_atom']*1000:.3f} meV/atom "
            f"F={baseline_val['force_structure_balanced_rmse_eV_A']:.6f} eV/A "
            f"valL={best_val_loss:.8f}"
        )

    for epoch in range(
        start_epoch,
        args.max_epochs + 1,
    ):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_stats = train_one_epoch(
            residual_model=residual_model,
            descriptor=descriptor,
            common_model=common_model,
            common_mean=common_mean,
            common_std=common_std,
            routing_model=routing_model,
            routing_mean=routing_mean,
            routing_std=routing_std,
            residual_mean=residual_mean,
            residual_std=residual_std,
            train_records=train_records,
            optimizer=optimizer,
            e_ref=e_ref,
            device=device,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            grad_clip=args.grad_clip,
            rng=rng,
        )

        val_stats, _ = evaluate_split(
            residual_model=residual_model,
            descriptor=descriptor,
            common_model=common_model,
            common_mean=common_mean,
            common_std=common_std,
            routing_model=routing_model,
            routing_mean=routing_mean,
            routing_std=routing_std,
            residual_mean=residual_mean,
            residual_std=residual_std,
            records=val_records,
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
                    "residual_model_state_dict":
                        residual_model.state_dict(),
                    "input_dim": 392,
                    "residual_hidden_dims":
                        list(args.residual_hidden_dims),
                    "expert_names": EXPERTS,
                    "e_ref_eV_per_atom": e_ref,
                    "energy_weight":
                        args.energy_weight,
                    "force_weight":
                        args.force_weight,
                    "seed": args.seed,
                    "best_epoch":
                        best_epoch,
                    "best_val_loss":
                        best_val_loss,
                    "common_checkpoint":
                        str(common_checkpoint_path),
                    "routing_checkpoint":
                        str(routing_checkpoint_path),
                    "model_role":
                        "C1b frozen-routing conservative Local-MoE residual",
                    "residual_centering":
                        "delta_ik = raw_delta_ik - mean_k(raw_delta_ik)",
                    "routing_coordinate_gradient":
                        "enabled; routing parameters frozen, weights not detached",
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
            "train_moe_residual_Epa_rms_eV":
                train_stats[
                    "moe_residual_energy_per_atom_rms_eV"
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
            "val_moe_residual_Epa_rms_eV":
                val_stats[
                    "moe_residual_energy_per_atom_rms_eV"
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
            residual_model=residual_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            bad_epochs=bad_epochs,
            args=args,
            e_ref=e_ref,
            rng=rng,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train E={train_stats['energy_rmse_eV_per_atom']*1000:8.3f} meV/atom "
            f"F={train_stats['force_structure_balanced_rmse_eV_A']:8.5f} eV/A "
            f"dE_MoE={train_stats['moe_residual_energy_per_atom_rms_eV']*1000:7.3f} meV/atom | "
            f"val E={val_stats['energy_rmse_eV_per_atom']*1000:8.3f} meV/atom "
            f"F={val_stats['force_structure_balanced_rmse_eV_A']:8.5f} eV/A "
            f"dE_MoE={val_stats['moe_residual_energy_per_atom_rms_eV']*1000:7.3f} meV/atom | "
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
            f"Best C1b model was never saved: {best_model_path}"
        )

    best = torch.load(
        best_model_path,
        map_location="cpu",
        weights_only=False,
    )
    residual_model.load_state_dict(
        best["residual_model_state_dict"]
    )
    residual_model.eval()

    print("\n" + "=" * 122)
    print("FINAL EVALUATION USING BEST C1b CHECKPOINT")
    print("=" * 122)
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
            residual_model=residual_model,
            descriptor=descriptor,
            common_model=common_model,
            common_mean=common_mean,
            common_std=common_std,
            routing_model=routing_model,
            routing_mean=routing_mean,
            routing_std=routing_std,
            residual_mean=residual_mean,
            residual_std=residual_std,
            records=split_records,
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
        "stage": "C1b",
        "model_role":
            "frozen-routing conservative Local-MoE residual",
        "best_checkpoint":
            str(best_model_path),
        "best_epoch":
            int(best["best_epoch"]),
        "best_val_loss":
            float(best["best_val_loss"]),
        "common_checkpoint":
            str(common_checkpoint_path),
        "routing_checkpoint":
            str(routing_checkpoint_path),
        "e_ref_eV_per_atom": e_ref,
        "residual_architecture":
            [392]
            + list(args.residual_hidden_dims)
            + [7],
        "residual_centering":
            "per atom across expert channels",
        "routing_parameters_frozen":
            True,
        "routing_coordinate_gradient_enabled":
            True,
        "energy_weight":
            args.energy_weight,
        "force_weight":
            args.force_weight,
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
        f"{'MoE_E_RMS(meV/a)':>18} "
        f"{'mean |Fnet|':>14} {'max |Fnet|':>14}"
    )
    print("-" * 126)

    for row in split_summaries:
        print(
            f"{row['split']:<7} "
            f"{1000*row['energy_rmse_eV_per_atom']:18.6f} "
            f"{1000*row['energy_mae_eV_per_atom']:17.6f} "
            f"{row['force_structure_balanced_rmse_eV_A']:18.6f} "
            f"{1000*row['moe_residual_energy_per_atom_rms_eV']:18.6f} "
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
        "best_c1b_residual_model.pt",
        "latest_checkpoint.pt",
        "training_history.csv",
        "final_split_metrics.csv",
        "test_region_force_metrics.csv",
        "training_info.json",
    ]:
        print(f"  {output_root / name}")


if __name__ == "__main__":
    main()
