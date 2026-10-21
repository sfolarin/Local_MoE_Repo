#!/usr/bin/env python3
"""
train_local_torchsoap_gate_distillation.py

Stage B2 of Conservative Local MoE v1.

Train a NEW local routing gate in the differentiable TorchSOAP feature space
created in Stage B1, using the frozen DScribe dense-gate routing weights from
Stage A as teacher targets.

Student:
    [local TorchSOAP 390 + species one-hot 2] -> 256 -> 128 -> 7 logits
    softmax(logits) -> local expert weights

Loss:
    structure-balanced KL(teacher || student)

Each optimizer step uses one complete heterostructure, so every structure has
equal weight regardless of atom count. Feature normalization is fitted ONLY on
the training split.

After training, the script evaluates:
  * routing distillation quality on train/val/test
  * per-region routing quality on test
  * downstream force-mixture RMSE using the original stored seven expert forces
    and MATPes-r2SCAN reference forces:
        F_i = sum_k w_ik F_i^(k)
    for both teacher and student.
This downstream force test is the key Stage-B2 acceptance diagnostic.

No LAMMPS or reference-PES inference is performed; all force arrays are read
from the frozen expanded expert shards.  Their legacy ``mace_forces`` field
name is retained for file compatibility, but the labels are MATPes-r2SCAN,
not MPA-0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EXPERTS = [
    "ML-BKS",
    "Original_BKS",
    "Chik_BKS",
    "PSO1_BKS",
    "PSO2_BKS",
    "Tersoff",
    "Vashishta",
]

REGIONS = ["all", "interface", "phase1_bulk", "phase2_bulk"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight_decay", type=float, default=1.0e-6)
    p.add_argument("--max_epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=35)
    p.add_argument("--min_delta", type=float, default=1.0e-6)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--torch_threads", type=int, default=16)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--expected_split_sizes",
        type=int,
        nargs=3,
        default=[369, 38, 105],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Expected frozen train/val/test structure counts.",
    )
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


@dataclass
class StructureData:
    source_file: str
    phase_pair: str
    split: str
    feature_file: Path
    source_shard: Path
    X: np.ndarray
    teacher_weights: np.ndarray
    phase: np.ndarray
    interface_center: np.ndarray

    @property
    def n_atoms(self) -> int:
        return int(self.X.shape[0])


class RoutingMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        n_experts: int,
        dropout: float,
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
        layers.append(nn.Linear(d, n_experts))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def resolve_feature_file(dataset_root: Path, output_file: str) -> Path:
    path = dataset_root / output_file
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def load_dataset(dataset_root: Path) -> Dict[str, List[StructureData]]:
    manifest = read_csv(dataset_root / "manifest.csv")
    out: Dict[str, List[StructureData]] = {"train": [], "val": [], "test": []}

    for row in manifest:
        split = str(row["split"]).strip()
        if split not in out:
            raise RuntimeError(f"Unexpected split {split!r}")

        feature_file = resolve_feature_file(dataset_root, row["output_file"])
        source_shard = Path(row["source_shard"]).expanduser().resolve()
        if not source_shard.is_file():
            raise FileNotFoundError(source_shard)

        with np.load(feature_file, allow_pickle=False) as data:
            X = np.asarray(data["X_local_torchsoap"], dtype=np.float32)
            teacher = np.asarray(data["teacher_weights"], dtype=np.float32)
            phase = np.asarray(data["phase"], dtype=np.int16).reshape(-1)
            center = np.asarray(
                data["interface_center"], dtype=np.int16
            ).reshape(-1)
            names = [str(x) for x in data["expert_names"].tolist()]

        if X.ndim != 2 or X.shape[1] != 392:
            raise RuntimeError(f"{feature_file}: X shape={X.shape}, expected (*,392)")
        if teacher.shape != (len(X), 7):
            raise RuntimeError(f"{feature_file}: teacher shape={teacher.shape}")
        if names != EXPERTS:
            raise RuntimeError(f"{feature_file}: expert order mismatch")
        if len(phase) != len(X) or len(center) != len(X):
            raise RuntimeError(f"{feature_file}: region-label length mismatch")
        if not np.all(np.isfinite(X)):
            raise RuntimeError(f"{feature_file}: non-finite features")
        if not np.all(np.isfinite(teacher)):
            raise RuntimeError(f"{feature_file}: non-finite teacher weights")

        out[split].append(
            StructureData(
                source_file=str(row["source_file"]),
                phase_pair=str(row["phase_pair"]),
                split=split,
                feature_file=feature_file,
                source_shard=source_shard,
                X=X,
                teacher_weights=teacher,
                phase=phase,
                interface_center=center,
            )
        )

    return out


def compute_train_normalization(
    train_structures: Sequence[StructureData],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pooled-atom normalization, fitted only on training structures.
    Uses float64 running sums to avoid concatenating another large copy.
    """
    n_total = 0
    sum_x = None
    sum_x2 = None

    for s in train_structures:
        x = s.X.astype(np.float64, copy=False)
        if sum_x is None:
            sum_x = np.zeros(x.shape[1], dtype=np.float64)
            sum_x2 = np.zeros(x.shape[1], dtype=np.float64)
        sum_x += np.sum(x, axis=0, dtype=np.float64)
        sum_x2 += np.sum(x * x, axis=0, dtype=np.float64)
        n_total += len(x)

    mean = sum_x / n_total
    var = np.maximum(sum_x2 / n_total - mean * mean, 1.0e-12)
    std = np.sqrt(var)

    # Avoid over-amplifying nearly constant channels.
    std = np.where(std < 1.0e-6, 1.0, std)

    return mean.astype(np.float32), std.astype(np.float32)


def masks_for_structure(s: StructureData) -> Dict[str, np.ndarray]:
    return {
        "all": np.ones(s.n_atoms, dtype=bool),
        "interface": s.interface_center == 1,
        "phase1_bulk": (s.phase == 1) & (s.interface_center == 0),
        "phase2_bulk": (s.phase == 2) & (s.interface_center == 0),
    }


@torch.no_grad()
def predict_weights(
    model: nn.Module,
    structure: StructureData,
    feature_mean_t: torch.Tensor,
    feature_std_t: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    x = torch.from_numpy(structure.X).to(device)
    x = (x - feature_mean_t) / feature_std_t
    logits = model(x)
    weights = torch.softmax(logits, dim=1)
    return weights.cpu().numpy().astype(np.float32, copy=False)


class RoutingMetricAccumulator:
    def __init__(self) -> None:
        self.n_atoms = 0
        self.n_structures = 0
        self.kl_sum_atoms = 0.0
        self.l1_sum_atoms = 0.0
        self.max_abs = 0.0
        self.top1_correct = 0
        self.cos_sum_atoms = 0.0
        self.structure_kl_sum = 0.0
        self.structure_l1_sum = 0.0

    def update(
        self,
        student: np.ndarray,
        teacher: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return

        s = np.asarray(student[mask], dtype=np.float64)
        t = np.asarray(teacher[mask], dtype=np.float64)
        s = np.clip(s, 1.0e-12, 1.0)
        t = np.clip(t, 1.0e-12, 1.0)

        kl_atom = np.sum(t * (np.log(t) - np.log(s)), axis=1)
        l1_atom = np.mean(np.abs(t - s), axis=1)

        dot = np.sum(t * s, axis=1)
        denom = (
            np.linalg.norm(t, axis=1)
            * np.linalg.norm(s, axis=1)
            + 1.0e-15
        )
        cos = dot / denom

        self.n_atoms += len(s)
        self.n_structures += 1
        self.kl_sum_atoms += float(np.sum(kl_atom))
        self.l1_sum_atoms += float(np.sum(l1_atom))
        self.max_abs = max(
            self.max_abs,
            float(np.max(np.abs(t - s))),
        )
        self.top1_correct += int(
            np.sum(np.argmax(t, axis=1) == np.argmax(s, axis=1))
        )
        self.cos_sum_atoms += float(np.sum(cos))
        self.structure_kl_sum += float(np.mean(kl_atom))
        self.structure_l1_sum += float(np.mean(l1_atom))

    def finalize(self) -> Dict[str, float]:
        if self.n_atoms == 0:
            return {}
        return {
            "n_atoms": self.n_atoms,
            "n_structures": self.n_structures,
            "pooled_mean_KL_teacher_to_student": self.kl_sum_atoms / self.n_atoms,
            "structure_balanced_mean_KL_teacher_to_student":
                self.structure_kl_sum / self.n_structures,
            "pooled_mean_abs_weight_error": self.l1_sum_atoms / self.n_atoms,
            "structure_balanced_mean_abs_weight_error":
                self.structure_l1_sum / self.n_structures,
            "max_abs_weight_error": self.max_abs,
            "top1_agreement_fraction": self.top1_correct / self.n_atoms,
            "mean_weight_cosine_similarity": self.cos_sum_atoms / self.n_atoms,
        }


class ForceMetricAccumulator:
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
            "n_atoms": self.n_atoms,
            "n_structures": self.n_structures,
            "atom_weighted_rmse_component_eV_A":
                math.sqrt(self.sse / self.n_components),
            "structure_balanced_rmse_component_eV_A":
                math.sqrt(self.structure_mse_sum / self.n_structures),
        }


@torch.no_grad()
def evaluate_routing(
    model: nn.Module,
    structures: Sequence[StructureData],
    feature_mean_t: torch.Tensor,
    feature_std_t: torch.Tensor,
    device: torch.device,
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    regions = {region: RoutingMetricAccumulator() for region in REGIONS}
    predictions: Dict[str, np.ndarray] = {}

    model.eval()
    for s in structures:
        pred = predict_weights(
            model, s, feature_mean_t, feature_std_t, device
        )
        predictions[s.source_file] = pred
        masks = masks_for_structure(s)
        for region in REGIONS:
            regions[region].update(
                pred, s.teacher_weights, masks[region]
            )

    rows = []
    for region in REGIONS:
        result = regions[region].finalize()
        if result:
            row = {"region": region}
            row.update(result)
            rows.append(row)
    return rows, predictions


@torch.no_grad()
def validation_structure_balanced_kl(
    model: nn.Module,
    structures: Sequence[StructureData],
    feature_mean_t: torch.Tensor,
    feature_std_t: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    values = []
    for s in structures:
        x = torch.from_numpy(s.X).to(device)
        teacher = torch.from_numpy(s.teacher_weights).to(device)
        x = (x - feature_mean_t) / feature_std_t
        log_student = F.log_softmax(model(x), dim=1)
        loss = F.kl_div(
            log_student,
            teacher,
            reduction="batchmean",
        )
        values.append(float(loss.item()))
    return float(np.mean(values))


def train_one_epoch(
    model: nn.Module,
    structures: Sequence[StructureData],
    optimizer: torch.optim.Optimizer,
    feature_mean_t: torch.Tensor,
    feature_std_t: torch.Tensor,
    device: torch.device,
    grad_clip: float,
    rng: np.random.Generator,
) -> float:
    model.train()
    order = rng.permutation(len(structures))
    losses = []

    for idx in order:
        s = structures[int(idx)]
        x = torch.from_numpy(s.X).to(device)
        teacher = torch.from_numpy(s.teacher_weights).to(device)

        x = (x - feature_mean_t) / feature_std_t

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        log_student = F.log_softmax(logits, dim=1)

        # One whole structure per optimizer step => structure-balanced training.
        loss = F.kl_div(
            log_student,
            teacher,
            reduction="batchmean",
        )
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def evaluate_force_mixture(
    structures: Sequence[StructureData],
    student_predictions: Dict[str, np.ndarray],
) -> List[Dict]:
    acc = {
        ("teacher", region): ForceMetricAccumulator()
        for region in REGIONS
    }
    acc.update({
        ("student", region): ForceMetricAccumulator()
        for region in REGIONS
    })

    for s in structures:
        with np.load(s.source_shard, allow_pickle=False) as data:
            expert_forces = np.asarray(
                data["expert_forces"],
                dtype=np.float64,
            )
            mace_forces = np.asarray(
                data["mace_forces"],
                dtype=np.float64,
            )

        if expert_forces.shape != (7, s.n_atoms, 3):
            raise RuntimeError(
                f"{s.source_shard}: bad expert_forces shape "
                f"{expert_forces.shape}"
            )

        teacher_pred = np.einsum(
            "nk,knc->nc",
            s.teacher_weights.astype(np.float64),
            expert_forces,
        )
        student_w = student_predictions[s.source_file].astype(np.float64)
        student_pred = np.einsum(
            "nk,knc->nc",
            student_w,
            expert_forces,
        )

        masks = masks_for_structure(s)
        for region in REGIONS:
            acc[("teacher", region)].update(
                teacher_pred, mace_forces, masks[region]
            )
            acc[("student", region)].update(
                student_pred, mace_forces, masks[region]
            )

    rows: List[Dict] = []
    for predictor in ["teacher", "student"]:
        for region in REGIONS:
            result = acc[(predictor, region)].finalize()
            if result:
                row = {
                    "predictor": predictor,
                    "region": region,
                }
                row.update(result)
                rows.append(row)
    return rows


def save_test_predictions(
    output_root: Path,
    test_structures: Sequence[StructureData],
    predictions: Dict[str, np.ndarray],
) -> None:
    pred_root = output_root / "test_predictions"
    for s in test_structures:
        # Avoid source path collisions by using feature shard stem.
        name = s.feature_file.stem.replace("_local_torchsoap", "")
        path = pred_root / f"{name}_student_routing.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            student_weights=predictions[s.source_file],
            teacher_weights=s.teacher_weights,
            phase=s.phase,
            interface_center=s.interface_center,
            expert_names=np.asarray(EXPERTS),
            source_file=np.asarray(s.source_file),
            phase_pair=np.asarray(s.phase_pair),
        )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {output_root}\n"
                "Use --overwrite only when intentionally regenerating."
            )

    output_root.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    dataset = load_dataset(dataset_root)

    train_set = dataset["train"]
    val_set = dataset["val"]
    test_set = dataset["test"]

    expected_sizes = tuple(args.expected_split_sizes)
    if (len(train_set), len(val_set), len(test_set)) != expected_sizes:
        raise RuntimeError(
            "Unexpected split sizes: "
            f"{len(train_set)}/{len(val_set)}/{len(test_set)}; "
            f"expected {expected_sizes}"
        )

    input_dim = train_set[0].X.shape[1]
    if input_dim != 392:
        raise RuntimeError(f"Expected input_dim=392, found {input_dim}")

    feature_mean, feature_std = compute_train_normalization(train_set)
    feature_mean_t = torch.from_numpy(feature_mean).to(device)
    feature_std_t = torch.from_numpy(feature_std).to(device)

    model = RoutingMLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        n_experts=len(EXPERTS),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1.0e-6,
    )

    rng = np.random.default_rng(args.seed)

    print("=" * 118)
    print("STAGE B2: DISTILL LOCAL ROUTING INTO TORCHSOAP GATE")
    print("=" * 118)
    print(f"Dataset:          {dataset_root}")
    print(f"Output:           {output_root}")
    print(f"Device:           {device}")
    print(f"Splits:           train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
    print(f"Input dim:        {input_dim}")
    print(f"Hidden dims:      {args.hidden_dims}")
    print(f"Dropout:          {args.dropout}")
    print(f"Loss:             structure-balanced KL(teacher || student)")
    print(f"Max epochs:       {args.max_epochs}")
    print(f"Patience:         {args.patience}")
    print(f"Seed:             {args.seed}")
    print("=" * 118)

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: List[Dict] = []
    best_path = output_root / "best_torchsoap_routing_gate.pt"

    for epoch in range(1, args.max_epochs + 1):
        train_kl = train_one_epoch(
            model=model,
            structures=train_set,
            optimizer=optimizer,
            feature_mean_t=feature_mean_t,
            feature_std_t=feature_std_t,
            device=device,
            grad_clip=args.grad_clip,
            rng=rng,
        )

        val_kl = validation_structure_balanced_kl(
            model=model,
            structures=val_set,
            feature_mean_t=feature_mean_t,
            feature_std_t=feature_std_t,
            device=device,
        )

        scheduler.step(val_kl)
        lr_now = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_structure_balanced_KL": train_kl,
            "val_structure_balanced_KL": val_kl,
            "lr": lr_now,
        })

        improved = val_kl < (best_val - args.min_delta)
        if improved:
            best_val = val_kl
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_mean": torch.from_numpy(feature_mean),
                    "feature_std": torch.from_numpy(feature_std),
                    "input_dim": input_dim,
                    "hidden_dims": list(args.hidden_dims),
                    "dropout": args.dropout,
                    "expert_names": EXPERTS,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_val_structure_balanced_KL": best_val,
                    "feature_space": (
                        "local TorchSOAP rcut=5,nmax=6,lmax=4 "
                        "+ [isSi,isO]"
                    ),
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"Epoch {epoch:03d} "
                f"train_KL={train_kl:.8f} "
                f"val_KL={val_kl:.8f} "
                f"lr={lr_now:.3e} "
                f"{'*' if improved else ''}"
            )

        if bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best epoch={best_epoch}, best val KL={best_val:.8f}"
            )
            break

    write_csv(output_root / "training_history.csv", history)

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    routing_rows_all: List[Dict] = []
    predictions_by_split: Dict[str, Dict[str, np.ndarray]] = {}

    for split, structures in [
        ("train", train_set),
        ("val", val_set),
        ("test", test_set),
    ]:
        rows, preds = evaluate_routing(
            model=model,
            structures=structures,
            feature_mean_t=feature_mean_t,
            feature_std_t=feature_std_t,
            device=device,
        )
        for row in rows:
            row = {"split": split, **row}
            routing_rows_all.append(row)
        predictions_by_split[split] = preds

    write_csv(output_root / "routing_distillation_metrics.csv", routing_rows_all)

    force_rows = evaluate_force_mixture(
        test_set,
        predictions_by_split["test"],
    )
    write_csv(output_root / "test_force_mixture_metrics.csv", force_rows)

    save_test_predictions(
        output_root,
        test_set,
        predictions_by_split["test"],
    )

    info = {
        "stage": "B2",
        "reference_pes": "MATPes-r2SCAN",
        "reference_label_policy": (
            "Frozen expanded MATPes-r2SCAN labels; no MPA-0 labels or inference."
        ),
        "purpose": (
            "Distill frozen DScribe dense local-routing teacher into a new "
            "gate operating in differentiable local TorchSOAP feature space."
        ),
        "dataset_root": str(dataset_root),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_structure_balanced_KL": best_val,
        "expert_names": EXPERTS,
        "input_dim": input_dim,
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "seed": args.seed,
        "important_note": (
            "This checkpoint must later be paired with on-the-fly differentiable "
            "TorchSOAP using the same rcut=5,nmax=6,lmax=4 and species one-hot "
            "ordering [Si,O]."
        ),
    }
    (output_root / "training_info.json").write_text(
        json.dumps(info, indent=2, sort_keys=True) + "\n"
    )

    print("\n" + "=" * 118)
    print("ROUTING DISTILLATION RESULTS")
    print("=" * 118)
    print(
        f"{'split':<7} {'region':<13} "
        f"{'SB_KL':>12} {'MAE_w':>12} {'top1':>10} {'cosine':>10}"
    )
    print("-" * 70)
    for row in routing_rows_all:
        print(
            f"{row['split']:<7} {row['region']:<13} "
            f"{row['structure_balanced_mean_KL_teacher_to_student']:12.7f} "
            f"{row['structure_balanced_mean_abs_weight_error']:12.7f} "
            f"{row['top1_agreement_fraction']:10.4f} "
            f"{row['mean_weight_cosine_similarity']:10.5f}"
        )

    print("\n" + "=" * 118)
    print("TEST FORCE-MIXTURE REPRODUCTION")
    print("=" * 118)
    print(
        f"{'predictor':<10} {'region':<13} "
        f"{'SB_RMSE':>12} {'pooled_RMSE':>14} {'Nstruct':>9} {'Natoms':>9}"
    )
    print("-" * 74)
    for row in force_rows:
        print(
            f"{row['predictor']:<10} {row['region']:<13} "
            f"{row['structure_balanced_rmse_component_eV_A']:12.6f} "
            f"{row['atom_weighted_rmse_component_eV_A']:14.6f} "
            f"{row['n_structures']:9d} {row['n_atoms']:9d}"
        )

    print("\nSaved:")
    for name in [
        "best_torchsoap_routing_gate.pt",
        "training_history.csv",
        "routing_distillation_metrics.csv",
        "test_force_mixture_metrics.csv",
        "training_info.json",
        "test_predictions/",
    ]:
        print(f"  {output_root / name}")

    test_force = {
        str(row["region"]): float(
            row["structure_balanced_rmse_component_eV_A"]
        )
        for row in force_rows
        if row["predictor"] == "teacher"
    }
    print(
        "\nStage B2 acceptance check: student/teacher force-mixture values "
        "are reported above; current frozen teacher TEST values are "
        f"all={test_force.get('all', float('nan')):.6f}, "
        f"interface={test_force.get('interface', float('nan')):.6f} eV/A SB."
    )


if __name__ == "__main__":
    main()
