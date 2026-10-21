#!/usr/bin/env python3
"""
export_dense_gate_routing_teacher.py

Stage A for the conservative Dynamic Local MoE project.

Purpose
-------
Export the already-validated dense atom-local force-routing gate as a fixed
TEACHER dataset for later routing distillation into a differentiable
coordinate-based gate.

Important:
  * This script DOES NOT train a new model.
  * It DOES NOT use phase/interface labels as gate inputs.
  * It recomputes teacher logits/weights from X_local using the frozen
    Trained_Local_Gate_MPA0_dense/best_local_gate.pt checkpoint.
  * It preserves the exact existing train/val/test split.
  * It audits that the recomputed test gate reproduces the existing dense
    force-mixture metrics and, when available, the saved test_predictions.

Outputs
-------
output_root/
  teacher_manifest.csv
  routing_summary.csv
  force_reproduction_metrics.csv
  test_prediction_consistency.csv
  export_info.json
  shards/{train,val,test}/..._teacher.npz

Each teacher shard contains:
  teacher_logits       (N, 7) float32
  teacher_weights      (N, 7) float32
  phase                (N,)   int16       [analysis only]
  interface_center     (N,)   int16       [analysis only]
  expert_names         (7,)   strings
  source_file          scalar string
  split                scalar string
  phase_pair           scalar string

The new conservative gate should later learn from teacher_weights using atomic
geometry in the MACE-labeled structures. X_local is deliberately NOT copied
into the teacher shards, so the production differentiable gate cannot
accidentally depend on precomputed SOAP features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

REGIONS = [
    "all",
    "interface",
    "phase1_bulk",
    "phase2_bulk",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--trained_root", type=Path, required=True)
    p.add_argument("--output_root", type=Path, required=True)
    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )
    p.add_argument("--chunk_atoms", type=int, default=8192)
    p.add_argument("--torch_threads", type=int, default=16)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing non-empty output directory.",
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


def clean_optional(value) -> str:
    text = str(value if value is not None else "").strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def resolve_shard(dataset_root: Path, shard_file: str) -> Path:
    rel = Path(str(shard_file))
    if rel.parts and rel.parts[0] == "shards":
        path = dataset_root / rel
    else:
        path = dataset_root / "shards" / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def safe_teacher_relative(shard_file: str, split: str) -> Path:
    """
    Base teacher output on shard_file rather than source_file so the output
    path is guaranteed to remain inside output_root and unique.
    """
    rel = Path(str(shard_file))
    if rel.parts and rel.parts[0] == "shards":
        rel = Path(*rel.parts[1:])
    rel = rel.with_suffix("")
    return Path("shards") / split / rel.parent / f"{rel.name}_teacher.npz"


class LocalGateMLP(nn.Module):
    """Exact architecture used by train_local_gate_from_expert_forces.py."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        n_experts: int,
        dropout: float,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(previous_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, n_experts))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def load_gate(
    checkpoint_path: Path,
    device: torch.device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    expert_names = [str(x) for x in checkpoint["expert_names"]]
    if expert_names != EXPERTS:
        raise RuntimeError(
            "Expert order mismatch.\n"
            f"Expected: {EXPERTS}\n"
            f"Found:    {expert_names}"
        )

    input_dim = int(checkpoint["input_dim"])
    hidden_dims = [int(x) for x in checkpoint["hidden_dims"]]
    dropout = float(checkpoint["dropout"])
    temperature = float(checkpoint.get("temperature", 1.0))

    model = LocalGateMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        n_experts=len(EXPERTS),
        dropout=dropout,
    ).to(device)

    state = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # Compatibility with any variant that used attribute name "net".
        remapped = {}
        for key, value in state.items():
            if key.startswith("net."):
                key = "network." + key[len("net."):]
            remapped[key] = value
        model.load_state_dict(remapped)

    model.eval()

    feature_mean = checkpoint["feature_mean"].detach().to(
        device=device,
        dtype=torch.float32,
    )
    feature_std = checkpoint["feature_std"].detach().to(
        device=device,
        dtype=torch.float32,
    )

    return (
        model,
        feature_mean,
        feature_std,
        temperature,
        input_dim,
        hidden_dims,
        dropout,
        checkpoint,
    )


@torch.no_grad()
def predict_teacher(
    model: nn.Module,
    features: np.ndarray,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    temperature: float,
    device: torch.device,
    chunk_atoms: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_atoms = len(features)
    logits_out = np.empty((n_atoms, len(EXPERTS)), dtype=np.float32)
    weights_out = np.empty_like(logits_out)

    for start in range(0, n_atoms, chunk_atoms):
        end = min(start + chunk_atoms, n_atoms)

        x = torch.from_numpy(
            np.asarray(features[start:end], dtype=np.float32)
        ).to(device, non_blocking=True)

        x = (x - feature_mean) / feature_std
        logits = model(x)
        weights = torch.softmax(logits / temperature, dim=1)

        logits_out[start:end] = logits.detach().cpu().numpy().astype(
            np.float32, copy=False
        )
        weights_out[start:end] = weights.detach().cpu().numpy().astype(
            np.float32, copy=False
        )

    return logits_out, weights_out


def region_masks(
    phase: np.ndarray,
    center: np.ndarray,
) -> Dict[str, np.ndarray]:
    phase = np.asarray(phase).reshape(-1)
    center = np.asarray(center).reshape(-1)
    n = len(phase)
    return {
        "all": np.ones(n, dtype=bool),
        "interface": center == 1,
        "phase1_bulk": (phase == 1) & (center == 0),
        "phase2_bulk": (phase == 2) & (center == 0),
    }


class ForceAccumulator:
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
        self.sse += float(np.sum(diff * diff, dtype=np.float64))
        self.n_components += int(diff.size)
        self.structure_mse_sum += float(np.mean(diff * diff))
        self.n_structures += 1
        self.n_atoms += int(mask.sum())

    def finalize(self) -> Dict[str, float]:
        return {
            "n_structures": int(self.n_structures),
            "n_atoms": int(self.n_atoms),
            "structure_balanced_rmse_component_eV_A": float(
                math.sqrt(self.structure_mse_sum / self.n_structures)
            ) if self.n_structures else float("nan"),
            "atom_weighted_rmse_component_eV_A": float(
                math.sqrt(self.sse / self.n_components)
            ) if self.n_components else float("nan"),
        }


class RoutingAccumulator:
    def __init__(self) -> None:
        self.weight_sum = np.zeros(len(EXPERTS), dtype=np.float64)
        self.top1_count = np.zeros(len(EXPERTS), dtype=np.int64)
        self.entropy_sum = 0.0
        self.n_atoms = 0

    def update(self, weights: np.ndarray, mask: np.ndarray) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return
        w = np.asarray(weights[mask], dtype=np.float64)
        self.weight_sum += np.sum(w, axis=0, dtype=np.float64)
        top1 = np.argmax(w, axis=1)
        self.top1_count += np.bincount(
            top1,
            minlength=len(EXPERTS),
        )
        entropy = -np.sum(
            w * np.log(np.clip(w, 1.0e-15, 1.0)),
            axis=1,
        )
        self.entropy_sum += float(np.sum(entropy, dtype=np.float64))
        self.n_atoms += len(w)


def existing_prediction_path(
    trained_root: Path,
    source_file: str,
) -> Path:
    relative = Path(source_file)
    if relative.is_absolute():
        return Path("__NO_MATCH_FOR_ABSOLUTE_SOURCE__")
    return (
        trained_root
        / "test_predictions"
        / relative.parent
        / f"{relative.stem}_predictions.npz"
    )


def main() -> None:
    args = parse_args()

    torch.set_num_threads(max(1, args.torch_threads))

    dataset_root = args.dataset_root.expanduser().resolve()
    trained_root = args.trained_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not trained_root.is_dir():
        raise FileNotFoundError(trained_root)

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output directory is not empty: {output_root}\n"
                "Use --overwrite only if you intentionally want to regenerate it."
            )

    output_root.mkdir(parents=True, exist_ok=True)

    split_csv = trained_root / "dataset_split.csv"
    checkpoint_path = trained_root / "best_local_gate.pt"

    split_rows = read_csv(split_csv)
    if not split_rows:
        raise RuntimeError(f"No rows in {split_csv}")

    source_file_counts: Dict[str, int] = {}
    for row in split_rows:
        source = str(row["source_file"])
        source_file_counts[source] = source_file_counts.get(source, 0) + 1

    required_columns = {
        "source_file",
        "shard_file",
        "phase_pair",
        "n_atoms",
        "split",
    }
    missing = required_columns - set(split_rows[0].keys())
    if missing:
        raise RuntimeError(
            f"dataset_split.csv is missing required columns: {sorted(missing)}"
        )

    device = choose_device(args.device)

    (
        model,
        feature_mean,
        feature_std,
        temperature,
        input_dim,
        hidden_dims,
        dropout,
        checkpoint,
    ) = load_gate(checkpoint_path, device)

    print("=" * 118)
    print("EXPORT DENSE LOCAL GATE AS ROUTING TEACHER")
    print("=" * 118)
    print(f"Dataset root:       {dataset_root}")
    print(f"Trained gate root:  {trained_root}")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Output root:        {output_root}")
    print(f"Device:             {device}")
    print(f"Input dimension:    {input_dim}")
    print(f"Hidden dimensions:  {hidden_dims}")
    print(f"Dropout:            {dropout}")
    print(f"Temperature:        {temperature}")
    print(f"Structures:         {len(split_rows)}")
    print(f"Checkpoint epoch:   {checkpoint.get('epoch', 'NA')}")
    print("=" * 118)

    force_acc: Dict[Tuple[str, str], ForceAccumulator] = {
        (split, region): ForceAccumulator()
        for split in ["train", "val", "test"]
        for region in REGIONS
    }
    routing_acc: Dict[Tuple[str, str], RoutingAccumulator] = {
        (split, region): RoutingAccumulator()
        for split in ["train", "val", "test"]
        for region in REGIONS
    }

    manifest_rows: List[Dict] = []
    consistency_rows: List[Dict] = []

    max_weight_sum_error = 0.0
    min_weight_seen = 1.0
    max_weight_seen = 0.0
    total_atoms = 0

    split_counts = {
        split: {"structures": 0, "atoms": 0}
        for split in ["train", "val", "test"]
    }

    for number, row in enumerate(split_rows, start=1):
        split = str(row["split"]).strip()
        if split not in split_counts:
            raise RuntimeError(f"Unexpected split {split!r}")

        source_file = str(row["source_file"])
        phase_pair = str(row["phase_pair"])
        n_atoms = int(row["n_atoms"])
        shard_path = resolve_shard(
            dataset_root,
            str(row["shard_file"]),
        )

        with np.load(shard_path, allow_pickle=False) as data:
            features = np.asarray(data["X_local"], dtype=np.float32)
            mace_forces = np.asarray(data["mace_forces"], dtype=np.float32)
            expert_forces = np.asarray(data["expert_forces"], dtype=np.float32)
            phase = (
                np.asarray(data["phase"], dtype=np.int16).reshape(-1)
                if "phase" in data.files
                else np.full(n_atoms, -1, dtype=np.int16)
            )
            center = (
                np.asarray(
                    data["interface_center"],
                    dtype=np.int16,
                ).reshape(-1)
                if "interface_center" in data.files
                else np.full(n_atoms, -1, dtype=np.int16)
            )
            expert_names = [str(x) for x in data["expert_names"].tolist()]

        if features.shape != (n_atoms, input_dim):
            raise RuntimeError(
                f"{shard_path}: X_local shape {features.shape}, "
                f"expected {(n_atoms, input_dim)}"
            )
        if mace_forces.shape != (n_atoms, 3):
            raise RuntimeError(f"{shard_path}: bad mace_forces shape")
        if expert_forces.shape != (len(EXPERTS), n_atoms, 3):
            raise RuntimeError(f"{shard_path}: bad expert_forces shape")
        if expert_names != EXPERTS:
            raise RuntimeError(
                f"{shard_path}: expert order mismatch: {expert_names}"
            )

        logits, weights = predict_teacher(
            model=model,
            features=features,
            feature_mean=feature_mean,
            feature_std=feature_std,
            temperature=temperature,
            device=device,
            chunk_atoms=args.chunk_atoms,
        )

        if not np.all(np.isfinite(logits)):
            raise RuntimeError(f"Non-finite teacher logits in {shard_path}")
        if not np.all(np.isfinite(weights)):
            raise RuntimeError(f"Non-finite teacher weights in {shard_path}")

        weight_sum_error = float(
            np.max(np.abs(np.sum(weights, axis=1) - 1.0))
        )
        max_weight_sum_error = max(
            max_weight_sum_error,
            weight_sum_error,
        )
        min_weight_seen = min(min_weight_seen, float(np.min(weights)))
        max_weight_seen = max(max_weight_seen, float(np.max(weights)))

        if weight_sum_error > 5.0e-6:
            raise RuntimeError(
                f"Weight simplex failure in {shard_path}: "
                f"max |sum(w)-1|={weight_sum_error:.3e}"
            )
        if np.min(weights) < -1.0e-7:
            raise RuntimeError(
                f"Negative teacher weight in {shard_path}: {np.min(weights)}"
            )

        dense_forces = np.einsum(
            "nk,knc->nc",
            weights.astype(np.float64),
            expert_forces.astype(np.float64),
        ).astype(np.float32)

        masks = region_masks(phase, center)

        for region, mask in masks.items():
            force_acc[(split, region)].update(
                dense_forces,
                mace_forces,
                mask,
            )
            routing_acc[(split, region)].update(
                weights,
                mask,
            )

        teacher_rel = safe_teacher_relative(
            str(row["shard_file"]),
            split,
        )
        teacher_path = output_root / teacher_rel
        teacher_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            teacher_path,
            teacher_logits=logits,
            teacher_weights=weights,
            phase=phase,
            interface_center=center,
            expert_names=np.asarray(EXPERTS),
            source_file=np.asarray(source_file),
            split=np.asarray(split),
            phase_pair=np.asarray(phase_pair),
        )

        existing_pred = existing_prediction_path(trained_root, source_file)
        consistency_status = "not_applicable"
        max_weight_diff = float("nan")

        if split == "test":
            # The dense trainer historically names saved test predictions by
            # source_file.  Oishi relaxed records can legitimately share the
            # source name ``relaxed`` while having different geometries, so a
            # shared path cannot be used for an identity check.
            if source_file_counts[source_file] > 1:
                consistency_status = "non_unique_source_path"
            elif existing_pred.is_file():
                with np.load(existing_pred, allow_pickle=False) as pred:
                    old_weights = np.asarray(
                        pred["dense_weights"],
                        dtype=np.float32,
                    )
                if old_weights.shape != weights.shape:
                    raise RuntimeError(
                        f"Existing dense-weight shape mismatch: {existing_pred}"
                    )
                max_weight_diff = float(
                    np.max(np.abs(old_weights - weights))
                )
                consistency_status = "matched"
                if max_weight_diff > 2.0e-6:
                    raise RuntimeError(
                        "Recomputed teacher weights do not reproduce saved "
                        f"test predictions: max diff={max_weight_diff:.3e}\n"
                        f"{existing_pred}"
                    )
            else:
                consistency_status = "saved_test_prediction_missing"

            consistency_rows.append({
                "source_file": source_file,
                "phase_pair": phase_pair,
                "existing_prediction_file": str(existing_pred),
                "status": consistency_status,
                "max_abs_dense_weight_difference": max_weight_diff,
            })

        manifest_entry = dict(row)
        manifest_entry.update({
            "teacher_file": str(teacher_rel),
            "teacher_weight_sum_max_abs_error": weight_sum_error,
            "teacher_mean_entropy": float(
                np.mean(
                    -np.sum(
                        weights.astype(np.float64)
                        * np.log(
                            np.clip(
                                weights.astype(np.float64),
                                1.0e-15,
                                1.0,
                            )
                        ),
                        axis=1,
                    )
                )
            ),
            "teacher_top1_ML-BKS_fraction": float(
                np.mean(np.argmax(weights, axis=1) == 0)
            ),
            "teacher_top1_Tersoff_fraction": float(
                np.mean(np.argmax(weights, axis=1) == 5)
            ),
            "teacher_top1_Vashishta_fraction": float(
                np.mean(np.argmax(weights, axis=1) == 6)
            ),
        })
        manifest_rows.append(manifest_entry)

        split_counts[split]["structures"] += 1
        split_counts[split]["atoms"] += n_atoms
        total_atoms += n_atoms

        print(
            f"[{number:03d}/{len(split_rows):03d}] "
            f"{split:<5} N={n_atoms:4d} "
            f"entropy={manifest_entry['teacher_mean_entropy']:.4f} "
            f"top1={EXPERTS[int(np.bincount(np.argmax(weights, axis=1), minlength=7).argmax())]:<12} "
            f"simplex_err={weight_sum_error:.2e}"
        )

    force_rows: List[Dict] = []
    for split in ["train", "val", "test"]:
        for region in REGIONS:
            row = {
                "split": split,
                "predictor": "recomputed_dense_teacher",
                "region": region,
            }
            row.update(force_acc[(split, region)].finalize())
            force_rows.append(row)

    routing_rows: List[Dict] = []
    for split in ["train", "val", "test"]:
        for region in REGIONS:
            acc = routing_acc[(split, region)]
            if acc.n_atoms == 0:
                continue
            for k, expert in enumerate(EXPERTS):
                routing_rows.append({
                    "split": split,
                    "region": region,
                    "expert": expert,
                    "n_atoms": acc.n_atoms,
                    "mean_teacher_weight": float(
                        acc.weight_sum[k] / acc.n_atoms
                    ),
                    "top1_count": int(acc.top1_count[k]),
                    "top1_fraction": float(
                        acc.top1_count[k] / acc.n_atoms
                    ),
                    "mean_routing_entropy": float(
                        acc.entropy_sum / acc.n_atoms
                    ),
                })

    write_csv(output_root / "teacher_manifest.csv", manifest_rows)
    write_csv(output_root / "routing_summary.csv", routing_rows)
    write_csv(
        output_root / "force_reproduction_metrics.csv",
        force_rows,
    )
    write_csv(
        output_root / "test_prediction_consistency.csv",
        consistency_rows,
    )

    export_info = {
        "purpose": (
            "Frozen routing-teacher targets for Stage A distillation into a "
            "future differentiable coordinate-based local gate."
        ),
        "dataset_root": str(dataset_root),
        "trained_root": str(trained_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "expert_names": EXPERTS,
        "input_dim_of_teacher_only": input_dim,
        "hidden_dims_of_teacher_only": hidden_dims,
        "dropout_of_teacher_only": dropout,
        "temperature": temperature,
        "total_structures": len(split_rows),
        "total_atoms": total_atoms,
        "split_counts": split_counts,
        "teacher_simplex_max_abs_sum_error": max_weight_sum_error,
        "teacher_min_weight": min_weight_seen,
        "teacher_max_weight": max_weight_seen,
        "important_note": (
            "X_local/SOAP is intentionally not copied into teacher shards. "
            "The future production gate must be differentiable with respect "
            "to coordinates and should learn from atomic geometry, using "
            "teacher_weights only as supervision."
        ),
    }
    (output_root / "export_info.json").write_text(
        json.dumps(export_info, indent=2, sort_keys=True) + "\n"
    )

    print("\n" + "=" * 118)
    print("FORCE REPRODUCTION AUDIT")
    print("=" * 118)
    print(
        f"{'split':<7} {'region':<13} {'SB_RMSE':>12} "
        f"{'pooled_RMSE':>14} {'Nstruct':>9} {'Natoms':>9}"
    )
    print("-" * 72)

    for row in force_rows:
        print(
            f"{row['split']:<7} {row['region']:<13} "
            f"{row['structure_balanced_rmse_component_eV_A']:12.6f} "
            f"{row['atom_weighted_rmse_component_eV_A']:14.6f} "
            f"{row['n_structures']:9d} {row['n_atoms']:9d}"
        )

    test_consistency_numeric = [
        r for r in consistency_rows
        if r["status"] == "matched"
    ]
    if test_consistency_numeric:
        max_saved_diff = max(
            float(r["max_abs_dense_weight_difference"])
            for r in test_consistency_numeric
        )
        print(
            "\nSaved test-prediction consistency: "
            f"max |recomputed-old dense weight| = {max_saved_diff:.3e}"
        )
    else:
        print(
            "\nSaved test-prediction consistency: no saved test prediction "
            "files were found; force reproduction audit above remains valid."
        )

    print(
        "\nTeacher simplex audit: "
        f"max |sum(w)-1|={max_weight_sum_error:.3e}, "
        f"min(w)={min_weight_seen:.3e}, max(w)={max_weight_seen:.6f}"
    )

    print("\nSaved:")
    for name in [
        "teacher_manifest.csv",
        "routing_summary.csv",
        "force_reproduction_metrics.csv",
        "test_prediction_consistency.csv",
        "export_info.json",
    ]:
        print(f"  {output_root / name}")

    print(
        "\nStage A acceptance check: the TEST rows above should reproduce the "
        "existing trained_dense metrics (approximately all=0.457742 and "
        "interface=0.630823 eV/A, structure-balanced)."
    )


if __name__ == "__main__":
    main()
