#!/usr/bin/env python3
"""
train_local_gate_from_expert_forces.py

Train and evaluate an atom-centered local gate for the Dynamic Local MoE
silica project.

Input
-----
A directory produced by build_local_expert_force_dataset.py, containing:

    dataset_manifest.csv
    shards/**/*.npz

Each shard must contain:

    X_local             (N, 392)
    mace_forces         (N, 3)
    expert_forces       (7, N, 3)
    expert_names        (7,)
    phase               (N,)                  [optional; -1 fallback]
    interface_center    (N,)                  [optional; -1 fallback]

Method
------
The gate predicts seven nonnegative weights for every atom:

    w_i = softmax(g(X_i))

and the force-level training prediction is:

    F_i^mix = sum_k w_ik F_i^(k)

The training loss is structure-balanced: every structure contributes equally
regardless of atom count. Complete phase-pair groups are assigned to train,
validation, or test. Candidate duplicate groups are unioned with phase-pair
groups before splitting, preventing leakage.

The script trains a dense seven-expert gate, then evaluates:

    • each individual expert;
    • the uniform seven-expert mixture;
    • an optimal global simplex mixture (diagnostic oracle);
    • the trained dense local gate;
    • Top-2-pruned inference from the trained dense logits;
    • local Top-1 and local convex Top-2 force oracles.

The local force mixtures and local oracles are static force diagnostics. They
are not automatically conservative and should not be treated as a production
MD force field.

Example
-------
srun -N 1 -n 1 -c 16 --gpus=1 --gpu-bind=none \
python3 -u train_local_gate_from_expert_forces.py \
    --dataset_root Local_Expert_Force_Dataset_MPA0 \
    --output_dir Trained_Local_Gate_MPA0_dense \
    --device cuda \
    --epochs 300 \
    --patience 45 \
    --batch_structures 4 \
    --atoms_per_structure 1024 \
    --hidden_dims 256 128 \
    --save_test_predictions \
    |& tee train_local_gate_mpa0_dense.log
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


EXPECTED_EXPERTS = [
    "ML-BKS",
    "Original_BKS",
    "Chik_BKS",
    "PSO1_BKS",
    "PSO2_BKS",
    "Tersoff",
    "Vashishta",
]

REGION_NAMES = [
    "all",
    "interface",
    "phase1_bulk",
    "phase2_bulk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a structure-balanced local force-mixing gate from "
            "SOAP descriptors, seven expert forces, and MACE reference forces."
        )
    )

    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260806,
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 128],
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=45,
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3.0e-4,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--batch_structures",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--atoms_per_structure",
        type=int,
        default=1024,
        help=(
            "Random atoms sampled from each structure per epoch. "
            "<=0 uses every atom."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=["mse", "huber"],
        default="mse",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--entropy_coefficient",
        type=float,
        default=0.0,
        help=(
            "Positive values penalize high routing entropy. "
            "Default 0 gives an unconstrained dense gate."
        ),
    )
    parser.add_argument(
        "--balance_coefficient",
        type=float,
        default=0.0,
        help=(
            "Optional penalty on deviation of batch-mean expert weights "
            "from a uniform load. Default 0."
        ),
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision during training.",
    )
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--eval_chunk_atoms",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--no_cache_in_memory",
        action="store_true",
        help=(
            "Read shards from disk whenever needed. By default all required "
            "arrays are cached in host memory."
        ),
    )
    parser.add_argument(
        "--save_test_predictions",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--preserve_manifest_split",
        action="store_true",
        help=(
            "Use the split column in dataset_manifest.csv exactly. "
            "Required for the expanded 410/44/58 dataset."
        ),
    )

    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested, but CUDA is unavailable."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        path.write_text("")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def parse_int(value: str) -> int:
    return int(float(value))


def parse_optional_string(value: Optional[str]) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
    }:
        return ""

    return text


@dataclass
class ShardRecord:
    index: int
    source_file: str
    shard_path: Path
    phase_pair: str
    duplicate_group: str
    n_atoms: int
    best_expert_all: str
    split: str = ""


class UnionFind:
    def __init__(self, n_items: int) -> None:
        self.parent = list(range(n_items))
        self.rank = [0] * n_items

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[
                self.parent[item]
            ]
            item = self.parent[item]

        return item

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = (
                root_second,
                root_first,
            )

        self.parent[root_second] = root_first

        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def load_records(dataset_root: Path) -> List[ShardRecord]:
    manifest_path = (
        dataset_root / "dataset_manifest.csv"
    )
    rows = read_csv(manifest_path)
    records: List[ShardRecord] = []

    for index, row in enumerate(rows):
        output_file = row.get("output_file")

        if not output_file:
            raise ValueError(
                "Manifest row lacks output_file: {}".format(
                    row
                )
            )

        shard_path = (
            dataset_root
            / "shards"
            / output_file
        )

        if not shard_path.is_file():
            raise FileNotFoundError(
                "Shard not found: {}".format(
                    shard_path
                )
            )

        records.append(
            ShardRecord(
                index=index,
                source_file=str(
                    row["source_file"]
                ),
                shard_path=shard_path,
                phase_pair=str(
                    row["phase_pair"]
                ),
                duplicate_group=parse_optional_string(
                    row.get("duplicate_group")
                ),
                n_atoms=parse_int(
                    row["n_atoms"]
                ),
                best_expert_all=str(
                    row.get(
                        "best_expert_all",
                        "",
                    )
                ),
                split=parse_optional_string(row.get("split")),
            )
        )

    if not records:
        raise RuntimeError(
            "Dataset manifest contains no structures."
        )

    return records


def build_split_units(
    records: Sequence[ShardRecord],
) -> List[List[int]]:
    union_find = UnionFind(len(records))

    first_by_phase_pair: Dict[str, int] = {}
    first_by_duplicate_group: Dict[str, int] = {}

    for record in records:
        if record.phase_pair in first_by_phase_pair:
            union_find.union(
                record.index,
                first_by_phase_pair[
                    record.phase_pair
                ],
            )
        else:
            first_by_phase_pair[
                record.phase_pair
            ] = record.index

        if record.duplicate_group:
            if (
                record.duplicate_group
                in first_by_duplicate_group
            ):
                union_find.union(
                    record.index,
                    first_by_duplicate_group[
                        record.duplicate_group
                    ],
                )
            else:
                first_by_duplicate_group[
                    record.duplicate_group
                ] = record.index

    units_by_root: Dict[int, List[int]] = {}

    for record in records:
        root = union_find.find(record.index)
        units_by_root.setdefault(
            root,
            [],
        ).append(record.index)

    return list(units_by_root.values())


def assign_grouped_splits(
    records: Sequence[ShardRecord],
    fractions: Dict[str, float],
    seed: int,
) -> Dict[int, str]:
    split_names = [
        "train",
        "val",
        "test",
    ]

    total_fraction = sum(
        fractions[name]
        for name in split_names
    )

    if not np.isclose(
        total_fraction,
        1.0,
        atol=1.0e-8,
    ):
        raise ValueError(
            "Train/validation/test fractions must sum to 1."
        )

    units = build_split_units(records)
    rng = np.random.default_rng(seed)

    unit_summaries = []

    for unit in units:
        unit_summaries.append({
            "indices": unit,
            "atoms": sum(
                records[index].n_atoms
                for index in unit
            ),
            "structures": len(unit),
            "jitter": float(rng.random()),
        })

    unit_summaries.sort(
        key=lambda item: (
            -item["atoms"],
            item["jitter"],
        )
    )

    total_atoms = sum(
        record.n_atoms
        for record in records
    )
    total_structures = len(records)
    total_units = len(unit_summaries)

    targets = {
        split: {
            "atoms": (
                total_atoms
                * fractions[split]
            ),
            "structures": (
                total_structures
                * fractions[split]
            ),
            "units": (
                total_units
                * fractions[split]
            ),
        }
        for split in split_names
    }

    current = {
        split: {
            "atoms": 0,
            "structures": 0,
            "units": 0,
        }
        for split in split_names
    }

    assignment: Dict[int, str] = {}

    for unit_number, unit in enumerate(
        unit_summaries
    ):
        remaining_units = (
            len(unit_summaries) - unit_number
        )
        empty_splits = [
            split
            for split in split_names
            if current[split]["units"] == 0
        ]

        candidate_splits = split_names

        if (
            remaining_units
            <= len(empty_splits)
            and empty_splits
        ):
            candidate_splits = empty_splits

        candidate_scores = []

        for split in candidate_splits:
            projected = copy.deepcopy(current)
            projected[split]["atoms"] += (
                unit["atoms"]
            )
            projected[split]["structures"] += (
                unit["structures"]
            )
            projected[split]["units"] += 1

            score = 0.0

            for name in split_names:
                target_atoms = max(
                    targets[name]["atoms"],
                    1.0,
                )
                target_structures = max(
                    targets[name]["structures"],
                    1.0,
                )
                target_units = max(
                    targets[name]["units"],
                    1.0,
                )

                atom_deviation = (
                    projected[name]["atoms"]
                    - targets[name]["atoms"]
                ) / target_atoms

                structure_deviation = (
                    projected[name]["structures"]
                    - targets[name]["structures"]
                ) / target_structures

                unit_deviation = (
                    projected[name]["units"]
                    - targets[name]["units"]
                ) / target_units

                score += (
                    atom_deviation ** 2
                    + 0.30
                    * structure_deviation ** 2
                    + 0.10
                    * unit_deviation ** 2
                )

            score += 1.0e-9 * float(
                rng.random()
            )
            candidate_scores.append(
                (score, split)
            )

        _, chosen_split = min(
            candidate_scores,
            key=lambda item: item[0],
        )

        current[chosen_split]["atoms"] += (
            unit["atoms"]
        )
        current[chosen_split]["structures"] += (
            unit["structures"]
        )
        current[chosen_split]["units"] += 1

        for record_index in unit["indices"]:
            assignment[
                record_index
            ] = chosen_split

    for split in split_names:
        if current[split]["structures"] == 0:
            raise RuntimeError(
                "Grouped split produced an empty {} set.".format(
                    split
                )
            )

    return assignment


class ShardStore:
    def __init__(
        self,
        records: Sequence[ShardRecord],
        cache_in_memory: bool,
    ) -> None:
        self.records = list(records)
        self.cache_in_memory = bool(
            cache_in_memory
        )
        self.cache: Dict[
            int,
            Dict[str, np.ndarray],
        ] = {}

        self.feature_dim: Optional[int] = None
        self.expert_names: Optional[List[str]] = None
        self.estimated_cache_bytes = 0

        self._inspect_all()

        if self.cache_in_memory:
            self._cache_all()

    def _read_arrays(
        self,
        record: ShardRecord,
    ) -> Dict[str, np.ndarray]:
        with np.load(
            record.shard_path,
            allow_pickle=False,
        ) as data:
            features = np.asarray(
                data["X_local"],
                dtype=np.float32,
            )
            n_atoms = int(features.shape[0])
            arrays = {
                "X_local": features,
                "mace_forces": np.asarray(
                    data["mace_forces"],
                    dtype=np.float32,
                ),
                "expert_forces": np.asarray(
                    data["expert_forces"],
                    dtype=np.float32,
                ),
                # External Oishi/Siddharth shards do not necessarily carry
                # region-analysis arrays.  Keep them explicitly unknown;
                # all-atom force metrics remain valid, while interface/bulk
                # regional metrics are omitted for those records.
                "phase": np.asarray(
                    data["phase"], dtype=np.int16,
                ) if "phase" in data.files else np.full(
                    n_atoms, -1, dtype=np.int16,
                ),
                "interface_center": np.asarray(
                    data["interface_center"], dtype=np.int16,
                ) if "interface_center" in data.files else np.full(
                    n_atoms, -1, dtype=np.int16,
                ),
                "expert_names": np.asarray(
                    data["expert_names"]
                ),
            }

        return arrays

    def _validate_arrays(
        self,
        record: ShardRecord,
        arrays: Dict[str, np.ndarray],
    ) -> None:
        n_atoms = record.n_atoms
        features = arrays["X_local"]
        reference = arrays["mace_forces"]
        expert_forces = arrays["expert_forces"]
        names = [
            str(name)
            for name in arrays[
                "expert_names"
            ].tolist()
        ]

        if features.shape[0] != n_atoms:
            raise ValueError(
                "{}: X_local atom count mismatch.".format(
                    record.shard_path
                )
            )

        if reference.shape != (
            n_atoms,
            3,
        ):
            raise ValueError(
                "{}: invalid MACE force shape {}.".format(
                    record.shard_path,
                    reference.shape,
                )
            )

        if expert_forces.shape != (
            len(EXPECTED_EXPERTS),
            n_atoms,
            3,
        ):
            raise ValueError(
                "{}: invalid expert force shape {}.".format(
                    record.shard_path,
                    expert_forces.shape,
                )
            )

        for key in ("phase", "interface_center"):
            if arrays[key].reshape(-1).shape != (n_atoms,):
                raise ValueError(
                    "{}: invalid {} shape {}.".format(
                        record.shard_path,
                        key,
                        arrays[key].shape,
                    )
                )

        if names != EXPECTED_EXPERTS:
            raise ValueError(
                "{}: expert order mismatch.\n"
                "Expected: {}\n"
                "Found: {}".format(
                    record.shard_path,
                    EXPECTED_EXPERTS,
                    names,
                )
            )

        if not np.all(
            np.isfinite(features)
        ):
            raise ValueError(
                "{}: non-finite X_local.".format(
                    record.shard_path
                )
            )

        if not np.all(
            np.isfinite(reference)
        ):
            raise ValueError(
                "{}: non-finite MACE forces.".format(
                    record.shard_path
                )
            )

        if not np.all(
            np.isfinite(expert_forces)
        ):
            raise ValueError(
                "{}: non-finite expert forces.".format(
                    record.shard_path
                )
            )

        if self.feature_dim is None:
            self.feature_dim = int(
                features.shape[1]
            )
        elif features.shape[1] != self.feature_dim:
            raise ValueError(
                "{}: inconsistent feature dimension.".format(
                    record.shard_path
                )
            )

        if self.expert_names is None:
            self.expert_names = names

    def _inspect_all(self) -> None:
        for record in self.records:
            arrays = self._read_arrays(
                record
            )
            self._validate_arrays(
                record,
                arrays,
            )

            self.estimated_cache_bytes += sum(
                array.nbytes
                for key, array in arrays.items()
                if key != "expert_names"
            )

    def _cache_all(self) -> None:
        for record in self.records:
            arrays = self._read_arrays(
                record
            )
            self._validate_arrays(
                record,
                arrays,
            )
            arrays.pop(
                "expert_names",
                None,
            )
            self.cache[
                record.index
            ] = arrays

    def get(
        self,
        record: ShardRecord,
    ) -> Dict[str, np.ndarray]:
        if self.cache_in_memory:
            return self.cache[
                record.index
            ]

        arrays = self._read_arrays(
            record
        )
        self._validate_arrays(
            record,
            arrays,
        )
        arrays.pop(
            "expert_names",
            None,
        )
        return arrays


def compute_feature_statistics(
    train_records: Sequence[ShardRecord],
    store: ShardStore,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_dim = int(
        store.feature_dim
    )
    total_count = 0
    feature_sum = np.zeros(
        feature_dim,
        dtype=np.float64,
    )
    feature_square_sum = np.zeros(
        feature_dim,
        dtype=np.float64,
    )

    for record in train_records:
        features = store.get(
            record
        )["X_local"].astype(
            np.float64,
            copy=False,
        )

        total_count += len(features)
        feature_sum += np.sum(
            features,
            axis=0,
            dtype=np.float64,
        )
        feature_square_sum += np.sum(
            features ** 2,
            axis=0,
            dtype=np.float64,
        )

    mean = feature_sum / total_count
    variance = (
        feature_square_sum / total_count
        - mean ** 2
    )
    variance = np.maximum(
        variance,
        1.0e-12,
    )
    standard_deviation = np.sqrt(
        variance
    )

    standard_deviation[
        standard_deviation < 1.0e-6
    ] = 1.0

    return (
        mean.astype(np.float32),
        standard_deviation.astype(
            np.float32
        ),
    )


class LocalGateMLP(nn.Module):
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
                nn.Linear(
                    previous_dim,
                    hidden_dim,
                ),
                nn.LayerNorm(
                    hidden_dim
                ),
                nn.SiLU(),
            ])

            if dropout > 0.0:
                layers.append(
                    nn.Dropout(
                        dropout
                    )
                )

            previous_dim = hidden_dim

        layers.append(
            nn.Linear(
                previous_dim,
                n_experts,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            features
        )


def normalize_features(
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> torch.Tensor:
    return (
        features - feature_mean
    ) / feature_std


def mix_forces(
    weights: torch.Tensor,
    expert_forces: torch.Tensor,
) -> torch.Tensor:
    # weights:       (N, K)
    # expert_forces: (K, N, 3)
    return torch.einsum(
        "nk,knc->nc",
        weights,
        expert_forces,
    )


def top2_prune(
    dense_weights: torch.Tensor,
) -> torch.Tensor:
    values, indices = torch.topk(
        dense_weights,
        k=2,
        dim=1,
    )
    values = values / torch.clamp(
        values.sum(
            dim=1,
            keepdim=True,
        ),
        min=1.0e-12,
    )

    sparse = torch.zeros_like(
        dense_weights
    )
    sparse.scatter_(
        1,
        indices,
        values,
    )
    return sparse


def structure_force_loss(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    loss_name: str,
    huber_delta: float,
) -> torch.Tensor:
    difference = prediction - reference

    if loss_name == "mse":
        return torch.mean(
            difference ** 2
        )

    absolute = torch.abs(
        difference
    )
    quadratic = torch.minimum(
        absolute,
        torch.tensor(
            huber_delta,
            device=absolute.device,
            dtype=absolute.dtype,
        ),
    )
    linear = absolute - quadratic

    return torch.mean(
        0.5 * quadratic ** 2
        + huber_delta * linear
    )


def sample_atom_indices(
    n_atoms: int,
    atoms_per_structure: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if (
        atoms_per_structure <= 0
        or n_atoms <= atoms_per_structure
    ):
        return np.arange(
            n_atoms,
            dtype=np.int64,
        )

    return np.sort(
        rng.choice(
            n_atoms,
            size=atoms_per_structure,
            replace=False,
        ).astype(
            np.int64
        )
    )


def train_one_epoch(
    model: nn.Module,
    train_records: Sequence[ShardRecord],
    store: ShardStore,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    args: argparse.Namespace,
    epoch_seed: int,
) -> Dict[str, float]:
    model.train()

    rng = np.random.default_rng(
        epoch_seed
    )
    order = rng.permutation(
        len(train_records)
    )

    total_loss_sum = 0.0
    force_loss_sum = 0.0
    entropy_sum = 0.0
    balance_sum = 0.0
    optimizer_steps = 0

    for batch_start in range(
        0,
        len(order),
        args.batch_structures,
    ):
        batch_indices = order[
            batch_start:
            batch_start
            + args.batch_structures
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        batch_total_loss = None
        batch_force_loss = 0.0
        batch_entropy = 0.0
        batch_balance = 0.0

        with torch.cuda.amp.autocast(
            enabled=(
                args.amp
                and device.type == "cuda"
            )
        ):
            structure_losses = []
            structure_force_losses = []
            structure_entropies = []
            structure_balances = []

            for record_index in batch_indices:
                record = train_records[
                    int(record_index)
                ]
                arrays = store.get(
                    record
                )
                atom_indices = sample_atom_indices(
                    record.n_atoms,
                    args.atoms_per_structure,
                    rng,
                )

                features = torch.from_numpy(
                    arrays["X_local"][
                        atom_indices
                    ]
                ).to(
                    device,
                    non_blocking=True,
                )
                reference = torch.from_numpy(
                    arrays["mace_forces"][
                        atom_indices
                    ]
                ).to(
                    device,
                    non_blocking=True,
                )
                expert_forces = torch.from_numpy(
                    arrays["expert_forces"][
                        :,
                        atom_indices,
                        :,
                    ]
                ).to(
                    device,
                    non_blocking=True,
                )

                normalized = normalize_features(
                    features,
                    feature_mean,
                    feature_std,
                )
                logits = model(
                    normalized
                )
                weights = torch.softmax(
                    logits
                    / args.temperature,
                    dim=1,
                )
                prediction = mix_forces(
                    weights,
                    expert_forces,
                )

                force_loss = structure_force_loss(
                    prediction,
                    reference,
                    args.loss,
                    args.huber_delta,
                )

                normalized_entropy = (
                    -torch.sum(
                        weights
                        * torch.log(
                            torch.clamp(
                                weights,
                                min=1.0e-12,
                            )
                        ),
                        dim=1,
                    ).mean()
                    / math.log(
                        len(EXPECTED_EXPERTS)
                    )
                )

                mean_weight = weights.mean(
                    dim=0
                )
                balance_penalty = torch.mean(
                    (
                        mean_weight
                        - 1.0
                        / len(EXPECTED_EXPERTS)
                    ) ** 2
                )

                total_structure_loss = (
                    force_loss
                    + args.entropy_coefficient
                    * normalized_entropy
                    + args.balance_coefficient
                    * balance_penalty
                )

                structure_losses.append(
                    total_structure_loss
                )
                structure_force_losses.append(
                    force_loss.detach()
                )
                structure_entropies.append(
                    normalized_entropy.detach()
                )
                structure_balances.append(
                    balance_penalty.detach()
                )

            batch_total_loss = torch.stack(
                structure_losses
            ).mean()

        scaler.scale(
            batch_total_loss
        ).backward()

        if args.gradient_clip > 0.0:
            scaler.unscale_(
                optimizer
            )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.gradient_clip,
            )

        scaler.step(
            optimizer
        )
        scaler.update()

        batch_force_loss = float(
            torch.stack(
                structure_force_losses
            ).mean().cpu()
        )
        batch_entropy = float(
            torch.stack(
                structure_entropies
            ).mean().cpu()
        )
        batch_balance = float(
            torch.stack(
                structure_balances
            ).mean().cpu()
        )

        total_loss_sum += float(
            batch_total_loss.detach().cpu()
        )
        force_loss_sum += batch_force_loss
        entropy_sum += batch_entropy
        balance_sum += batch_balance
        optimizer_steps += 1

    return {
        "train_total_loss": (
            total_loss_sum
            / optimizer_steps
        ),
        "train_force_loss": (
            force_loss_sum
            / optimizer_steps
        ),
        "train_normalized_entropy": (
            entropy_sum
            / optimizer_steps
        ),
        "train_balance_penalty": (
            balance_sum
            / optimizer_steps
        ),
    }


@torch.no_grad()
def predict_gate_for_structure(
    model: nn.Module,
    features: np.ndarray,
    expert_forces: np.ndarray,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    temperature: float,
    chunk_atoms: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    model.eval()

    n_atoms = len(features)
    n_experts = len(EXPECTED_EXPERTS)

    dense_weights = np.empty(
        (n_atoms, n_experts),
        dtype=np.float32,
    )
    top2_weights = np.empty_like(
        dense_weights
    )
    dense_forces = np.empty(
        (n_atoms, 3),
        dtype=np.float32,
    )
    top2_forces = np.empty_like(
        dense_forces
    )

    for start in range(
        0,
        n_atoms,
        chunk_atoms,
    ):
        end = min(
            start + chunk_atoms,
            n_atoms,
        )

        feature_tensor = torch.from_numpy(
            features[start:end]
        ).to(
            device,
            non_blocking=True,
        )
        expert_tensor = torch.from_numpy(
            expert_forces[
                :,
                start:end,
                :,
            ]
        ).to(
            device,
            non_blocking=True,
        )

        normalized = normalize_features(
            feature_tensor,
            feature_mean,
            feature_std,
        )
        logits = model(
            normalized
        )
        dense = torch.softmax(
            logits / temperature,
            dim=1,
        )
        sparse = top2_prune(
            dense
        )

        dense_prediction = mix_forces(
            dense,
            expert_tensor,
        )
        sparse_prediction = mix_forces(
            sparse,
            expert_tensor,
        )

        dense_weights[
            start:end
        ] = dense.cpu().numpy()
        top2_weights[
            start:end
        ] = sparse.cpu().numpy()
        dense_forces[
            start:end
        ] = dense_prediction.cpu().numpy()
        top2_forces[
            start:end
        ] = sparse_prediction.cpu().numpy()

    return (
        dense_weights,
        top2_weights,
        dense_forces,
        top2_forces,
    )


@torch.no_grad()
def evaluate_validation_basic(
    model: nn.Module,
    records: Sequence[ShardRecord],
    store: ShardStore,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    temperature: float,
    chunk_atoms: int,
) -> Dict[str, float]:
    dense_structure_mse = []
    top2_structure_mse = []
    dense_total_sse = 0.0
    top2_total_sse = 0.0
    total_components = 0

    for record in records:
        arrays = store.get(
            record
        )

        (
            _,
            _,
            dense_forces,
            top2_forces,
        ) = predict_gate_for_structure(
            model=model,
            features=arrays["X_local"],
            expert_forces=arrays[
                "expert_forces"
            ],
            device=device,
            feature_mean=feature_mean,
            feature_std=feature_std,
            temperature=temperature,
            chunk_atoms=chunk_atoms,
        )

        reference = arrays[
            "mace_forces"
        ]
        dense_difference = (
            dense_forces - reference
        )
        top2_difference = (
            top2_forces - reference
        )

        dense_mse = float(
            np.mean(
                dense_difference ** 2
            )
        )
        top2_mse = float(
            np.mean(
                top2_difference ** 2
            )
        )

        dense_structure_mse.append(
            dense_mse
        )
        top2_structure_mse.append(
            top2_mse
        )

        dense_total_sse += float(
            np.sum(
                dense_difference ** 2,
                dtype=np.float64,
            )
        )
        top2_total_sse += float(
            np.sum(
                top2_difference ** 2,
                dtype=np.float64,
            )
        )
        total_components += (
            3 * record.n_atoms
        )

    return {
        "val_dense_structure_balanced_rmse": float(
            np.sqrt(
                np.mean(
                    dense_structure_mse
                )
            )
        ),
        "val_top2_structure_balanced_rmse": float(
            np.sqrt(
                np.mean(
                    top2_structure_mse
                )
            )
        ),
        "val_dense_atom_weighted_rmse": float(
            np.sqrt(
                dense_total_sse
                / total_components
            )
        ),
        "val_top2_atom_weighted_rmse": float(
            np.sqrt(
                top2_total_sse
                / total_components
            )
        ),
    }


def build_region_masks(
    arrays: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    n_atoms = len(
        arrays["mace_forces"]
    )
    phase = arrays["phase"].reshape(-1)
    center = arrays[
        "interface_center"
    ].reshape(-1)

    return {
        "all": np.ones(
            n_atoms,
            dtype=bool,
        ),
        "interface": center == 1,
        "phase1_bulk": (
            (phase == 1)
            & (center == 0)
        ),
        "phase2_bulk": (
            (phase == 2)
            & (center == 0)
        ),
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.n_atoms = 0
        self.n_components = 0
        self.sse = 0.0
        self.absolute_component_error = 0.0
        self.vector_squared_error = 0.0
        self.vector_absolute_error = 0.0
        self.cosine_sum = 0.0
        self.cosine_count = 0
        self.structure_mse_sum = 0.0
        self.structure_count = 0

    def update(
        self,
        prediction: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        mask = np.asarray(
            mask,
            dtype=bool,
        )

        if mask.sum() == 0:
            return

        prediction = prediction[mask]
        reference = reference[mask]
        difference = (
            prediction - reference
        )
        vector_error = np.linalg.norm(
            difference,
            axis=1,
        )

        self.n_atoms += len(prediction)
        self.n_components += (
            3 * len(prediction)
        )
        self.sse += float(
            np.sum(
                difference ** 2,
                dtype=np.float64,
            )
        )
        self.absolute_component_error += float(
            np.sum(
                np.abs(difference),
                dtype=np.float64,
            )
        )
        self.vector_squared_error += float(
            np.sum(
                vector_error ** 2,
                dtype=np.float64,
            )
        )
        self.vector_absolute_error += float(
            np.sum(
                vector_error,
                dtype=np.float64,
            )
        )

        prediction_norm = np.linalg.norm(
            prediction,
            axis=1,
        )
        reference_norm = np.linalg.norm(
            reference,
            axis=1,
        )
        denominator = (
            prediction_norm
            * reference_norm
        )
        valid = denominator > 1.0e-12

        if np.any(valid):
            cosine = np.clip(
                np.sum(
                    prediction[valid]
                    * reference[valid],
                    axis=1,
                )
                / denominator[valid],
                -1.0,
                1.0,
            )
            self.cosine_sum += float(
                np.sum(
                    cosine,
                    dtype=np.float64,
                )
            )
            self.cosine_count += int(
                valid.sum()
            )

        structure_mse = float(
            np.mean(
                difference ** 2
            )
        )
        self.structure_mse_sum += (
            structure_mse
        )
        self.structure_count += 1

    def finalize(self) -> Dict[str, float]:
        if self.n_atoms == 0:
            return {
                "n_atoms": 0,
                "n_structures": 0,
                "atom_weighted_rmse_component_eV_A": float("nan"),
                "structure_balanced_rmse_component_eV_A": float("nan"),
                "mae_component_eV_A": float("nan"),
                "rmse_vector_eV_A": float("nan"),
                "mae_vector_eV_A": float("nan"),
                "mean_cosine": float("nan"),
            }

        return {
            "n_atoms": self.n_atoms,
            "n_structures": self.structure_count,
            "atom_weighted_rmse_component_eV_A": float(
                np.sqrt(
                    self.sse
                    / self.n_components
                )
            ),
            "structure_balanced_rmse_component_eV_A": float(
                np.sqrt(
                    self.structure_mse_sum
                    / self.structure_count
                )
            ),
            "mae_component_eV_A": float(
                self.absolute_component_error
                / self.n_components
            ),
            "rmse_vector_eV_A": float(
                np.sqrt(
                    self.vector_squared_error
                    / self.n_atoms
                )
            ),
            "mae_vector_eV_A": float(
                self.vector_absolute_error
                / self.n_atoms
            ),
            "mean_cosine": float(
                self.cosine_sum
                / self.cosine_count
            )
            if self.cosine_count
            else float("nan"),
        }


def local_top1_oracle(
    expert_forces: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    squared_error = np.sum(
        (
            expert_forces
            - reference[None, :, :]
        ) ** 2,
        axis=2,
    )
    best_index = np.argmin(
        squared_error,
        axis=0,
    )
    atom_index = np.arange(
        len(reference)
    )
    prediction = expert_forces[
        best_index,
        atom_index,
        :,
    ]

    return (
        prediction.astype(
            np.float32,
            copy=False,
        ),
        best_index.astype(
            np.int16
        ),
    )


def local_top2_oracle(
    expert_forces: np.ndarray,
    reference: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_experts, n_atoms, _ = (
        expert_forces.shape
    )

    best_error = np.full(
        n_atoms,
        np.inf,
        dtype=np.float64,
    )
    best_prediction = np.zeros(
        (n_atoms, 3),
        dtype=np.float64,
    )
    best_a = np.full(
        n_atoms,
        -1,
        dtype=np.int16,
    )
    best_b = np.full(
        n_atoms,
        -1,
        dtype=np.int16,
    )
    best_weight_b = np.zeros(
        n_atoms,
        dtype=np.float64,
    )

    reference64 = reference.astype(
        np.float64,
        copy=False,
    )
    expert64 = expert_forces.astype(
        np.float64,
        copy=False,
    )

    for expert_a, expert_b in combinations(
        range(n_experts),
        2,
    ):
        force_a = expert64[
            expert_a
        ]
        force_b = expert64[
            expert_b
        ]
        direction = force_b - force_a
        denominator = np.sum(
            direction ** 2,
            axis=1,
        )
        numerator = np.sum(
            (
                reference64
                - force_a
            )
            * direction,
            axis=1,
        )

        weight_b = np.zeros(
            n_atoms,
            dtype=np.float64,
        )
        valid = denominator > 1.0e-20
        weight_b[valid] = (
            numerator[valid]
            / denominator[valid]
        )
        weight_b = np.clip(
            weight_b,
            0.0,
            1.0,
        )

        prediction = (
            force_a
            + weight_b[:, None]
            * direction
        )
        error = np.sum(
            (
                prediction
                - reference64
            ) ** 2,
            axis=1,
        )
        improved = error < best_error

        best_error[improved] = (
            error[improved]
        )
        best_prediction[improved] = (
            prediction[improved]
        )
        best_a[improved] = expert_a
        best_b[improved] = expert_b
        best_weight_b[improved] = (
            weight_b[improved]
        )

    return (
        best_prediction.astype(
            np.float32
        ),
        best_a,
        best_b,
        best_weight_b.astype(
            np.float32
        ),
    )


def compute_global_optimal_weights(
    records: Sequence[ShardRecord],
    store: ShardStore,
) -> np.ndarray:
    n_experts = len(
        EXPECTED_EXPERTS
    )
    quadratic = np.zeros(
        (n_experts, n_experts),
        dtype=np.float64,
    )
    linear = np.zeros(
        n_experts,
        dtype=np.float64,
    )

    for record in records:
        arrays = store.get(
            record
        )
        expert_forces = arrays[
            "expert_forces"
        ].astype(
            np.float64,
            copy=False,
        )
        reference = arrays[
            "mace_forces"
        ].astype(
            np.float64,
            copy=False,
        )

        flattened_experts = (
            expert_forces.reshape(
                n_experts,
                -1,
            )
        )
        flattened_reference = (
            reference.reshape(-1)
        )
        normalization = float(
            flattened_reference.size
        )

        quadratic += (
            flattened_experts
            @ flattened_experts.T
            / normalization
        )
        linear += (
            flattened_experts
            @ flattened_reference
            / normalization
        )

    quadratic /= len(records)
    linear /= len(records)

    def objective(
        weights: np.ndarray,
    ) -> float:
        return float(
            weights
            @ quadratic
            @ weights
            - 2.0
            * linear
            @ weights
        )

    def gradient(
        weights: np.ndarray,
    ) -> np.ndarray:
        return (
            2.0
            * quadratic
            @ weights
            - 2.0
            * linear
        )

    initial = np.zeros(
        n_experts,
        dtype=np.float64,
    )
    initial[-2] = 0.5
    initial[-1] = 0.5

    result = minimize(
        objective,
        x0=initial,
        jac=gradient,
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
        ] * n_experts,
        constraints={
            "type": "eq",
            "fun": lambda weights: (
                np.sum(weights) - 1.0
            ),
            "jac": lambda weights: np.ones_like(
                weights
            ),
        },
        options={
            "maxiter": 2000,
            "ftol": 1.0e-14,
            "disp": False,
        },
    )

    if not result.success:
        raise RuntimeError(
            "Global simplex optimization failed: "
            + str(result.message)
        )

    weights = np.clip(
        result.x,
        0.0,
        1.0,
    )
    weights /= weights.sum()

    return weights.astype(
        np.float64
    )


def net_force_row(
    predictor_name: str,
    record: ShardRecord,
    prediction: np.ndarray,
) -> Dict:
    net_force = np.sum(
        prediction,
        axis=0,
        dtype=np.float64,
    )
    norm = float(
        np.linalg.norm(
            net_force
        )
    )

    return {
        "source_file": record.source_file,
        "phase_pair": record.phase_pair,
        "split": record.split,
        "predictor": predictor_name,
        "n_atoms": record.n_atoms,
        "net_force_x_eV_A": float(
            net_force[0]
        ),
        "net_force_y_eV_A": float(
            net_force[1]
        ),
        "net_force_z_eV_A": float(
            net_force[2]
        ),
        "net_force_norm_eV_A": norm,
        "net_force_norm_per_atom_eV_A": (
            norm / record.n_atoms
        ),
    }


def evaluate_split_full(
    split_name: str,
    records: Sequence[ShardRecord],
    store: ShardStore,
    model: nn.Module,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    args: argparse.Namespace,
    output_root: Path,
) -> Tuple[
    List[Dict],
    List[Dict],
    List[Dict],
]:
    global_weights = compute_global_optimal_weights(
        records,
        store,
    )

    global_weight_rows = [
        {
            "split": split_name,
            "expert": expert,
            "weight": float(
                global_weights[index]
            ),
        }
        for index, expert in enumerate(
            EXPECTED_EXPERTS
        )
    ]

    predictor_names = [
        *EXPECTED_EXPERTS,
        "uniform_7",
        "global_optimal_simplex_oracle",
        "best_structure_expert_oracle",
        "trained_dense",
        "trained_top2_pruned",
        "local_top1_oracle",
        "local_top2_oracle",
    ]

    accumulators: Dict[
        Tuple[str, str],
        MetricAccumulator,
    ] = {
        (
            predictor,
            region,
        ): MetricAccumulator()
        for predictor in predictor_names
        for region in REGION_NAMES
    }

    net_force_rows: List[Dict] = []
    routing_weight_sum = {
        region: np.zeros(
            len(EXPECTED_EXPERTS),
            dtype=np.float64,
        )
        for region in REGION_NAMES
    }
    routing_top1_count = {
        region: np.zeros(
            len(EXPECTED_EXPERTS),
            dtype=np.int64,
        )
        for region in REGION_NAMES
    }
    routing_atom_count = {
        region: 0
        for region in REGION_NAMES
    }

    prediction_root = (
        output_root
        / "{}_predictions".format(
            split_name
        )
    )

    for structure_number, record in enumerate(
        records,
        start=1,
    ):
        arrays = store.get(
            record
        )
        reference = arrays[
            "mace_forces"
        ]
        expert_forces = arrays[
            "expert_forces"
        ]
        masks = build_region_masks(
            arrays
        )

        (
            dense_weights,
            top2_weights,
            dense_prediction,
            top2_prediction,
        ) = predict_gate_for_structure(
            model=model,
            features=arrays["X_local"],
            expert_forces=expert_forces,
            device=device,
            feature_mean=feature_mean,
            feature_std=feature_std,
            temperature=args.temperature,
            chunk_atoms=args.eval_chunk_atoms,
        )

        uniform_prediction = np.mean(
            expert_forces,
            axis=0,
        )
        global_prediction = np.tensordot(
            global_weights,
            expert_forces,
            axes=(0, 0),
        ).astype(
            np.float32
        )

        expert_structure_mse = np.mean(
            (
                expert_forces
                - reference[None, :, :]
            ) ** 2,
            axis=(1, 2),
        )
        best_structure_index = int(
            np.argmin(
                expert_structure_mse
            )
        )
        best_structure_prediction = (
            expert_forces[
                best_structure_index
            ]
        )

        (
            top1_oracle_prediction,
            top1_oracle_index,
        ) = local_top1_oracle(
            expert_forces,
            reference,
        )

        (
            top2_oracle_prediction,
            top2_oracle_a,
            top2_oracle_b,
            top2_oracle_weight_b,
        ) = local_top2_oracle(
            expert_forces,
            reference,
        )

        predictions: Dict[
            str,
            np.ndarray,
        ] = {
            EXPECTED_EXPERTS[index]:
                expert_forces[index]
            for index in range(
                len(EXPECTED_EXPERTS)
            )
        }
        predictions.update({
            "uniform_7": uniform_prediction,
            "global_optimal_simplex_oracle":
                global_prediction,
            "best_structure_expert_oracle":
                best_structure_prediction,
            "trained_dense": dense_prediction,
            "trained_top2_pruned":
                top2_prediction,
            "local_top1_oracle":
                top1_oracle_prediction,
            "local_top2_oracle":
                top2_oracle_prediction,
        })

        for predictor_name, prediction in (
            predictions.items()
        ):
            for region_name, mask in (
                masks.items()
            ):
                accumulators[
                    (
                        predictor_name,
                        region_name,
                    )
                ].update(
                    prediction,
                    reference,
                    mask,
                )

            net_force_rows.append(
                net_force_row(
                    predictor_name,
                    record,
                    prediction,
                )
            )

        dense_top1 = np.argmax(
            dense_weights,
            axis=1,
        )

        for region_name, mask in masks.items():
            if mask.sum() == 0:
                continue

            routing_weight_sum[
                region_name
            ] += np.sum(
                dense_weights[mask],
                axis=0,
                dtype=np.float64,
            )
            routing_top1_count[
                region_name
            ] += np.bincount(
                dense_top1[mask],
                minlength=len(
                    EXPECTED_EXPERTS
                ),
            )
            routing_atom_count[
                region_name
            ] += int(
                mask.sum()
            )

        if (
            args.save_test_predictions
            and split_name == "test"
        ):
            relative = Path(
                record.source_file
            )
            destination = (
                prediction_root
                / relative.parent
                / (
                    relative.stem
                    + "_predictions.npz"
                )
            )
            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savez_compressed(
                destination,
                expert_names=np.asarray(
                    EXPECTED_EXPERTS
                ),
                dense_weights=dense_weights,
                top2_weights=top2_weights,
                dense_forces=dense_prediction,
                top2_forces=top2_prediction,
                mace_forces=reference,
                local_top1_oracle_index=
                    top1_oracle_index,
                local_top2_oracle_a=
                    top2_oracle_a,
                local_top2_oracle_b=
                    top2_oracle_b,
                local_top2_oracle_weight_b=
                    top2_oracle_weight_b,
                best_structure_expert_index=
                    np.asarray(
                        best_structure_index,
                        dtype=np.int16,
                    ),
            )

        print(
            "  [{} {:03d}/{:03d}] {} | "
            "best={} | dense RMSE={:.5f} | "
            "top2 RMSE={:.5f}".format(
                split_name,
                structure_number,
                len(records),
                record.phase_pair,
                EXPECTED_EXPERTS[
                    best_structure_index
                ],
                np.sqrt(
                    np.mean(
                        (
                            dense_prediction
                            - reference
                        ) ** 2
                    )
                ),
                np.sqrt(
                    np.mean(
                        (
                            top2_prediction
                            - reference
                        ) ** 2
                    )
                ),
            )
        )

    metric_rows: List[Dict] = []

    for predictor_name in predictor_names:
        for region_name in REGION_NAMES:
            row = {
                "split": split_name,
                "predictor": predictor_name,
                "region": region_name,
            }
            row.update(
                accumulators[
                    (
                        predictor_name,
                        region_name,
                    )
                ].finalize()
            )
            metric_rows.append(
                row
            )

    routing_rows: List[Dict] = []

    for region_name in REGION_NAMES:
        count = routing_atom_count[
            region_name
        ]

        if count == 0:
            continue

        for expert_index, expert in enumerate(
            EXPECTED_EXPERTS
        ):
            routing_rows.append({
                "split": split_name,
                "region": region_name,
                "expert": expert,
                "mean_dense_weight": float(
                    routing_weight_sum[
                        region_name
                    ][expert_index]
                    / count
                ),
                "top1_count": int(
                    routing_top1_count[
                        region_name
                    ][expert_index]
                ),
                "top1_fraction": float(
                    routing_top1_count[
                        region_name
                    ][expert_index]
                    / count
                ),
                "n_atoms": count,
            })

    return (
        metric_rows,
        net_force_rows,
        routing_rows
        + global_weight_rows,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_rmse: float,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    args: argparse.Namespace,
    input_dim: int,
) -> None:
    payload = {
        "epoch": int(epoch),
        "best_val_structure_balanced_rmse":
            float(best_val_rmse),
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "feature_mean":
            torch.from_numpy(
                feature_mean
            ),
        "feature_std":
            torch.from_numpy(
                feature_std
            ),
        "expert_names":
            EXPECTED_EXPERTS,
        "input_dim":
            int(input_dim),
        "hidden_dims":
            list(args.hidden_dims),
        "dropout":
            float(args.dropout),
        "temperature":
            float(args.temperature),
        "training_arguments":
            vars(args),
    }

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )
    torch.save(
        payload,
        temporary,
    )
    temporary.replace(
        path
    )


def print_split_summary(
    records: Sequence[ShardRecord],
) -> None:
    total_atoms = sum(
        record.n_atoms
        for record in records
    )

    print(
        "  Structures: {:3d} | Atoms: {:8,d} | "
        "Phase pairs: {:3d} | Best expert counts: {}".format(
            len(records),
            total_atoms,
            len(
                {
                    record.phase_pair
                    for record in records
                }
            ),
            {
                expert: sum(
                    record.best_expert_all
                    == expert
                    for record in records
                )
                for expert in [
                    "Tersoff",
                    "Vashishta",
                ]
            },
        )
    )


def main() -> None:
    args = parse_args()

    if args.temperature <= 0.0:
        raise SystemExit(
            "ERROR: --temperature must be positive."
        )

    if args.batch_structures <= 0:
        raise SystemExit(
            "ERROR: --batch_structures must be positive."
        )

    torch.set_num_threads(
        max(
            1,
            args.torch_threads,
        )
    )
    set_global_seed(
        args.seed
    )

    dataset_root = (
        args.dataset_root
        .expanduser()
        .resolve()
    )
    output_root = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not dataset_root.is_dir():
        raise SystemExit(
            "ERROR: dataset root not found: {}".format(
                dataset_root
            )
        )

    if (
        output_root.exists()
        and any(
            output_root.iterdir()
        )
    ):
        if not args.overwrite:
            raise SystemExit(
                "ERROR: output directory is not empty: {}. "
                "Use --overwrite or a new directory.".format(
                    output_root
                )
            )

        import shutil
        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device(
        args.device
    )

    records = load_records(
        dataset_root
    )

    if args.preserve_manifest_split:
        if any(record.split not in {"train", "val", "test"} for record in records):
            raise SystemExit(
                "ERROR: --preserve_manifest_split requires train/val/test in dataset_manifest.csv"
            )
        assignment = {record.index: record.split for record in records}
        expected = {"train": 369, "val": 38, "test": 105}  # run-specific: expanded_holdout40_v1
        observed = {
            split: sum(value == split for value in assignment.values())
            for split in ("train", "val", "test")
        }
        if observed != expected:
            raise SystemExit(
                f"ERROR: frozen split mismatch: observed={observed}, expected={expected}"
            )
    else:
        fractions = {
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": args.test_fraction,
        }
        assignment = assign_grouped_splits(
            records,
            fractions,
            args.seed,
        )

    for record in records:
        record.split = assignment[
            record.index
        ]

    split_records = {
        split: [
            record
            for record in records
            if record.split == split
        ]
        for split in [
            "train",
            "val",
            "test",
        ]
    }

    split_rows = []

    for record in records:
        split_rows.append({
            "source_file": record.source_file,
            "shard_file": str(
                record.shard_path.relative_to(
                    dataset_root
                )
            ),
            "phase_pair": record.phase_pair,
            "duplicate_group":
                record.duplicate_group,
            "n_atoms": record.n_atoms,
            "best_expert_all":
                record.best_expert_all,
            "split": record.split,
        })

    write_csv(
        output_root / "dataset_split.csv",
        split_rows,
    )

    cache_in_memory = (
        not args.no_cache_in_memory
    )

    print("=" * 112)
    print("TRAIN LOCAL GATE FROM EXPERT FORCES")
    print("Dataset root:       {}".format(dataset_root))
    print("Output root:        {}".format(output_root))
    print("Device:             {}".format(device))

    if device.type == "cuda":
        print(
            "GPU:                {}".format(
                torch.cuda.get_device_name(0)
            )
        )

    print("Cache in memory:    {}".format(cache_in_memory))
    print("Grouped split:")
    print("Train")
    print_split_summary(split_records["train"])
    print("Validation")
    print_split_summary(split_records["val"])
    print("Test")
    print_split_summary(split_records["test"])
    print("=" * 112)

    store = ShardStore(
        records,
        cache_in_memory=cache_in_memory,
    )

    print(
        "Feature dimension:   {}".format(
            store.feature_dim
        )
    )
    print(
        "Estimated array RAM: {:.2f} GiB".format(
            store.estimated_cache_bytes
            / 1024 ** 3
        )
    )

    feature_mean, feature_std = (
        compute_feature_statistics(
            split_records["train"],
            store,
        )
    )

    np.savez(
        output_root
        / "feature_normalization.npz",
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    feature_mean_tensor = torch.from_numpy(
        feature_mean
    ).to(
        device
    )
    feature_std_tensor = torch.from_numpy(
        feature_std
    ).to(
        device
    )

    model = LocalGateMLP(
        input_dim=int(
            store.feature_dim
        ),
        hidden_dims=args.hidden_dims,
        n_experts=len(
            EXPECTED_EXPERTS
        ),
        dropout=args.dropout,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(
            5,
            args.patience // 4,
        ),
        threshold=args.min_delta,
        min_lr=1.0e-6,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(
            args.amp
            and device.type == "cuda"
        )
    )

    model_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    print(
        "Trainable parameters: {:,}".format(
            model_parameters
        )
    )

    history_rows: List[Dict] = []
    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = (
        output_root
        / "best_local_gate.pt"
    )

    start_training = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        start_epoch = time.time()

        training_metrics = train_one_epoch(
            model=model,
            train_records=split_records[
                "train"
            ],
            store=store,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            feature_mean=feature_mean_tensor,
            feature_std=feature_std_tensor,
            args=args,
            epoch_seed=(
                args.seed + epoch
            ),
        )

        row: Dict = {
            "epoch": epoch,
            **training_metrics,
            "learning_rate": optimizer.param_groups[
                0
            ]["lr"],
            "epoch_seconds": float(
                time.time() - start_epoch
            ),
        }

        if (
            epoch % args.eval_every == 0
            or epoch == 1
        ):
            validation_metrics = (
                evaluate_validation_basic(
                    model=model,
                    records=split_records[
                        "val"
                    ],
                    store=store,
                    device=device,
                    feature_mean=feature_mean_tensor,
                    feature_std=feature_std_tensor,
                    temperature=args.temperature,
                    chunk_atoms=args.eval_chunk_atoms,
                )
            )
            row.update(
                validation_metrics
            )

            current_val = validation_metrics[
                "val_dense_structure_balanced_rmse"
            ]
            scheduler.step(
                current_val
            )

            improved = (
                current_val
                < best_val_rmse
                - args.min_delta
            )

            if improved:
                best_val_rmse = current_val
                best_epoch = epoch
                epochs_without_improvement = 0

                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_rmse=
                        best_val_rmse,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    args=args,
                    input_dim=int(
                        store.feature_dim
                    ),
                )
            else:
                epochs_without_improvement += (
                    args.eval_every
                )

            print(
                "Epoch {:04d} | train MSE={:.7f} | "
                "val dense RMSE={:.6f} | "
                "val top2 RMSE={:.6f} | "
                "entropy={:.4f} | lr={:.3e} | "
                "{} | {:.2f} s".format(
                    epoch,
                    training_metrics[
                        "train_force_loss"
                    ],
                    validation_metrics[
                        "val_dense_structure_balanced_rmse"
                    ],
                    validation_metrics[
                        "val_top2_structure_balanced_rmse"
                    ],
                    training_metrics[
                        "train_normalized_entropy"
                    ],
                    optimizer.param_groups[
                        0
                    ]["lr"],
                    "BEST"
                    if improved
                    else (
                        "patience {}/{}".format(
                            epochs_without_improvement,
                            args.patience,
                        )
                    ),
                    time.time() - start_epoch,
                )
            )
        else:
            print(
                "Epoch {:04d} | train MSE={:.7f} | "
                "entropy={:.4f} | lr={:.3e} | "
                "{:.2f} s".format(
                    epoch,
                    training_metrics[
                        "train_force_loss"
                    ],
                    training_metrics[
                        "train_normalized_entropy"
                    ],
                    optimizer.param_groups[
                        0
                    ]["lr"],
                    time.time() - start_epoch,
                )
            )

        history_rows.append(
            row
        )
        write_csv(
            output_root
            / "training_history.csv",
            history_rows,
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print(
                "Early stopping at epoch {}. "
                "Best epoch: {}.".format(
                    epoch,
                    best_epoch,
                )
            )
            break

    if not checkpoint_path.is_file():
        raise RuntimeError(
            "Training ended without saving a checkpoint."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print("\n" + "=" * 112)
    print(
        "TRAINING COMPLETE | best epoch={} | "
        "best validation structure-balanced RMSE={:.6f} eV/A".format(
            best_epoch,
            best_val_rmse,
        )
    )
    print("=" * 112)

    all_metric_rows: List[Dict] = []
    all_net_force_rows: List[Dict] = []
    all_routing_rows: List[Dict] = []

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        print(
            "\nFULL EVALUATION: {}".format(
                split_name.upper()
            )
        )

        (
            metric_rows,
            net_force_rows,
            routing_rows,
        ) = evaluate_split_full(
            split_name=split_name,
            records=split_records[
                split_name
            ],
            store=store,
            model=model,
            device=device,
            feature_mean=feature_mean_tensor,
            feature_std=feature_std_tensor,
            args=args,
            output_root=output_root,
        )

        all_metric_rows.extend(
            metric_rows
        )
        all_net_force_rows.extend(
            net_force_rows
        )
        all_routing_rows.extend(
            routing_rows
        )

    write_csv(
        output_root
        / "evaluation_metrics.csv",
        all_metric_rows,
    )
    write_csv(
        output_root
        / "net_force_audit.csv",
        all_net_force_rows,
    )
    write_csv(
        output_root
        / "routing_summary.csv",
        all_routing_rows,
    )

    evaluation_lookup = {
        (
            row["split"],
            row["predictor"],
            row["region"],
        ): row
        for row in all_metric_rows
    }

    print("\n" + "=" * 112)
    print("TEST-SET SUMMARY: STRUCTURE-BALANCED COMPONENT RMSE [eV/A]")
    print("=" * 112)

    summary_predictors = [
        *EXPECTED_EXPERTS,
        "uniform_7",
        "global_optimal_simplex_oracle",
        "best_structure_expert_oracle",
        "trained_dense",
        "trained_top2_pruned",
        "local_top1_oracle",
        "local_top2_oracle",
    ]

    for predictor in summary_predictors:
        all_row = evaluation_lookup[
            (
                "test",
                predictor,
                "all",
            )
        ]
        interface_row = evaluation_lookup[
            (
                "test",
                predictor,
                "interface",
            )
        ]

        print(
            "{:<31} all={:.6f}  interface={:.6f}".format(
                predictor,
                all_row[
                    "structure_balanced_rmse_component_eV_A"
                ],
                interface_row[
                    "structure_balanced_rmse_component_eV_A"
                ],
            )
        )

    run_info = {
        "dataset_root": str(
            dataset_root
        ),
        "output_root": str(
            output_root
        ),
        "seed": args.seed,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else None
        ),
        "expert_names": EXPECTED_EXPERTS,
        "feature_dim": int(
            store.feature_dim
        ),
        "hidden_dims": list(
            args.hidden_dims
        ),
        "model_parameters":
            model_parameters,
        "best_epoch": best_epoch,
        "best_validation_structure_balanced_rmse_eV_A":
            best_val_rmse,
        "training_seconds": float(
            time.time() - start_training
        ),
        "split_counts": {
            split: {
                "structures": len(
                    split_records[split]
                ),
                "atoms": sum(
                    record.n_atoms
                    for record in split_records[
                        split
                    ]
                ),
                "phase_pairs": len(
                    {
                        record.phase_pair
                        for record in split_records[
                            split
                        ]
                    }
                ),
            }
            for split in [
                "train",
                "val",
                "test",
            ]
        },
        "training_objective": (
            "Structure-balanced atomwise force loss for the dense "
            "seven-expert softmax gate."
        ),
        "top2_evaluation": (
            "Top-2 pruning and renormalization of the trained dense "
            "gate weights; this is not separately trained."
        ),
        "conservation_warning": (
            "Atom-dependent force mixtures are not automatically "
            "conservative and may have nonzero net force. See "
            "net_force_audit.csv."
        ),
        "files": {
            "checkpoint":
                "best_local_gate.pt",
            "split":
                "dataset_split.csv",
            "history":
                "training_history.csv",
            "metrics":
                "evaluation_metrics.csv",
            "routing":
                "routing_summary.csv",
            "net_force":
                "net_force_audit.csv",
        },
    }

    (
        output_root
        / "run_info.json"
    ).write_text(
        json.dumps(
            run_info,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print("\nSaved:")
    print(
        output_root
        / "best_local_gate.pt"
    )
    print(
        output_root
        / "dataset_split.csv"
    )
    print(
        output_root
        / "training_history.csv"
    )
    print(
        output_root
        / "evaluation_metrics.csv"
    )
    print(
        output_root
        / "routing_summary.csv"
    )
    print(
        output_root
        / "net_force_audit.csv"
    )
    print(
        output_root
        / "run_info.json"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise
