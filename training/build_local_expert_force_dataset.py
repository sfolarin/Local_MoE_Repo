#!/usr/bin/env python3
"""
build_local_expert_force_dataset.py

Build the atom-resolved training dataset for the Dynamic Local MoE project.

For every MACE-labeled silica heterostructure, this script:

  1. reads the complete periodic structure and its MACE reference labels;
  2. computes atom-centered SOAP descriptors;
  3. appends a Si/O one-hot encoding to form the local gate input;
  4. evaluates all seven classical experts on the same complete structure;
  5. stores reference and expert energies, forces, stresses, labels, and metadata
     in one compressed NPZ shard per structure;
  6. writes a dataset manifest, failure log, and reproducibility metadata.

The old global gate is never loaded and its weights are never used.

Local gate input
----------------
    X_local[i] = [SOAP_i, one_hot(Si/O)]

With r_cut=5.0 A, n_max=6, l_max=4 and species=["Si", "O"], the expected
SOAP dimension is 390, giving 392 local features after adding the two
species indicators.

Important
---------
- phase, interface_center, and mace_halo are preserved for analysis only.
  They are not included in X_local.
- Expert energies are stored in their raw native energy references.
  They must not be compared or mixed across experts without a later,
  explicitly defined energy-alignment procedure.
- The immediate training target is the MACE atom-resolved force field.
- One shard is written per structure to avoid holding the full dataset in RAM.

Example pilot
-------------
python3 -u build_local_expert_force_dataset.py \
    --input_root MACE_Labeled_Interfaces_MPA0 \
    --output_dir Local_Expert_Force_Dataset_MPA0_pilot \
    --core "$CORE" \
    --pot_dir "$POT_DIR" \
    --lmp_lib "$LMP_LIB" \
    --lmp_python "$LMP_PYTHON" \
    --max_files 3 \
    --overwrite

Example full run
----------------
python3 -u build_local_expert_force_dataset.py \
    --input_root MACE_Labeled_Interfaces_MPA0 \
    --output_dir Local_Expert_Force_Dataset_MPA0 \
    --core "$CORE" \
    --pot_dir "$POT_DIR" \
    --lmp_lib "$LMP_LIB" \
    --lmp_python "$LMP_PYTHON"
"""

from __future__ import annotations
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
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase.io import read


EXPERT_NAMES = [
    "ML-BKS",
    "Original_BKS",
    "Chik_BKS",
    "PSO1_BKS",
    "PSO2_BKS",
    "Tersoff",
    "Vashishta",
]

ANALYSIS_LABELS = [
    "phase",
    "interface_center",
    "mace_halo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build atom-resolved SOAP + MACE reference + seven-expert "
            "force shards for local-gate training."
        )
    )

    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root of the MACE-labeled EXTXYZ dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output dataset root.",
    )
    parser.add_argument(
        "--pattern",
        default="*_mace_mpa0.extxyz",
        help="Recursive input filename pattern.",
    )
    parser.add_argument(
        "--name_contains",
        default=None,
        help="Optional substring filter applied to relative input paths.",
    )
    parser.add_argument(
        "--file_list",
        type=Path,
        default=None,
        help=(
            "Optional text file containing relative or absolute input paths, "
            "one per line. Blank lines and lines beginning with # are ignored."
        ),
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Process only the first N selected structures; <=0 means all.",
    )

    parser.add_argument(
        "--core",
        type=Path,
        default=os.environ.get("CORE"),
        help=(
            "Directory containing moe_persistent_calculator.py. "
            "Defaults to $CORE."
        ),
    )
    parser.add_argument(
        "--pot_dir",
        type=Path,
        default=os.environ.get("POT_DIR"),
        help="Directory containing Tersoff.pot and Vashishta.pot.",
    )
    parser.add_argument(
        "--lmp_lib",
        type=Path,
        default=os.environ.get("LMP_LIB"),
        help="Path to liblammps.so.0.",
    )
    parser.add_argument(
        "--lmp_python",
        type=Path,
        default=os.environ.get("LMP_PYTHON"),
        help="LAMMPS Python binding directory.",
    )

    parser.add_argument("--soap_rcut", type=float, default=5.0)
    parser.add_argument("--soap_nmax", type=int, default=6)
    parser.add_argument("--soap_lmax", type=int, default=4)
    parser.add_argument(
        "--soap_jobs",
        type=int,
        default=1,
        help="Number of DScribe jobs used inside one structure.",
    )
    parser.add_argument(
        "--expected_soap_dim",
        type=int,
        default=390,
        help="Abort if SOAP dimension differs; set <=0 to disable.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Storage dtype for SOAP and force arrays.",
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed.",
    )

    parser.add_argument(
        "--mace_manifest",
        type=Path,
        default=None,
        help=(
            "Optional MACE manifest. Defaults to input_root/mace_manifest.csv "
            "when present."
        ),
    )
    parser.add_argument(
        "--duplicate_csv",
        type=Path,
        default=None,
        help=(
            "Optional candidate_duplicate_structures.csv. Duplicate group IDs "
            "are stored as metadata."
        ),
    )
    parser.add_argument(
        "--unique_only",
        action="store_true",
        help=(
            "Keep only the first structure in every candidate duplicate group. "
            "Requires --duplicate_csv."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and replace existing NPZ shards.",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop at the first failed structure.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=1,
        help="Rewrite manifest/failure files every N processed structures.",
    )

    return parser.parse_args()


def require_path(
    value: Optional[Path],
    label: str,
    kind: str,
) -> Path:
    if value is None:
        raise SystemExit(
            "ERROR: missing {}. Supply its command-line option or "
            "environment variable.".format(label)
        )

    path = Path(value).expanduser().resolve()

    if kind == "file" and not path.is_file():
        raise SystemExit("ERROR: {} not found: {}".format(label, path))
    if kind == "dir" and not path.is_dir():
        raise SystemExit("ERROR: {} not found: {}".format(label, path))

    return path


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        if not path.exists():
            path.write_text("")
        return

    fieldnames: List[str] = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def load_existing_rows(path: Path, key: str) -> Dict[str, Dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    return {
        str(row[key]): row
        for row in rows
        if row.get(key)
    }


def load_file_list(
    file_list: Path,
    input_root: Path,
) -> List[Path]:
    selected: List[Path] = []

    for raw_line in file_list.read_text().splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        candidate = Path(line).expanduser()

        if not candidate.is_absolute():
            candidate = input_root / candidate

        candidate = candidate.resolve()

        if not candidate.is_file():
            raise FileNotFoundError(
                "Input listed in {} does not exist: {}".format(
                    file_list,
                    candidate,
                )
            )

        selected.append(candidate)

    return selected


def select_input_files(
    input_root: Path,
    pattern: str,
    name_contains: Optional[str],
    file_list: Optional[Path],
    max_files: int,
) -> List[Path]:
    if file_list is not None:
        files = load_file_list(
            file_list.expanduser().resolve(),
            input_root,
        )
    else:
        files = sorted(input_root.rglob(pattern))

    if name_contains:
        files = [
            path
            for path in files
            if name_contains in str(path.relative_to(input_root))
        ]

    if max_files > 0:
        files = files[:max_files]

    return files


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_mace_path_maps(
    input_root: Path,
    manifest_path: Optional[Path],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Returns:
      output_relative -> original source relative
      output_relative -> complete manifest row
    """
    if manifest_path is None:
        candidate = input_root / "mace_manifest.csv"
        manifest_path = candidate if candidate.is_file() else None

    if manifest_path is None:
        return {}, {}

    rows = read_csv_rows(manifest_path)

    output_to_source: Dict[str, str] = {}
    output_to_row: Dict[str, Dict[str, str]] = {}

    for row in rows:
        output_file = row.get("output_file")
        source_file = row.get("source_file")

        if output_file:
            output_to_row[output_file] = row
            if source_file:
                output_to_source[output_file] = source_file

    return output_to_source, output_to_row


def build_duplicate_map(
    duplicate_csv: Optional[Path],
) -> Tuple[Dict[str, str], Dict[str, bool]]:
    """
    Returns:
      original source relative -> group ID
      original source relative -> is representative
    """
    if duplicate_csv is None:
        return {}, {}

    rows = read_csv_rows(duplicate_csv)
    grouped: Dict[str, List[str]] = {}

    for row in rows:
        source = row.get("source_file")
        group = row.get("candidate_duplicate_group")

        if source and group:
            grouped.setdefault(group, []).append(source)

    source_to_group: Dict[str, str] = {}
    source_to_representative: Dict[str, bool] = {}

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: min(item[1]),
    )

    for group_number, (_, sources) in enumerate(
        ordered_groups,
        start=1,
    ):
        group_id = "duplicate_group_{:04d}".format(group_number)
        ordered_sources = sorted(set(sources))
        representative = ordered_sources[0]

        for source in ordered_sources:
            source_to_group[source] = group_id
            source_to_representative[source] = (
                source == representative
            )

    return source_to_group, source_to_representative


def validate_structure_and_reference(atoms, source: Path) -> np.ndarray:
    if len(atoms) == 0:
        raise ValueError("{} contains no atoms.".format(source))

    symbols = np.asarray(
        atoms.get_chemical_symbols(),
        dtype="<U3",
    )

    unsupported = sorted(
        set(symbols.tolist()) - {"Si", "O"}
    )

    if unsupported:
        raise ValueError(
            "{} contains unsupported species: {}".format(
                source,
                unsupported,
            )
        )

    if not np.all(atoms.pbc):
        raise ValueError(
            "{} is not fully periodic: PBC={}".format(
                source,
                atoms.pbc,
            )
        )

    if not np.all(np.isfinite(atoms.positions)):
        raise ValueError(
            "{} contains non-finite positions.".format(source)
        )

    if atoms.get_volume() <= 0.0:
        raise ValueError(
            "{} has non-positive cell volume.".format(source)
        )

    if "mace_forces" not in atoms.arrays:
        raise ValueError(
            "{} lacks required per-atom array 'mace_forces'.".format(
                source
            )
        )

    reference_forces = np.asarray(
        atoms.arrays["mace_forces"],
        dtype=np.float64,
    )

    if reference_forces.shape != (len(atoms), 3):
        raise ValueError(
            "{} has MACE force shape {}, expected ({}, 3).".format(
                source,
                reference_forces.shape,
                len(atoms),
            )
        )

    if not np.all(np.isfinite(reference_forces)):
        raise ValueError(
            "{} contains non-finite MACE forces.".format(source)
        )

    return symbols


def get_reference_energy(atoms) -> float:
    for key in [
        "mace_energy_eV",
        "energy",
        "free_energy",
    ]:
        if key in atoms.info:
            value = float(atoms.info[key])
            if np.isfinite(value):
                return value

    try:
        value = float(atoms.get_potential_energy())
        if np.isfinite(value):
            return value
    except Exception:
        pass

    raise ValueError(
        "No finite MACE reference energy found in atoms.info "
        "or attached SinglePointCalculator."
    )


def get_reference_stress(atoms) -> np.ndarray:
    for key in [
        "mace_stress_voigt_eV_A3",
        "stress",
    ]:
        if key in atoms.info:
            value = np.asarray(
                atoms.info[key],
                dtype=np.float64,
            ).reshape(-1)

            if value.shape == (6,) and np.all(np.isfinite(value)):
                return value

    try:
        value = np.asarray(
            atoms.get_stress(voigt=True),
            dtype=np.float64,
        ).reshape(-1)

        if value.shape == (6,) and np.all(np.isfinite(value)):
            return value
    except Exception:
        pass

    return np.full(6, np.nan, dtype=np.float64)


class LocalSOAPDescriptor:
    def __init__(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        n_jobs: int,
        expected_dim: int,
    ) -> None:
        try:
            from dscribe.descriptors import SOAP
        except ImportError as exc:
            raise ImportError(
                "DScribe is required. Install or activate the environment "
                "containing dscribe."
            ) from exc

        self.r_cut = float(r_cut)
        self.n_max = int(n_max)
        self.l_max = int(l_max)
        self.n_jobs = int(n_jobs)

        self.soap = SOAP(
            species=["Si", "O"],
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            average="off",
            sparse=False,
            periodic=True,
        )

        self.soap_dim = int(
            self.soap.get_number_of_features()
        )
        self.feature_dim = self.soap_dim + 2

        if expected_dim > 0 and self.soap_dim != expected_dim:
            raise RuntimeError(
                "SOAP dimension is {}, expected {}. Refusing to build an "
                "incompatible dataset.".format(
                    self.soap_dim,
                    expected_dim,
                )
            )

    def compute(
        self,
        atoms,
        symbols: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        soap = np.asarray(
            self.soap.create(
                atoms,
                n_jobs=self.n_jobs,
            ),
            dtype=np.float64,
        )

        expected_shape = (
            len(atoms),
            self.soap_dim,
        )

        if soap.shape != expected_shape:
            raise RuntimeError(
                "SOAP shape is {}, expected {}.".format(
                    soap.shape,
                    expected_shape,
                )
            )

        one_hot = np.zeros(
            (len(atoms), 2),
            dtype=np.float64,
        )
        one_hot[symbols == "Si", 0] = 1.0
        one_hot[symbols == "O", 1] = 1.0

        local_features = np.concatenate(
            [soap, one_hot],
            axis=1,
        )

        return local_features, one_hot


def force_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)

    if mask.shape != (len(reference),):
        raise ValueError("Region mask has incorrect shape.")

    if mask.sum() == 0:
        return {
            "n": 0,
            "rmse_component": float("nan"),
            "mae_component": float("nan"),
            "rmse_vector": float("nan"),
            "mae_vector": float("nan"),
            "mean_cosine": float("nan"),
        }

    prediction = predicted[mask]
    target = reference[mask]
    difference = prediction - target
    vector_error = np.linalg.norm(
        difference,
        axis=1,
    )

    prediction_norm = np.linalg.norm(
        prediction,
        axis=1,
    )
    target_norm = np.linalg.norm(
        target,
        axis=1,
    )

    denominator = prediction_norm * target_norm
    valid = denominator > 1.0e-12
    cosine = np.full(
        len(prediction),
        np.nan,
        dtype=np.float64,
    )

    cosine[valid] = np.clip(
        np.sum(
            prediction[valid] * target[valid],
            axis=1,
        ) / denominator[valid],
        -1.0,
        1.0,
    )

    return {
        "n": int(mask.sum()),
        "rmse_component": float(
            np.sqrt(np.mean(difference ** 2))
        ),
        "mae_component": float(
            np.mean(np.abs(difference))
        ),
        "rmse_vector": float(
            np.sqrt(np.mean(vector_error ** 2))
        ),
        "mae_vector": float(
            np.mean(vector_error)
        ),
        "mean_cosine": float(
            np.nanmean(cosine)
        ),
    }


def region_masks(atoms) -> Dict[str, np.ndarray]:
    n_atoms = len(atoms)
    masks: Dict[str, np.ndarray] = {
        "all": np.ones(n_atoms, dtype=bool),
    }

    phase = atoms.arrays.get("phase")
    center = atoms.arrays.get("interface_center")

    if center is not None:
        center = np.asarray(center).reshape(-1)

        if len(center) != n_atoms:
            raise ValueError(
                "interface_center length does not match atom count."
            )

        masks["interface"] = center == 1

    if phase is not None and center is not None:
        phase = np.asarray(phase).reshape(-1)

        if len(phase) != n_atoms:
            raise ValueError(
                "phase length does not match atom count."
            )

        masks["phase1_bulk"] = (
            (phase == 1) & (center == 0)
        )
        masks["phase2_bulk"] = (
            (phase == 2) & (center == 0)
        )

    return masks


def evaluate_experts(
    atoms,
    persistent_lammps_class,
    pot_dir: Path,
    lmp_lib: Path,
    lmp_python: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    energy_per_atom: List[float] = []
    forces: List[np.ndarray] = []
    stresses: List[np.ndarray] = []

    for expert_name in EXPERT_NAMES:
        engine = None

        try:
            engine = persistent_lammps_class(
                expert_name=expert_name,
                atoms=atoms,
                pot_dir=str(pot_dir),
                lmp_lib=str(lmp_lib),
                lmp_python=str(lmp_python),
            )

            energy, expert_forces, expert_stress = (
                engine.compute(atoms)
            )

            energy = float(energy)
            expert_forces = np.asarray(
                expert_forces,
                dtype=np.float64,
            )
            expert_stress = np.asarray(
                expert_stress,
                dtype=np.float64,
            ).reshape(-1)

            if not np.isfinite(energy):
                raise ValueError(
                    "{} returned non-finite energy.".format(
                        expert_name
                    )
                )

            if expert_forces.shape != (len(atoms), 3):
                raise ValueError(
                    "{} force shape is {}, expected ({}, 3).".format(
                        expert_name,
                        expert_forces.shape,
                        len(atoms),
                    )
                )

            if not np.all(np.isfinite(expert_forces)):
                raise ValueError(
                    "{} returned non-finite forces.".format(
                        expert_name
                    )
                )

            if (
                expert_stress.shape != (6,)
                or not np.all(np.isfinite(expert_stress))
            ):
                raise ValueError(
                    "{} returned invalid stress {}.".format(
                        expert_name,
                        expert_stress,
                    )
                )

            energy_per_atom.append(energy)
            forces.append(expert_forces)
            stresses.append(expert_stress)

        finally:
            if engine is not None:
                engine.close()

    return (
        np.asarray(
            energy_per_atom,
            dtype=np.float64,
        ),
        np.stack(
            forces,
            axis=0,
        ),
        np.stack(
            stresses,
            axis=0,
        ),
    )


def atomic_save_npz(
    path: Path,
    compressed: bool,
    **arrays,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp.npz"
    )

    if compressed:
        np.savez_compressed(
            temporary,
            **arrays,
        )
    else:
        np.savez(
            temporary,
            **arrays,
        )

    temporary.replace(path)


def validate_saved_shard(
    path: Path,
    n_atoms: int,
    n_features: int,
) -> None:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        required = {
            "X_local",
            "mace_forces",
            "expert_forces",
            "expert_energy_per_atom_raw",
            "expert_stress",
            "positions",
            "cell",
            "atomic_numbers",
            "expert_names",
        }

        missing = required - set(data.files)

        if missing:
            raise RuntimeError(
                "Saved shard is missing arrays: {}".format(
                    sorted(missing)
                )
            )

        if data["X_local"].shape != (
            n_atoms,
            n_features,
        ):
            raise RuntimeError(
                "Saved X_local shape is {}, expected ({}, {}).".format(
                    data["X_local"].shape,
                    n_atoms,
                    n_features,
                )
            )

        if data["mace_forces"].shape != (
            n_atoms,
            3,
        ):
            raise RuntimeError(
                "Saved MACE force shape is incorrect."
            )

        if data["expert_forces"].shape != (
            len(EXPERT_NAMES),
            n_atoms,
            3,
        ):
            raise RuntimeError(
                "Saved expert force shape is incorrect."
            )


def main() -> None:
    args = parse_args()

    input_root = require_path(
        args.input_root,
        "--input_root",
        "dir",
    )
    core = require_path(
        args.core,
        "--core / $CORE",
        "dir",
    )
    pot_dir = require_path(
        args.pot_dir,
        "--pot_dir / $POT_DIR",
        "dir",
    )
    lmp_lib = require_path(
        args.lmp_lib,
        "--lmp_lib / $LMP_LIB",
        "file",
    )
    lmp_python = require_path(
        args.lmp_python,
        "--lmp_python / $LMP_PYTHON",
        "dir",
    )

    calculator_path = (
        core / "moe_persistent_calculator.py"
    )

    if not calculator_path.is_file():
        raise SystemExit(
            "ERROR: calculator not found: {}".format(
                calculator_path
            )
        )

    for potential_name in [
        "Tersoff.pot",
        "Vashishta.pot",
    ]:
        potential_path = pot_dir / potential_name

        if not potential_path.is_file():
            raise SystemExit(
                "ERROR: potential file not found: {}".format(
                    potential_path
                )
            )

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    shard_root = output_root / "shards"
    shard_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.environ["MPICH_GPU_SUPPORT_ENABLED"] = "0"

    library_dir = str(lmp_lib.parent)
    current_ld = os.environ.get(
        "LD_LIBRARY_PATH",
        "",
    )

    if library_dir not in current_ld.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            library_dir
            + (":" + current_ld if current_ld else "")
        )

    for python_path in [
        str(lmp_python),
        str(core),
    ]:
        if python_path not in sys.path:
            sys.path.insert(0, python_path)

    from moe_persistent_calculator import (
        E_QUARTZ_REF,
        EXPERT_SETUP,
        PersistentLammps,
    )

    calculator_experts = list(EXPERT_SETUP.keys())

    if calculator_experts != EXPERT_NAMES:
        raise RuntimeError(
            "Expert order mismatch.\n"
            "Expected: {}\n"
            "Calculator: {}".format(
                EXPERT_NAMES,
                calculator_experts,
            )
        )

    soap_descriptor = LocalSOAPDescriptor(
        r_cut=args.soap_rcut,
        n_max=args.soap_nmax,
        l_max=args.soap_lmax,
        n_jobs=args.soap_jobs,
        expected_dim=args.expected_soap_dim,
    )

    storage_dtype = (
        np.float32
        if args.dtype == "float32"
        else np.float64
    )

    files = select_input_files(
        input_root=input_root,
        pattern=args.pattern,
        name_contains=args.name_contains,
        file_list=args.file_list,
        max_files=args.max_files,
    )

    if not files:
        raise SystemExit(
            "ERROR: no input files selected under {}".format(
                input_root
            )
        )

    mace_manifest_path = args.mace_manifest

    if mace_manifest_path is not None:
        mace_manifest_path = (
            mace_manifest_path
            .expanduser()
            .resolve()
        )

    (
        output_to_original,
        output_to_mace_row,
    ) = build_mace_path_maps(
        input_root=input_root,
        manifest_path=mace_manifest_path,
    )

    duplicate_csv = args.duplicate_csv

    if duplicate_csv is not None:
        duplicate_csv = (
            duplicate_csv
            .expanduser()
            .resolve()
        )

        if not duplicate_csv.is_file():
            raise SystemExit(
                "ERROR: duplicate CSV not found: {}".format(
                    duplicate_csv
                )
            )

    if args.unique_only and duplicate_csv is None:
        raise SystemExit(
            "ERROR: --unique_only requires --duplicate_csv."
        )

    (
        original_to_duplicate_group,
        original_to_representative,
    ) = build_duplicate_map(duplicate_csv)

    if args.unique_only:
        filtered_files: List[Path] = []

        for path in files:
            relative = str(
                path.relative_to(input_root)
            )
            original = output_to_original.get(
                relative,
                relative,
            )

            if (
                original in original_to_representative
                and not original_to_representative[original]
            ):
                continue

            filtered_files.append(path)

        files = filtered_files

    manifest_path = (
        output_root / "dataset_manifest.csv"
    )
    failures_path = (
        output_root / "dataset_failures.csv"
    )

    manifest_by_source = load_existing_rows(
        manifest_path,
        key="source_file",
    )
    failures_by_source = load_existing_rows(
        failures_path,
        key="source_file",
    )

    print("=" * 112)
    print("BUILD LOCAL EXPERT FORCE DATASET")
    print("Input root:              {}".format(input_root))
    print("Output root:             {}".format(output_root))
    print("Selected structures:     {}".format(len(files)))
    print(
        "SOAP:                   r_cut={} n_max={} l_max={}".format(
            args.soap_rcut,
            args.soap_nmax,
            args.soap_lmax,
        )
    )
    print(
        "SOAP/local dimensions:  {} / {}".format(
            soap_descriptor.soap_dim,
            soap_descriptor.feature_dim,
        )
    )
    print("Storage dtype:           {}".format(args.dtype))
    print("Experts:                 {}".format(", ".join(EXPERT_NAMES)))
    print("Unique-only mode:        {}".format(bool(args.unique_only)))
    print("=" * 112)

    start_all = time.time()
    processed_since_flush = 0
    successful_this_run = 0
    skipped_this_run = 0
    failed_this_run = 0
    atoms_this_run = 0

    for file_number, source_path in enumerate(
        files,
        start=1,
    ):
        start_structure = time.time()
        relative = source_path.relative_to(input_root)
        relative_string = str(relative)

        original_source = output_to_original.get(
            relative_string,
            relative_string,
        )

        duplicate_group = original_to_duplicate_group.get(
            original_source,
            "",
        )
        is_duplicate_representative = (
            original_to_representative.get(
                original_source,
                True,
            )
        )

        output_relative = (
            relative.parent
            / (
                relative.stem
                + "_local_expert_forces.npz"
            )
        )
        output_path = (
            shard_root / output_relative
        )

        if output_path.is_file() and not args.overwrite:
            skipped_this_run += 1
            print(
                "[{:03d}/{:03d}] SKIP existing: {}".format(
                    file_number,
                    len(files),
                    relative,
                )
            )
            continue

        try:
            atoms = read(
                source_path,
                index=0,
            )

            symbols = validate_structure_and_reference(
                atoms,
                source_path,
            )

            mace_forces = np.asarray(
                atoms.arrays["mace_forces"],
                dtype=np.float64,
            )
            mace_energy = get_reference_energy(atoms)
            mace_stress = get_reference_stress(atoms)

            local_features, species_one_hot = (
                soap_descriptor.compute(
                    atoms,
                    symbols,
                )
            )

            (
                expert_energy_per_atom,
                expert_forces,
                expert_stress,
            ) = evaluate_experts(
                atoms=atoms,
                persistent_lammps_class=PersistentLammps,
                pot_dir=pot_dir,
                lmp_lib=lmp_lib,
                lmp_python=lmp_python,
            )

            expert_energy_total = (
                expert_energy_per_atom
                * len(atoms)
            )

            expert_force_sq_error = np.sum(
                (
                    expert_forces
                    - mace_forces[None, :, :]
                ) ** 2,
                axis=2,
            )

            top1_expert_index = np.argmin(
                expert_force_sq_error,
                axis=0,
            ).astype(np.int16)

            masks = region_masks(atoms)
            all_metrics: Dict[str, Dict[str, float]] = {}
            interface_metrics: Dict[str, Dict[str, float]] = {}

            for expert_index, expert_name in enumerate(
                EXPERT_NAMES
            ):
                all_metrics[expert_name] = force_metrics(
                    expert_forces[expert_index],
                    mace_forces,
                    masks["all"],
                )

                interface_mask = masks.get(
                    "interface",
                    np.zeros(
                        len(atoms),
                        dtype=bool,
                    ),
                )
                interface_metrics[expert_name] = force_metrics(
                    expert_forces[expert_index],
                    mace_forces,
                    interface_mask,
                )

            uniform_forces = np.mean(
                expert_forces,
                axis=0,
            )
            uniform_metrics = force_metrics(
                uniform_forces,
                mace_forces,
                masks["all"],
            )

            best_expert_all = min(
                EXPERT_NAMES,
                key=lambda name: all_metrics[name][
                    "rmse_component"
                ],
            )

            finite_interface = [
                name
                for name in EXPERT_NAMES
                if np.isfinite(
                    interface_metrics[name][
                        "rmse_component"
                    ]
                )
            ]

            best_expert_interface = (
                min(
                    finite_interface,
                    key=lambda name: interface_metrics[name][
                        "rmse_component"
                    ],
                )
                if finite_interface
                else ""
            )

            phase = np.asarray(
                atoms.arrays.get(
                    "phase",
                    np.full(
                        len(atoms),
                        -1,
                        dtype=np.int16,
                    ),
                )
            ).reshape(-1)

            interface_center = np.asarray(
                atoms.arrays.get(
                    "interface_center",
                    np.full(
                        len(atoms),
                        -1,
                        dtype=np.int16,
                    ),
                )
            ).reshape(-1)

            mace_halo = np.asarray(
                atoms.arrays.get(
                    "mace_halo",
                    np.full(
                        len(atoms),
                        -1,
                        dtype=np.int16,
                    ),
                )
            ).reshape(-1)

            metadata = {
                "source_file": relative_string,
                "original_source_file": original_source,
                "structure_id": relative.stem,
                "phase_pair": str(relative.parent),
                "duplicate_group": duplicate_group,
                "is_duplicate_representative": bool(
                    is_duplicate_representative
                ),
                "reference_method": atoms.info.get(
                    "reference_method",
                    "MACE",
                ),
                "reference_status": atoms.info.get(
                    "reference_status",
                    "unknown",
                ),
                "mace_model_identifier": atoms.info.get(
                    "mace_model_identifier",
                    "unknown",
                ),
                "expert_energy_note": (
                    "Raw expert energies use different native zero references. "
                    "Do not mix them before an explicit alignment procedure."
                ),
                "gate_input_note": (
                    "X_local contains only SOAP and Si/O one-hot features. "
                    "phase, interface_center, and mace_halo are analysis labels."
                ),
            }

            arrays = {
                "X_local": local_features.astype(
                    storage_dtype,
                    copy=False,
                ),
                "soap": local_features[
                    :,
                    :soap_descriptor.soap_dim,
                ].astype(
                    storage_dtype,
                    copy=False,
                ),
                "species_one_hot": species_one_hot.astype(
                    storage_dtype,
                    copy=False,
                ),
                "positions": np.asarray(
                    atoms.positions,
                    dtype=storage_dtype,
                ),
                "cell": np.asarray(
                    atoms.cell.array,
                    dtype=storage_dtype,
                ),
                "pbc": np.asarray(
                    atoms.pbc,
                    dtype=np.bool_,
                ),
                "atomic_numbers": np.asarray(
                    atoms.numbers,
                    dtype=np.int16,
                ),
                "atom_index": np.arange(
                    len(atoms),
                    dtype=np.int32,
                ),
                "phase": phase.astype(
                    np.int16,
                    copy=False,
                ),
                "interface_center": interface_center.astype(
                    np.int16,
                    copy=False,
                ),
                "mace_halo": mace_halo.astype(
                    np.int16,
                    copy=False,
                ),
                "mace_energy_total_eV": np.asarray(
                    mace_energy,
                    dtype=np.float64,
                ),
                "mace_energy_per_atom_eV": np.asarray(
                    mace_energy / len(atoms),
                    dtype=np.float64,
                ),
                "mace_forces": mace_forces.astype(
                    storage_dtype,
                    copy=False,
                ),
                "mace_stress_voigt_eV_A3": mace_stress.astype(
                    np.float64,
                    copy=False,
                ),
                "expert_names": np.asarray(
                    EXPERT_NAMES,
                    dtype="<U32",
                ),
                "expert_energy_per_atom_raw": expert_energy_per_atom.astype(
                    np.float64,
                    copy=False,
                ),
                "expert_energy_total_raw": expert_energy_total.astype(
                    np.float64,
                    copy=False,
                ),
                "expert_quartz_reference_eV_per_atom": np.asarray(
                    [
                        E_QUARTZ_REF[name]
                        for name in EXPERT_NAMES
                    ],
                    dtype=np.float64,
                ),
                "expert_forces": expert_forces.astype(
                    storage_dtype,
                    copy=False,
                ),
                "expert_stress": expert_stress.astype(
                    np.float64,
                    copy=False,
                ),
                "expert_force_sq_error": expert_force_sq_error.astype(
                    storage_dtype,
                    copy=False,
                ),
                "top1_expert_index": top1_expert_index,
                "metadata_json": np.asarray(
                    json.dumps(
                        metadata,
                        sort_keys=True,
                    )
                ),
            }

            atomic_save_npz(
                output_path,
                compressed=not args.uncompressed,
                **arrays,
            )

            validate_saved_shard(
                output_path,
                n_atoms=len(atoms),
                n_features=soap_descriptor.feature_dim,
            )

            mace_force_norm = np.linalg.norm(
                mace_forces,
                axis=1,
            )

            row: Dict = {
                "status": "PASS",
                "source_file": relative_string,
                "original_source_file": original_source,
                "output_file": str(output_relative),
                "structure_id": relative.stem,
                "phase_pair": str(relative.parent),
                "duplicate_group": duplicate_group,
                "is_duplicate_representative": bool(
                    is_duplicate_representative
                ),
                "n_atoms": int(len(atoms)),
                "n_Si": int(np.sum(symbols == "Si")),
                "n_O": int(np.sum(symbols == "O")),
                "cell_volume_A3": float(
                    atoms.get_volume()
                ),
                "soap_dim": int(
                    soap_descriptor.soap_dim
                ),
                "local_feature_dim": int(
                    soap_descriptor.feature_dim
                ),
                "mace_energy_total_eV": float(
                    mace_energy
                ),
                "mace_energy_per_atom_eV": float(
                    mace_energy / len(atoms)
                ),
                "mace_mean_force_eV_A": float(
                    np.mean(mace_force_norm)
                ),
                "mace_max_force_eV_A": float(
                    np.max(mace_force_norm)
                ),
                "mace_net_force_norm_eV_A": float(
                    np.linalg.norm(
                        np.sum(
                            mace_forces,
                            axis=0,
                        )
                    )
                ),
                "best_expert_all": best_expert_all,
                "best_expert_all_rmse_component_eV_A": float(
                    all_metrics[best_expert_all][
                        "rmse_component"
                    ]
                ),
                "best_expert_interface": best_expert_interface,
                "best_expert_interface_rmse_component_eV_A": (
                    float(
                        interface_metrics[
                            best_expert_interface
                        ]["rmse_component"]
                    )
                    if best_expert_interface
                    else float("nan")
                ),
                "uniform7_rmse_component_eV_A": float(
                    uniform_metrics["rmse_component"]
                ),
                "wall_seconds": float(
                    time.time() - start_structure
                ),
            }

            mace_row = output_to_mace_row.get(
                relative_string,
                {},
            )

            if mace_row.get("warning_level"):
                row["mace_warning_level"] = (
                    mace_row["warning_level"]
                )

            for expert_name in EXPERT_NAMES:
                key = safe_name(expert_name)
                metrics_all = all_metrics[expert_name]
                metrics_interface = (
                    interface_metrics[expert_name]
                )
                expert_index = EXPERT_NAMES.index(
                    expert_name
                )

                row[
                    "{}_energy_per_atom_raw_eV".format(key)
                ] = float(
                    expert_energy_per_atom[
                        expert_index
                    ]
                )
                row[
                    "{}_rmse_component_eV_A".format(key)
                ] = float(
                    metrics_all["rmse_component"]
                )
                row[
                    "{}_mae_vector_eV_A".format(key)
                ] = float(
                    metrics_all["mae_vector"]
                )
                row[
                    "{}_mean_cosine".format(key)
                ] = float(
                    metrics_all["mean_cosine"]
                )
                row[
                    "{}_interface_rmse_component_eV_A".format(
                        key
                    )
                ] = float(
                    metrics_interface["rmse_component"]
                )
                row[
                    "{}_net_force_norm_eV_A".format(key)
                ] = float(
                    np.linalg.norm(
                        np.sum(
                            expert_forces[
                                expert_index
                            ],
                            axis=0,
                        )
                    )
                )

            manifest_by_source[
                relative_string
            ] = row
            failures_by_source.pop(
                relative_string,
                None,
            )

            successful_this_run += 1
            atoms_this_run += len(atoms)

            print(
                "[{:03d}/{:03d}] {} | N={:5d} | "
                "best={} ({:.5f}) | uniform={:.5f} | "
                "{:.2f} s".format(
                    file_number,
                    len(files),
                    relative,
                    len(atoms),
                    best_expert_all,
                    all_metrics[best_expert_all][
                        "rmse_component"
                    ],
                    uniform_metrics[
                        "rmse_component"
                    ],
                    time.time() - start_structure,
                )
            )

        except Exception as exc:
            failed_this_run += 1

            failure = {
                "source_file": relative_string,
                "original_source_file": original_source,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds": float(
                    time.time() - start_structure
                ),
            }

            failures_by_source[
                relative_string
            ] = failure

            print(
                "[{:03d}/{:03d}] FAILED {} | {}: {}".format(
                    file_number,
                    len(files),
                    relative,
                    type(exc).__name__,
                    exc,
                ),
                file=sys.stderr,
            )

            if args.fail_fast:
                raise

        processed_since_flush += 1

        if (
            processed_since_flush
            >= max(1, args.flush_every)
        ):
            write_csv(
                manifest_path,
                list(manifest_by_source.values()),
            )
            write_csv(
                failures_path,
                list(failures_by_source.values()),
            )
            processed_since_flush = 0

    write_csv(
        manifest_path,
        list(manifest_by_source.values()),
    )
    write_csv(
        failures_path,
        list(failures_by_source.values()),
    )

    dataset_info = {
        "dataset_name": (
            "Dynamic Local MoE atom-resolved expert-force dataset"
        ),
        "created_unix_time": time.time(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_pattern": args.pattern,
        "n_selected_files": len(files),
        "n_successful_this_run": successful_this_run,
        "n_skipped_this_run": skipped_this_run,
        "n_failed_this_run": failed_this_run,
        "n_atoms_successful_this_run": atoms_this_run,
        "experts": EXPERT_NAMES,
        "expert_order_is_semantic": True,
        "soap": {
            "species": ["Si", "O"],
            "r_cut_A": args.soap_rcut,
            "n_max": args.soap_nmax,
            "l_max": args.soap_lmax,
            "average": "off",
            "periodic": True,
            "soap_dim": soap_descriptor.soap_dim,
            "species_one_hot_dim": 2,
            "local_feature_dim": (
                soap_descriptor.feature_dim
            ),
        },
        "gate_input": (
            "X_local = per-atom SOAP concatenated with "
            "[is_Si, is_O]."
        ),
        "analysis_labels_not_gate_inputs": ANALYSIS_LABELS,
        "reference": {
            "forces": "MACE atom-resolved forces in eV/Angstrom",
            "energy": "MACE total energy in eV",
            "stress": (
                "MACE ASE Voigt stress [xx, yy, zz, yz, xz, xy] "
                "in eV/Angstrom^3"
            ),
        },
        "expert_outputs": {
            "energy": (
                "Raw native expert energy per atom and total energy. "
                "Different experts have different energy zeros."
            ),
            "forces": (
                "Seven complete-structure atom-resolved force arrays "
                "in eV/Angstrom."
            ),
            "stress": (
                "Seven ASE Voigt stress arrays in eV/Angstrom^3."
            ),
        },
        "storage_dtype": args.dtype,
        "compression": (
            "none"
            if args.uncompressed
            else "np.savez_compressed"
        ),
        "core_calculator": {
            "path": str(calculator_path),
            "sha256": sha256sum(
                calculator_path
            ),
        },
        "potential_files": {
            name: {
                "path": str(pot_dir / name),
                "sha256": sha256sum(
                    pot_dir / name
                ),
            }
            for name in [
                "Tersoff.pot",
                "Vashishta.pot",
            ]
        },
        "lammps_library": {
            "path": str(lmp_lib),
            "size_bytes": lmp_lib.stat().st_size,
            "mtime": lmp_lib.stat().st_mtime,
        },
        "mace_manifest": (
            str(mace_manifest_path)
            if mace_manifest_path is not None
            else (
                str(input_root / "mace_manifest.csv")
                if (
                    input_root
                    / "mace_manifest.csv"
                ).is_file()
                else None
            )
        ),
        "duplicate_csv": (
            str(duplicate_csv)
            if duplicate_csv is not None
            else None
        ),
        "unique_only": bool(args.unique_only),
        "total_wall_seconds": float(
            time.time() - start_all
        ),
        "training_note": (
            "The first local gate should be trained against MACE forces "
            "using structure-balanced sampling and interface-pair-level "
            "train/validation/test splits. The global MoE is not a target."
        ),
    }

    info_path = (
        output_root / "dataset_info.json"
    )
    temporary_info = info_path.with_suffix(
        ".json.tmp"
    )
    temporary_info.write_text(
        json.dumps(
            dataset_info,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary_info.replace(info_path)

    print("\n" + "=" * 112)
    print("LOCAL EXPERT FORCE DATASET BUILD COMPLETE")
    print(
        "Successful this run:  {}".format(
            successful_this_run
        )
    )
    print(
        "Skipped this run:     {}".format(
            skipped_this_run
        )
    )
    print(
        "Failed this run:      {}".format(
            failed_this_run
        )
    )
    print(
        "Atoms written:        {:,}".format(
            atoms_this_run
        )
    )
    print(
        "Manifest:             {}".format(
            manifest_path
        )
    )
    print(
        "Failures:             {}".format(
            failures_path
        )
    )
    print(
        "Dataset information:  {}".format(
            info_path
        )
    )
    print(
        "Total wall time:      {:.1f} s".format(
            time.time() - start_all
        )
    )
    print("=" * 112)

    if failed_this_run:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nInterrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
