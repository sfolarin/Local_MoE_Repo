#!/usr/bin/env python3
"""Export raw parity points from the completed common-energy evaluator.

The swapped Oishi C1b run selected a zero residual at validation, so the
common checkpoint is the final model.  This wrapper reuses the project's own
evaluation path and additionally writes per-test-structure force arrays.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-script", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--energy-weight", type=float, default=None)
    args = parser.parse_args()

    # The swapped run was trained against this fixed split.  Set both names
    # used by the project patches so evaluation reproduces that contract even
    # when the parent shell was freshly opened.
    fixed_split = args.out.parent / "dataset_split.csv"
    if fixed_split.is_file():
        os.environ["FIXED_SPLIT"] = str(fixed_split)
        os.environ["FIXED_SPLIT_MANIFEST"] = str(fixed_split)

    common_script = args.common_script.resolve()
    sys.path.insert(0, str(common_script.parent))
    spec = importlib.util.spec_from_file_location("common_swap_module", common_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import common script: {common_script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    original_evaluate = module.evaluate_records
    raw_root = args.out / "raw_predictions"

    def evaluate_with_raw(ctx, model, records, eval_args, stop_requested=None):
        rows = original_evaluate(ctx, model, records, eval_args, stop_requested)
        if not records:
            return rows
        split = str(records[0].split)
        raw_dir = raw_root / split
        raw_dir.mkdir(parents=True, exist_ok=True)
        with ctx.torch.enable_grad():
            for record in records:
                energy, forces = module.prediction(
                    ctx, model, record, create_graph=False
                )
                safe_id = str(record.source_file).replace("/", "__")
                np.savez_compressed(
                    raw_dir / f"{safe_id}.npz",
                    record_id=np.asarray(str(record.source_file)),
                    n_atoms=np.asarray(record.n_atoms, dtype=np.int64),
                    E_reference_eV=np.asarray(record.target_energy_total_eV, dtype=np.float64),
                    E_predicted_eV=np.asarray(float(energy.detach().cpu()), dtype=np.float64),
                    reference_forces=np.asarray(record.target_forces_np, dtype=np.float64),
                    predicted_forces=np.asarray(forces.detach().cpu(), dtype=np.float64),
                )
                del energy, forces
        print(f"Raw test predictions: {raw_dir}", flush=True)
        return rows

    module.evaluate_records = evaluate_with_raw
    sys.argv = [
        str(common_script),
        "evaluate",
        "--features", str(args.features),
        "--out", str(args.out),
        "--splits", *args.splits,
        "--threads", str(args.threads),
        *(["--energy-weight", str(args.energy_weight)] if args.energy_weight is not None else []),
    ]
    module.main()


if __name__ == "__main__":
    main()
