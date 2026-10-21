#!/usr/bin/env python3
"""md_status.py - progress of the holdout-40 MD runs: one row per manifest structure.
   python md_status.py [--out-root DIR] [--manifest CSV]"""
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
import json
from pathlib import Path

import pandas as pd

HET = (_os.environ.get("MOE_HET", "."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path(HET) / "Local_MoE_MATPES_R2SCAN_Expanded_v1/md_holdout40_v1")
    p.add_argument("--manifest", type=Path, default=None)
    a = p.parse_args()
    man = pd.read_csv(a.manifest or a.out_root / "md_manifest.csv")
    rows = []
    for _, r in man.iterrows():
        d = a.out_root / f"case_{r['record_id']}"
        state, drift, sps, note = "pending", "", "", ""
        if (d / "final_protocol_summary.json").is_file() and (d / "nve_final_structure.extxyz").is_file():
            s = json.loads((d / "final_protocol_summary.json").read_text())
            state = "done" if s["energy_conserving"] else "DONE-NOT-CONSERVING"
            drift = f"{s['nve_max_abs_energy_drift_meV_per_atom']:.5f}"
            sps = f"{s['nve_steps_per_second']:.1f}"
        elif d.is_dir():
            if (d / "nve_10ps_frames.extxyz").is_file():
                state = "NVE running"
            elif (d / "npt_final_structure.extxyz").is_file():
                state = "NVE starting"
            elif (d / "npt_5ps_frames.extxyz").is_file():
                state = "NPT running"
            elif (d / "relaxed_structure.extxyz").is_file():
                state = "NPT starting"
            else:
                state = "relaxing"
        rows.append(dict(job=r["job"], slot=r["slot"], order=r.get("order", ""), record_id=r["record_id"], n_atoms=r.get("n_atoms", ""),
                         state=state, max_abs_drift_meV=drift, nve_steps_per_s=sps))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\n" + df["state"].value_counts().to_string())


if __name__ == "__main__":
    main()
