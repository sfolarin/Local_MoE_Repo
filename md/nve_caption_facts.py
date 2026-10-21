#!/usr/bin/env python3
"""
nve_caption_facts.py   (CPU only; reads files, runs no MD)

Reads what the MD runner (run_local_moe_md.py) recorded for every finished case and prints the facts a caption or a
methods paragraph needs, and checks that all cases used one protocol:

  * split / dtype / seed / selected_model / protocol (time step, thermostat, barostat, lengths, sampling interval)
  * relaxation, NPT and velocity-rescale numbers (so the NVE start state can be described)
  * NVE temperature, sampling interval and number of samples read from the logs themselves
  * versions and GPU name (from provenance and from the slurm logs)

  python nve_caption_facts.py [--md-root DIR]
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
import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HET = (_os.environ.get("MOE_HET", "."))


def uniq(values, label, width=26):
    c = Counter(json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v) for v in values)
    flag = "" if len(c) == 1 else "   <-- DIFFERS BETWEEN CASES"
    print(f"  {label:<{width}} " + "; ".join(f"{k} (x{n})" for k, n in c.most_common()) + flag)


def span(values, label, unit="", fmt=".3f", width=26):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        print(f"  {label:<{width}} not recorded")
        return
    print(f"  {label:<{width}} min {v.min():{fmt}}  median {np.median(v):{fmt}}  max {v.max():{fmt}} {unit}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md-root", type=Path, default=Path(HET) / "Local_MoE_MATPES_R2SCAN_Expanded_v1/md_holdout40_v1")
    a = p.parse_args()

    files = sorted(glob.glob(str(a.md_root / "case_*" / "final_protocol_summary.json")))
    if not files:
        raise SystemExit(f"no final_protocol_summary.json under {a.md_root}/case_*/")
    S = [json.loads(Path(f).read_text()) for f in files]
    print("=" * 100)
    print(f"{len(S)} finished cases under {a.md_root}")
    print("=" * 100)

    print("\nWHAT WAS RUN (each line must show one value):")
    uniq([s.get("split") for s in S], "split")
    uniq([s.get("selected_model") for s in S], "model")
    uniq([s.get("dtype") for s in S], "calculator dtype")
    seeds = [s.get("seed") for s in S]
    print(f"  {'seed':<26} {len(set(seeds))} distinct values over {len(seeds)} cases (one per structure is expected)")
    keys = sorted({k for s in S for k in s.get("protocol", {})})
    for k in keys:
        uniq([s.get("protocol", {}).get(k) for s in S], k)

    print("\nSTARTING THE NVE (per-case ranges):")
    span([s.get("relaxation", {}).get("Fmax_final_eV_A") for s in S], "Fmax after relaxation", "eV/A", ".4f")
    span([s.get("npt_mean_temperature_K") for s in S], "NPT mean temperature", "K", ".1f")
    span([s.get("npt_mean_pressure_GPa") for s in S], "NPT mean pressure", "GPa", ".4f")
    span([s.get("npt_volume_change_percent") for s in S], "NPT volume change", "%", ".2f")
    span([s.get("temperature_before_velocity_rescale_K") for s in S], "T before rescale", "K", ".2f")
    span([s.get("temperature_after_velocity_rescale_K") for s in S], "T after rescale", "K", ".2f")
    span([s.get("nve_mean_temperature_K") for s in S], "NVE mean temperature", "K", ".1f")
    span([s.get("nve_min_temperature_K") for s in S], "NVE minimum temperature", "K", ".1f")
    span([s.get("nve_max_temperature_K") for s in S], "NVE maximum temperature", "K", ".1f")
    span([s.get("nve_steps_per_second") for s in S], "NVE speed", "steps/s", ".1f")

    # cases that did not relax to the target, or whose cell changed a lot during the NPT
    FMAX_TOL, DV_TOL = 0.011, 5.0
    flagged = [s for s in S if s.get("relaxation", {}).get("Fmax_final_eV_A", 0.0) > FMAX_TOL
               or abs(s.get("npt_volume_change_percent", 0.0)) > DV_TOL]
    n_unrelaxed = sum(1 for s in S if s.get("relaxation", {}).get("Fmax_final_eV_A", 0.0) > FMAX_TOL)
    n_bigdv = sum(1 for s in S if abs(s.get("npt_volume_change_percent", 0.0)) > DV_TOL)
    print(f"\nCASES TO LOOK AT: {n_unrelaxed} not relaxed to {FMAX_TOL} eV/A, {n_bigdv} with |NPT volume change| > "
          f"{DV_TOL:g}%  ({len(flagged)} distinct of {len(S)})")
    for s in sorted(flagged, key=lambda x: -x.get("relaxation", {}).get("Fmax_final_eV_A", 0.0)):
        r = s.get("relaxation", {})
        print(f"  {s.get('record_id', '?'):<14} {s.get('phase_pair', ''):<22} N={s.get('n_atoms', '?')!s:<5} "
              f"Fmax {r.get('Fmax_final_eV_A', float('nan')):8.4f} eV/A "
              f"(FIRE {r.get('fire_steps', '?')} + {r.get('optimizers', ['', '?'])[1]} {r.get('polish_steps', '?')} steps) | "
              f"dV {s.get('npt_volume_change_percent', float('nan')):+6.2f}% | "
              f"NVE max|dE| {s.get('nve_max_abs_energy_drift_meV_per_atom', float('nan')):.5f} meV/atom")

    print("\nFROM THE NVE LOGS THEMSELVES:")
    dts, nsamp, lens = [], [], []
    for f in files:
        lg = Path(f).parent / "nve_10ps_log.csv"
        if not lg.is_file():
            continue
        d = pd.read_csv(lg).sort_values("step").drop_duplicates("step", keep="last")
        t = d["time_ps"].to_numpy(float)
        if len(t) > 2:
            dts.append(float(np.median(np.diff(t)) * 1000.0))    # fs between samples
            nsamp.append(len(t))
            lens.append(float(t[-1] - t[0]))
    span(dts, "sampling interval", "fs", ".2f")
    span(nsamp, "samples per trajectory", "", ".0f")
    span(lens, "NVE length", "ps", ".3f")

    print("\nSOFTWARE AND HARDWARE:")
    prov = [s.get("provenance", {}) for s in S]
    uniq([x.get("ase") for x in prov], "ASE")
    uniq([x.get("torch") for x in prov], "torch")
    uniq([x.get("host", "")[:5] for x in prov], "host (first 5 chars)")
    gpus = Counter()
    for lg in glob.glob(str(a.md_root / "logs" / "*.out")) + glob.glob(str(a.md_root / "logs" / "*.log")):
        for line in open(lg, errors="ignore"):
            m = re.match(r"\s*GPU:\s*(.+?)\s+CUDA_VISIBLE_DEVICES", line)
            if m:
                gpus[m.group(1)] += 1
    print("  GPU name (slurm logs)     " + ("; ".join(f"{k} (x{n})" for k, n in gpus.items()) if gpus else
                                         "not found in logs/*.out"))

    print("\nCAPTION / METHODS FACTS TO TAKE FROM THE LINES ABOVE:")
    print("  - split must be 'test' for every case; dtype tells which precision the conservation claim is for")
    print("  - time step, thermostat/barostat, lengths and sampling interval: the 'protocol' block")
    print("  - start of the NVE: the 'STARTING THE NVE' block")


if __name__ == "__main__":
    main()
