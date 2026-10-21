#!/usr/bin/env python3
"""
build_gulp_vs_oishi_table.py

Builds the side-by-side comparison: Oishi's own published interfacial
energy vs. our own GULP/SLC reproduction.

Component filenames are per-pair-scoped (case_id + framework), matching
generate_gulp_inputs.py's fix for the cross-pair component-sharing bug
confirmed tonight (a shared "{name}_..." filename let one pair's
component silently reuse another pair's wrong-registry file).

Usage:
    python build_gulp_vs_oishi_table.py --gulp-dir gulp-6.4 \
        --oishi-table Supplementary_Table3_lattice_match.xlsx \
        --pairs-csv pairs_20_total.csv
"""
import argparse
import math
import re
from pathlib import Path

import pandas as pd

def build_pairs_from_csv(pairs_csv_path):
    df = pd.read_csv(pairs_csv_path)
    pairs = []
    for _, row in df.iterrows():
        case_id, pa, pb = row["case_id"], row["parent_A"], row["parent_B"]
        label = f"{pa}/{pb}"
        ab_rel = f"{case_id}_slc_run/{case_id}_slc.got"
        a_rel = f"{case_id}_{pa}_component_fixedcell_slc_run/{case_id}_{pa}_component_fixedcell_slc.got"
        b_rel = f"{case_id}_{pb}_component_fixedcell_slc_run/{case_id}_{pb}_component_fixedcell_slc.got"
        pairs.append((label, pa, pb, ab_rel, a_rel, b_rel))
    return pairs

EV_A2_TO_J_M2 = 16.02176634


def parse_got(path):
    if not path.exists():
        return False, None, None
    text = path.read_text()
    if "Job Finished" not in text:
        return False, None, None

    m = re.search(r"Final energy\s*=\s*(-?\d+\.\d+)\s*eV", text)
    energy = float(m.group(1)) if m else None

    cell = None
    m2 = re.search(
        r"Final cell parameters and derivatives\s*:\s*\n-+\n"
        r"\s*a\s+(-?\d+\.\d+)\s+Angstrom.*?\n"
        r"\s*b\s+(-?\d+\.\d+)\s+Angstrom.*?\n"
        r"\s*c\s+(-?\d+\.\d+)\s+Angstrom.*?\n"
        r"\s*alpha\s+(-?\d+\.\d+)\s+Degrees.*?\n"
        r"\s*beta\s+(-?\d+\.\d+)\s+Degrees.*?\n"
        r"\s*gamma\s+(-?\d+\.\d+)\s+Degrees",
        text, re.DOTALL,
    )
    if m2:
        a, b, c, alpha, beta, gamma = (float(x) for x in m2.groups())
        cell = {"a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma}

    return True, energy, cell


def cell_area(cell):
    return cell["a"] * cell["b"] * math.sin(math.radians(cell["gamma"]))


def pct_drift(a, b):
    return abs(a - b) / abs(b) * 100.0 if b else float("nan")


def lookup_oishi_gamma(oishi_df, name_a, name_b):
    mask = (
        oishi_df["Pair"].astype(str).str.fullmatch(f"{name_a}/{name_b}", case=False)
        | oishi_df["Pair"].astype(str).str.fullmatch(f"{name_b}/{name_a}", case=False)
    )
    hits = oishi_df[mask]
    if len(hits) == 0:
        return None
    return float(hits.iloc[0]["gamma_J_m2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gulp-dir", required=True)
    ap.add_argument("--oishi-table", required=True)
    ap.add_argument("--pairs-csv", required=True)
    ap.add_argument("--max-drift-pct", type=float, default=1.0)
    ap.add_argument("--out-csv", default="gulp_vs_oishi_comparison.csv")
    args = ap.parse_args()

    gulp_dir = Path(args.gulp_dir)

    oishi_df = pd.ExcelFile(args.oishi_table).parse(
        "min_energy_with_lattice_match_s", skiprows=1
    )
    oishi_df.columns = [
        "Pair", "Miller_indices", "gamma_J_m2", "dE_kJ_mol_Si", "area_A2",
        "length_mismatch_pct", "angle_mismatch_pct", "area_mismatch_pct",
    ]

    pairs_list = build_pairs_from_csv(args.pairs_csv)

    rows = []
    print(f"{'Pair':<12} {'Status':<40} {'Oishi (J/m2)':>14} {'GULP/SLC (J/m2)':>16}")
    print("-" * 86)

    for label, name_a, name_b, ab_rel, a_rel, b_rel in pairs_list:
        ab_done, e_ab, cell_ab = parse_got(gulp_dir / ab_rel)
        a_done, e_a, cell_a = parse_got(gulp_dir / a_rel)
        b_done, e_b, cell_b = parse_got(gulp_dir / b_rel)

        oishi_gamma = lookup_oishi_gamma(oishi_df, name_a, name_b)
        oishi_str = f"{oishi_gamma:.4f}" if oishi_gamma is not None else "not in table"

        if not (ab_done and a_done and b_done):
            missing = []
            if not ab_done:
                missing.append("AB")
            if not a_done:
                missing.append(name_a)
            if not b_done:
                missing.append(name_b)
            status = f"pending ({', '.join(missing)} not finished)"
            print(f"{label:<12} {status:<40} {oishi_str:>14} {'-':>16}")
            rows.append({"pair": label, "status": status, "oishi_gamma_J_m2": oishi_gamma,
                         "gulp_gamma_J_m2": None})
            continue

        if e_ab is None or e_a is None or e_b is None:
            status = "ERROR: could not parse energy (file may have been mid-write)"
            print(f"{label:<12} {status:<40} {oishi_str:>14} {'-':>16}")
            rows.append({"pair": label, "status": status, "oishi_gamma_J_m2": oishi_gamma,
                         "gulp_gamma_J_m2": None})
            continue

        if cell_ab is None or cell_a is None or cell_b is None:
            status = "ERROR: could not parse cell parameters"
            print(f"{label:<12} {status:<40} {oishi_str:>14} {'-':>16}")
            rows.append({"pair": label, "status": status, "oishi_gamma_J_m2": oishi_gamma,
                         "gulp_gamma_J_m2": None})
            continue

        drift_a_a = pct_drift(cell_a["a"], cell_ab["a"])
        drift_a_b = pct_drift(cell_a["b"], cell_ab["b"])
        drift_a_g = pct_drift(cell_a["gamma"], cell_ab["gamma"])
        drift_b_a = pct_drift(cell_b["a"], cell_ab["a"])
        drift_b_b = pct_drift(cell_b["b"], cell_ab["b"])
        drift_b_g = pct_drift(cell_b["gamma"], cell_ab["gamma"])
        max_drift = max(drift_a_a, drift_a_b, drift_a_g, drift_b_a, drift_b_b, drift_b_g)

        if max_drift > args.max_drift_pct:
            status = f"FAILED coherence check (max drift {max_drift:.2f}% > {args.max_drift_pct}%)"
            print(f"{label:<12} {status:<40} {oishi_str:>14} {'-':>16}")
            rows.append({"pair": label, "status": status, "oishi_gamma_J_m2": oishi_gamma,
                         "gulp_gamma_J_m2": None, "max_drift_pct": max_drift})
            continue

        area_ab = cell_area(cell_ab)
        delta_e = e_ab - e_a - e_b
        gamma_gulp = delta_e / (2.0 * area_ab) * EV_A2_TO_J_M2

        status = f"OK (max component drift {max_drift:.2f}%)"
        print(f"{label:<12} {status:<40} {oishi_str:>14} {gamma_gulp:>16.4f}")
        rows.append({
            "pair": label, "status": status, "oishi_gamma_J_m2": oishi_gamma,
            "gulp_gamma_J_m2": gamma_gulp, "max_drift_pct": max_drift,
            "E_AB_eV": e_ab, f"E_{name_a}_eV": e_a, f"E_{name_b}_eV": e_b,
            "area_ab_A2": area_ab,
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
