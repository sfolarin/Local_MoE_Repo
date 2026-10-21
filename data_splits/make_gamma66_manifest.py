#!/usr/bin/env python3
"""
make_gamma66_manifest.py     (READ-ONLY on the geometry files; writes one CSV)

One row per Oishi interface with positive MATPes gamma (the rows kept in oishi_gamma_candidates.csv),
with the explicit paths of the relaxed AB cell and its two component references, and the status of the
family under the NEW split (holdout_split_*.csv): test = held out, train = in training.

Paths
  Primary50 rows : AB  <abroot>/<case>/relaxed.extxyz
                   A,B <components>/<pair_directory>/<component>/relaxed.extxyz   (from the Primary50 table)
  other rows     : ab_path / a_path / b_path from own252_full42_explicit_paths.csv
Every path is checked to exist, and MATPes gamma is checked against the source table; any problem
is listed and the script exits with an error instead of writing a partial manifest.

Run from External_Benchmark_Oishi2025/:
    python make_gamma66_manifest.py --split-table holdout_split_n40.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--ext", default=".")
ap.add_argument("--split-table", default="holdout_split_n40.csv")
ap.add_argument("--candidates", default="oishi_gamma_candidates.csv")
ap.add_argument("--primary50", default="Gamma54_MATPES_Interface_Energies/gamma54_matpes_interface_energies_PRIMARY50.csv")
ap.add_argument("--explicit", default="own252_full42_explicit_paths.csv")
ap.add_argument("--abroot", default="MATPES_Relaxed_Oishi_Gamma54_AB")
ap.add_argument("--components", default="MATPES_Oishi_Component_References")
ap.add_argument("--out", default="gamma66_manifest.csv")
args = ap.parse_args()
EXT = Path(args.ext)
pd.set_option("display.width", 200)

cand = pd.read_csv(EXT / args.candidates, keep_default_na=False)
keep = cand[cand.drop_reason.astype(str) == ""].copy()
p50 = pd.read_csv(EXT / args.primary50).set_index("case")
exp = pd.read_csv(EXT / args.explicit)
tab = pd.read_csv(EXT / args.split_table)

problems, rows = [], []
for r in keep.itertuples():
    case = str(r.case)
    d = {"row_id": case, "case": case, "origin": r.origin, "gamma_MATPES_J_m2": float(r.gamma_MATPES_J_m2),
         "old_pool_split": r.pool_split}
    if r.origin == "Primary50":
        if case not in p50.index:
            problems.append(f"{case}: not in the Primary50 table"); continue
        q = p50.loc[case]
        d.update(ab_path=f"{args.abroot}/{case}/relaxed.extxyz",
                 a_path=f"{args.components}/{q.pair_directory}/{q.component_A}/relaxed.extxyz",
                 b_path=f"{args.components}/{q.pair_directory}/{q.component_B}/relaxed.extxyz", N_AB=int(q.N_AB))
        if abs(float(q.gamma_MATPES_J_m2) - d["gamma_MATPES_J_m2"]) > 1e-9:
            problems.append(f"{case}: gamma differs between candidates and Primary50")
    else:
        m = exp[exp.case == case]
        if len(m) != 1:
            problems.append(f"{case}: {len(m)} rows in the explicit-paths table (need exactly 1)"); continue
        q = m.iloc[0]
        d.update(ab_path=q.ab_path, a_path=q.a_path, b_path=q.b_path, N_AB=int(q.N_AB))
        if abs(float(q.gamma_MATPES_J_m2) - d["gamma_MATPES_J_m2"]) > 1e-9:
            problems.append(f"{case}: gamma differs between candidates and the explicit-paths table")
    t = tab[(tab.family == case) & tab.gamma_family.astype(str).str.lower().eq("true")]
    if len(t) == 0:
        problems.append(f"{case}: no record in the split table"); continue
    if t.new_split.nunique() != 1:
        problems.append(f"{case}: records of this family are in more than one split: {sorted(t.new_split.unique())}"); continue
    d["new_split"] = t.new_split.iloc[0]
    d["pair"] = t.pair.iloc[0]
    d["n_records"] = len(t)
    for k in ("ab_path", "a_path", "b_path"):
        if not (EXT / d[k]).is_file():
            problems.append(f"{case}: missing file {d[k]}")
    rows.append(d)

if problems:
    print(f"{len(problems)} PROBLEM(S), no manifest written:")
    for p in problems[:40]:
        print("  -", p)
    raise SystemExit(1)

m = pd.DataFrame(rows)
assert m.case.is_unique and len(m) == len(keep)
m.to_csv(args.out, index=False)
print(f"{len(m)} rows -> {args.out}")
print("\nnew-split status by origin:")
print(pd.crosstab(m.origin, m.new_split).to_string())
print("\nMATPes gamma (J/m2) by new split:")
print(m.groupby("new_split").gamma_MATPES_J_m2.describe()[["count", "min", "50%", "max"]].to_string(float_format=lambda v: f"{v:.4f}"))
print("\nN_AB by new split:")
print(m.groupby("new_split").N_AB.describe()[["min", "50%", "max"]].to_string())
print("\nrows whose status changed from the old pool split to the new one:")
print(pd.crosstab(m.old_pool_split, m.new_split).to_string())
