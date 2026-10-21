#!/usr/bin/env python3
"""
make_md_manifest.py

Choose which held-out structures get the NVE-stability MD and spread them over the premium jobs.

Source of truth (both written by the split builder, in the holdout-40 dataset folder):
  test.csv               one row per test structure: record_id, source, family, split_group, n_atoms,
                         has_region_metadata, existing_split ...
  group_assignments.csv  one row per group: group_id, split, reason ...
The 40 NEW held-out families are the groups with split == test and reason == fixed_seed_new_group_assignment.

Selection
---------
  1. pool        the new held-out families (expected 40; the script says so if it finds a different number)
  2. draw        --n-families of them (default 30). The largest family (and any --include) is always in;
                 the rest are drawn stratified by size: the families are sorted by atom count, cut into
                 three equal bins and drawn from each bin in proportion, with a seeded RNG (--seed), so
                 that small, medium and large cells are all represented and the draw is reproducible.
                 The dropped families are listed in md_manifest_dropped.csv.
  3. structure   one structure per Oishi family: the DFT-relaxed geometry (Oishi_relaxed52) when the family
                 has one, otherwise the Oishi_all179 geometry. A selected Own family contributes both of its
                 structures (own structures come in pairs).
  4. --own all   (default) additionally runs every Own_augmented test structure (22), because only those carry
                 the phase / interface-centre labels that the region-resolved routing figure needs;
     --own none  runs only the drawn families (the routing figure is then all-atom only).

Load balancing: structures are assigned to --jobs x --slots-per-job GPU slots (default 5 x 4) by longest-first
greedy packing on a cost proportional to (n_atoms + 300) and run smallest-first inside a slot.

Output (in --out-dir): md_manifest.csv (columns record_id, label, source, split_group, n_atoms,
has_region_metadata, why, job, slot, order), md_manifest_dropped.csv, and a printed summary.
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
import random
from pathlib import Path

import numpy as np
import pandas as pd

HET_DEFAULT = (_os.environ.get("MOE_HET", "."))
NEW_REASON = "fixed_seed_new_group_assignment"
SHORT = {"Oishi_all179": "Oishi", "Oishi_relaxed52": "Oishi", "Own_augmented": "Own", "Siddharth": "Siddharth"}
PREFERENCE = {"Oishi_relaxed52": 0, "Oishi_all179": 1}       # lower is preferred; anything else last


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=None, help="default: <proj>/data/expanded_holdout40_v1")
    p.add_argument("--out-dir", type=Path, default=None, help="default: <proj>/md_holdout40_v1")
    p.add_argument("--n-families", type=int, default=30)
    p.add_argument("--own", choices=["all", "none"], default="all")
    p.add_argument("--include", default="", help="comma-separated families that must be in the draw")
    p.add_argument("--no-include-largest", action="store_true",
                   help="by default the largest new family is always in the draw (it is the large-N case)")
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--slots-per-job", type=int, default=4)
    return p.parse_args()


def stratified_draw(families, sizes, k, rng):
    """families sorted by size; return the k drawn, proportionally from three contiguous size bins."""
    order = sorted(families, key=lambda f: (sizes[f], f))
    bins = [list(b) for b in np.array_split(np.array(order, dtype=object), 3)]
    quota = [k * len(b) / len(order) for b in bins]
    take = [int(np.floor(q)) for q in quota]
    for i in sorted(range(3), key=lambda i: -(quota[i] - take[i])):       # distribute the remainder
        if sum(take) < k:
            take[i] += 1
    out = []
    for b, t in zip(bins, take):
        out += rng.sample(b, min(t, len(b)))
    return sorted(out)


def main():
    a = parse_args()
    proj = Path(a.dataset).expanduser().resolve().parents[1] if a.dataset else \
        Path(HET_DEFAULT) / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    ds = Path(a.dataset).expanduser().resolve() if a.dataset else proj / "data/expanded_holdout40_v1"
    out = Path(a.out_dir).expanduser().resolve() if a.out_dir else proj / "md_holdout40_v1"
    out.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(ds / "test.csv")
    ga = pd.read_csv(ds / "group_assignments.csv")
    need = {"record_id", "source", "split_group", "n_atoms", "has_region_metadata"}
    if not need <= set(test.columns):
        raise SystemExit(f"test.csv lacks columns {sorted(need - set(test.columns))}")
    test["has_region_metadata"] = test["has_region_metadata"].astype(str).str.lower().isin(["true", "1"])

    new_groups = sorted(ga.loc[(ga["split"] == "test") & (ga["reason"] == NEW_REASON), "group_id"].astype(str))
    in_test = set(test["split_group"].astype(str))
    missing = [g for g in new_groups if g not in in_test]
    if missing:
        raise SystemExit(f"groups assigned to test but absent from test.csv: {missing}")
    print(f"test structures: {len(test)}   test groups: {len(in_test)}   new held-out groups: {len(new_groups)}"
          + ("" if len(new_groups) == 40 else "   (expected 40 - check before submitting)"))

    # one candidate structure per family, by preference; Own families keep every structure
    def structures_of(group):
        rows = test[test["split_group"].astype(str) == group]
        if (rows["source"] == "Own_augmented").all():
            return rows.sort_values("record_id")
        rows = rows.assign(_p=rows["source"].map(PREFERENCE).fillna(9)).sort_values(["_p", "record_id"])
        return rows.iloc[:1].drop(columns="_p")

    cand = {g: structures_of(g) for g in new_groups}
    sizes = {g: int(cand[g]["n_atoms"].iloc[0]) for g in new_groups}
    k = min(a.n_families, len(new_groups))
    forced = [g.strip() for g in a.include.split(",") if g.strip()]
    bad = [g for g in forced if g not in new_groups]
    if bad:
        raise SystemExit(f"--include families not among the new held-out families: {bad}")
    if not a.no_include_largest and new_groups:
        forced.append(max(new_groups, key=lambda g: (sizes[g], g)))
    forced = sorted(set(forced))
    rest = [g for g in new_groups if g not in forced]
    chosen = sorted(forced + stratified_draw(rest, sizes, max(0, k - len(forced)), random.Random(a.seed)))
    dropped = [g for g in new_groups if g not in chosen]

    rows = []
    for g in chosen:
        for _, r in cand[g].iterrows():
            rows.append({**r[["record_id", "source", "split_group", "n_atoms", "has_region_metadata"]].to_dict(),
                         "why": "new_family_draw"})
    have = {r["record_id"] for r in rows}
    if a.own == "all":
        for _, r in test[test["source"] == "Own_augmented"].sort_values("record_id").iterrows():
            if r["record_id"] not in have:
                rows.append({**r[["record_id", "source", "split_group", "n_atoms", "has_region_metadata"]].to_dict(),
                             "why": "own_for_region_routing"})
    man = pd.DataFrame(rows).drop_duplicates("record_id").reset_index(drop=True)
    man["n_atoms"] = man["n_atoms"].astype(int)
    man["label"] = [f"{g} ({SHORT.get(s, s)})" for g, s in zip(man["split_group"], man["source"])]

    # balance over slots (longest first), then order smallest-first inside a slot
    n_slots = a.jobs * a.slots_per_job
    load = np.zeros(n_slots)
    slot_of = {}
    for i in man.sort_values("n_atoms", ascending=False).index:
        s = int(np.argmin(load))
        slot_of[i] = s
        load[s] += man.at[i, "n_atoms"] + 300.0
    man["gpu_slot"] = [slot_of[i] for i in man.index]
    man["job"] = man["gpu_slot"] // a.slots_per_job
    man["slot"] = man["gpu_slot"] % a.slots_per_job
    man = man.sort_values(["job", "slot", "n_atoms", "record_id"]).reset_index(drop=True)
    man["order"] = man.groupby("gpu_slot").cumcount()
    man = man.drop(columns="gpu_slot")
    cols = ["record_id", "label", "source", "split_group", "n_atoms", "has_region_metadata", "why", "job", "slot",
            "order"]
    man[cols].to_csv(out / "md_manifest.csv", index=False)
    pd.DataFrame({"split_group": dropped, "n_atoms_of_candidate": [sizes[g] for g in dropped]}) \
        .to_csv(out / "md_manifest_dropped.csv", index=False)

    print(f"\ndrawn families: {len(chosen)} of {len(new_groups)}  (seed {a.seed}); dropped {len(dropped)}; "
          f"forced in: {', '.join(f'{g}({sizes[g]})' for g in forced) or 'none'}")
    print(f"structures to run: {len(man)}   by source: {man['source'].value_counts().to_dict()}   "
          f"with region metadata: {int(man['has_region_metadata'].sum())}")
    print(f"atoms: min {man['n_atoms'].min()}  median {int(man['n_atoms'].median())}  max {man['n_atoms'].max()}")
    print("\nload per job (sum of n_atoms) and structures per slot:")
    for j in range(a.jobs):
        m = man[man["job"] == j]
        print(f"  job {j}: {len(m):2d} structures, {int(m['n_atoms'].sum()):6d} atoms; per slot "
              + " ".join(f"{int((m['slot'] == s).sum())}" for s in range(a.slots_per_job)))
    print("\ndropped families: " + ", ".join(f"{g}({sizes[g]})" for g in dropped))
    print(f"\nwrote {out / 'md_manifest.csv'}")
    print(f"wrote {out / 'md_manifest_dropped.csv'}")


if __name__ == "__main__":
    main()
