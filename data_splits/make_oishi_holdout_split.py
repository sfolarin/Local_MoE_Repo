#!/usr/bin/env python3
"""
make_oishi_holdout_split.py     (DRY RUN: writes a proposed split CSV, trains nothing)

Re-split the expanded pool (Own_augmented, Oishi_all179, Oishi_relaxed52, Siddharth) so that many
Oishi interface-energy families are held out, with every split decision made at the level of the
PHASE PAIR across all sources:

  * test = old test records (all sources)  +  every record of each held-out Oishi gamma pair
  * held-out gamma pairs = the pairs of the gamma families that are currently unseen (val/test),
    plus a seeded random draw from the remaining gamma pairs, balanced across terciles of the
    MATPes gamma, until --n-holdout gamma families are held out. No model error is looked at.
  * validation = old validation records whose pair is not held out, topped up with Oishi pairs
    that have NO gamma family, until Oishi validation is --val-fraction of the Oishi records.
    No gamma family is ever used for validation.
  * train = everything else
  * Oishi pairs that are shared with --protect-sources (default Siddharth, Own_augmented) are kept entirely in
    TRAIN, so those sources are not drained by the hold-out; use --hold-shared to turn this off

Audit: no phase pair and no geometry hash appears in two splits.

Inputs : data/expanded_grouped_4608_v1/manifest.csv (pool), oishi_gamma_candidates.csv
Output : <out> (record_id, source, family, pair, old_split, new_split, gamma_family) + printed audit

Run from External_Benchmark_Oishi2025/:
    python make_oishi_holdout_split.py --n-holdout 50
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
from pathlib import Path

import numpy as np
import pandas as pd

HET_DEFAULT = (_os.environ.get("MOE_HET", "."))
ap = argparse.ArgumentParser()
ap.add_argument("--het", default=HET_DEFAULT)
ap.add_argument("--n-holdout", type=int, default=50, help="number of Oishi gamma families to hold out (max = number with gamma)")
ap.add_argument("--seed", type=int, default=20261009)
ap.add_argument("--val-fraction", type=float, default=0.10)
ap.add_argument("--out", default="holdout_split_v1.csv")
ap.add_argument("--protect-sources", default="Siddharth,Own_augmented",
                help="Oishi pairs that also contain records of these sources stay in TRAIN (never held out)")
ap.add_argument("--hold-shared", action="store_true", help="disable the protection: shared pairs may be held out")
args = ap.parse_args()

HET = Path(args.het)
EXT = HET / "External_Benchmark_Oishi2025"
PROJ = HET / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
pd.set_option("display.width", 200)

pool = pd.read_csv(PROJ / "data/expanded_grouped_4608_v1/manifest.csv")
cand = pd.read_csv(EXT / "oishi_gamma_candidates.csv", keep_default_na=False)
g = cand[cand.drop_reason.astype(str) == ""][["case", "gamma_MATPES_J_m2"]].rename(columns={"case": "family"})

norm = lambda s: "__".join(sorted(str(s).split("__")))
pool["pair"] = pool.phase_pair_group.map(norm)
pool["old_split"] = pool["split"]
pool["is_oishi"] = pool.source.isin(["Oishi_all179", "Oishi_relaxed52"])
pool["gamma_family"] = pool.is_oishi & pool.family.isin(set(g.family))
assert pool.record_id.is_unique

# ---- audit of the OLD split under the sorted-label rule ---------------------------------------
old_span = pool.groupby("pair").old_split.nunique()
leaky = old_span[old_span > 1].index
print(f"OLD split audit (phase-pair labels sorted): {len(leaky)} pairs span more than one split")
if len(leaky):
    t = pool[pool.pair.isin(leaky)].groupby("pair").agg(
        sources=("source", lambda x: ",".join(sorted(set(x)))), splits=("old_split", lambda x: ",".join(sorted(set(x)))),
        records=("record_id", "size"))
    print(t.sort_values("records", ascending=False).head(15).to_string())

prot = [x.strip() for x in args.protect_sources.split(",") if x.strip()]
shared = set(pool[pool.is_oishi].pair) & set(pool[pool.source.isin(prot)].pair)
protect = bool(shared) and not args.hold_shared
print(f"\nOishi phase pairs shared with {prot}: {len(shared)} "
      f"({'kept in TRAIN' if protect else 'NOT protected'})")
if shared:
    sh = pool[pool.pair.isin(shared)].groupby(["pair", "source"]).size().unstack(fill_value=0)
    print(sh.assign(total=sh.sum(axis=1)).sort_values("total", ascending=False).head(10).to_string())

gam_rec = pool[pool.gamma_family]
fam_pair = gam_rec.drop_duplicates("family")[["family", "pair"]].merge(g, on="family")
n_gam = fam_pair.family.nunique()
assert args.n_holdout <= n_gam, f"only {n_gam} gamma families can be held out"
gp = fam_pair.groupby("pair").family.nunique()                 # gamma families per phase pair
if protect:
    gp = gp[~gp.index.isin(shared)]
    n_gam = int(gp.sum())
    print(f"gamma families eligible for hold-out after protection: {n_gam}")
pair_gamma = fam_pair.groupby("pair").gamma_MATPES_J_m2.mean()

# ---- held-out gamma pairs ---------------------------------------------------------------------
hold = set(gam_rec[gam_rec.old_split.isin(["val", "test"])].pair) - (shared if protect else set())
rng = np.random.default_rng(args.seed)
rest = [p for p in gp.index if p not in hold]
terc = pd.qcut(pair_gamma.reindex(rest), 3, labels=False, duplicates="drop")
queues = []
for t in sorted(terc.dropna().unique()):
    q = [p for p in rest if terc[p] == t]
    rng.shuffle(q)
    queues.append(q)
count = lambda hs: int(gp.reindex(sorted(hs)).fillna(0).sum())
while count(hold) < args.n_holdout and any(queues):
    for q in queues:
        if q and count(hold) < args.n_holdout:
            hold.add(q.pop())

old_test_pairs = set(pool[pool.old_split == "test"].pair) - (shared if protect else set())
test_pairs = old_test_pairs | hold
pool["new_split"] = "train"
pool.loc[pool.pair.isin(test_pairs), "new_split"] = "test"

# ---- validation -------------------------------------------------------------------------------
keep_val = (pool.old_split == "val") & ~pool.pair.isin(test_pairs) & ~pool.gamma_family & ~pool.pair.isin(shared if protect else set())
pool.loc[keep_val, "new_split"] = "val"
oi_nontest = pool[pool.is_oishi & (pool.new_split != "test")]
target_val = int(round(args.val_fraction * len(oi_nontest)))
have_val = int((pool.is_oishi & (pool.new_split == "val")).sum())
gamma_pairs_all = set(gp.index)
cand_pairs = sorted(set(pool[pool.is_oishi & (pool.new_split == "train")].pair) - gamma_pairs_all
                    - set(pool[pool.new_split == "val"].pair) - (shared if protect else set()))
rng.shuffle(cand_pairs)
for p in cand_pairs:
    if have_val >= target_val:
        break
    m = pool.pair == p
    if (pool.loc[m, "new_split"] != "train").any():
        continue
    pool.loc[m, "new_split"] = "val"
    have_val += int((m & pool.is_oishi).sum())

if protect:
    pool.loc[pool.pair.isin(shared), "new_split"] = "train"

# ---- audit ------------------------------------------------------------------------------------
split_per_pair = pool.groupby("pair").new_split.nunique()
split_per_hash = pool.groupby("geometry_sha256").new_split.nunique()
bad_pairs, bad_hash = split_per_pair[split_per_pair > 1], split_per_hash[split_per_hash > 1]
print("AUDIT: phase pairs in more than one split:", len(bad_pairs), "| geometries in more than one split:", len(bad_hash))
assert len(bad_pairs) == 0 and len(bad_hash) == 0, "leakage in the proposed split"

print(f"\nOishi gamma families eligible: {n_gam} | held out: {count(hold)} in {len(hold)} phase pairs "
      f"({len(hold & set(gam_rec[gam_rec.old_split.isin(['val', 'test'])].pair))} pairs already unseen today) | "
      f"left in training: {n_gam - count(hold)}")
hf = fam_pair[fam_pair.pair.isin(hold)]
tf = fam_pair[~fam_pair.pair.isin(hold)]
print("MATPes gamma of held-out families: min %.4f median %.4f max %.4f | training families: min %.4f median %.4f max %.4f"
      % (hf.gamma_MATPES_J_m2.min(), hf.gamma_MATPES_J_m2.median(), hf.gamma_MATPES_J_m2.max(),
         tf.gamma_MATPES_J_m2.min() if len(tf) else np.nan, tf.gamma_MATPES_J_m2.median() if len(tf) else np.nan,
         tf.gamma_MATPES_J_m2.max() if len(tf) else np.nan))
print("\nrecords by source and split, OLD:")
print(pool.groupby(["source", "old_split"]).size().unstack(fill_value=0).to_string())
print("\nrecords by source and split, NEW:")
print(pool.groupby(["source", "new_split"]).size().unstack(fill_value=0).to_string())
moved = pool[(pool.old_split != pool.new_split)]
print("\nrecords that change split (old -> new):")
print(moved.groupby(["source", "old_split", "new_split"]).size().to_string())
print(f"\nOishi records left for TRAINING: {int(((pool.is_oishi) & (pool.new_split == 'train')).sum())} "
      f"of {int(pool.is_oishi.sum())} (the current model trains on {int(((pool.is_oishi) & (pool.old_split == 'train')).sum())})")
print(f"gamma families in TEST: {count(hold)} | in TRAIN: {int(pool[pool.gamma_family & (pool.new_split == 'train')].family.nunique())} | in VAL: 0")

out = pool[["record_id", "source", "family", "pair", "old_split", "new_split", "gamma_family", "n_atoms", "geometry_sha256"]]
out.to_csv(args.out, index=False)
print(f"\nsaved {args.out} ({len(out)} records)")
