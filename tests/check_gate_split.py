#!/usr/bin/env python3
"""
check_gate_split.py

Read-only check that the dense gate trained on exactly the intended split.
Compares, record by record:
  gate     = <run>/dataset_split.csv            (record id = stem of shard_file)
  holdout  = $EXT/holdout_split_n40.csv          (record_id, new_split, ...)
  manifest = <routing dataset>/dataset_manifest.csv (record_id, split, ...)

Exit status 0 only if every FAIL-type check passes.

Usage:
    python3 check_gate_split.py GATE_SPLIT.csv HOLDOUT_TABLE.csv ROUTING_MANIFEST.csv
"""
import csv
import os
import sys
from collections import Counter, defaultdict


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def need(rows, cols, name):
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        print(f"ERROR: {name} lacks columns {missing}; has {list(rows[0])}")
        sys.exit(2)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    gate = read(sys.argv[1])
    hold = read(sys.argv[2])
    man = read(sys.argv[3])
    for rows, name in ((gate, "gate"), (hold, "holdout"), (man, "manifest")):
        if not rows:
            print(f"ERROR: {name} table is empty")
            sys.exit(2)
    need(gate, ["source_file", "shard_file", "duplicate_group", "n_atoms", "split"], "gate")
    need(hold, ["record_id", "old_split", "new_split", "n_atoms", "geometry_sha256"], "holdout")
    need(man, ["record_id", "source_file", "duplicate_group", "n_atoms", "split",
               "geometry_sha256"], "manifest")

    bad = 0

    def fail(msg):
        nonlocal bad
        bad += 1
        print("FAIL:", msg)

    # --- index each table by record id; refuse duplicates
    g = {}
    for r in gate:
        rid = os.path.splitext(os.path.basename(r["shard_file"]))[0]
        if rid in g:
            fail(f"gate table repeats record {rid}")
        g[rid] = r
    h = {}
    for r in hold:
        if r["record_id"] in h:
            fail(f"holdout table repeats record {r['record_id']}")
        h[r["record_id"]] = r
    m = {}
    for r in man:
        if r["record_id"] in m:
            fail(f"manifest repeats record {r['record_id']}")
        m[r["record_id"]] = r

    print(f"records: gate={len(g)}  holdout={len(h)}  manifest={len(m)}")

    # --- same record sets
    for a, an, b, bn in ((g, "gate", h, "holdout"), (g, "gate", m, "manifest")):
        only_a = sorted(set(a) - set(b))
        only_b = sorted(set(b) - set(a))
        if only_a or only_b:
            fail(f"{an} vs {bn}: {len(only_a)} only in {an} {only_a[:5]}, "
                 f"{len(only_b)} only in {bn} {only_b[:5]}")
    common = sorted(set(g) & set(h) & set(m))

    # --- the split labels agree record by record
    mism_h = [k for k in common if g[k]["split"] != h[k]["new_split"]]
    mism_m = [k for k in common if g[k]["split"] != m[k]["split"]]
    if mism_h:
        fail(f"{len(mism_h)} records where gate split != holdout new_split; "
             f"first: {[(k, g[k]['split'], h[k]['new_split']) for k in mism_h[:5]]}")
    if mism_m:
        fail(f"{len(mism_m)} records where gate split != manifest split; "
             f"first: {[(k, g[k]['split'], m[k]['split']) for k in mism_m[:5]]}")

    # --- same structure behind each id
    sha = [k for k in common if h[k]["geometry_sha256"] != m[k]["geometry_sha256"]]
    if sha:
        fail(f"{len(sha)} records where holdout and manifest geometry_sha256 differ; "
             f"first: {sha[:5]}")
    nat = [k for k in common
           if not (g[k]["n_atoms"] == h[k]["n_atoms"] == m[k]["n_atoms"])]
    if nat:
        fail(f"{len(nat)} records where n_atoms differ between tables; first: {nat[:5]}")
    src = [k for k in common if g[k]["source_file"] != m[k]["source_file"]]
    if src:
        fail(f"{len(src)} records where gate and manifest source_file differ; first: {src[:5]}")

    # --- no duplicate group spans two splits (leakage)
    grp = defaultdict(set)
    for k in common:
        grp[g[k]["duplicate_group"]].add(g[k]["split"])
    span = {gr: sorted(s) for gr, s in grp.items() if len(s) > 1}
    if span:
        fail(f"{len(span)} duplicate groups span more than one split; "
             f"first: {list(span.items())[:5]}")

    # --- report
    print("gate split counts:", dict(Counter(g[k]["split"] for k in g)))
    cross = Counter((h[k]["old_split"], h[k]["new_split"]) for k in h)
    print("old_split -> new_split (holdout table):")
    for (o, n), c in sorted(cross.items()):
        print(f"   {o:>5} -> {n:<5} {c}")
    if "gamma_family" in hold[0]:
        gx = Counter((h[k]["gamma_family"], g[k]["split"]) for k in common)
        print("gamma_family x gate split:", dict(sorted(gx.items())))

    print()
    if bad:
        print(f"{bad} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
