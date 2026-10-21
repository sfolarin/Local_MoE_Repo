#!/usr/bin/env python3
"""
qc_audit_pool.py      (READ-ONLY)

Quality audit of every structure in a dataset root, using ONLY the reference data (geometry and the
MATPes energy/forces). It never reads a model prediction or error, and the thresholds below are fixed
constants, so a structure cannot be flagged because a model does badly on it.

Checks
  close_contact        any interatomic distance < CLOSE_A (angstrom, periodic)
  coordination_defect  atoms with Si-O coordination other than Si:4 / O:2 (bond cutoff BOND_A);
                       reported as a count; flagged when > 0 (INFORMATION: dangling bonds can be real at interfaces)
  energy_outlier       |robust z| of E/atom > Z_LIMIT, relative to the whole pool (median / MAD)
  density_outlier      |robust z| of volume per atom > Z_LIMIT
  force_outlier        |robust z| of log(max|F|) > Z_LIMIT
  net_force            |sum of reference forces| / N > NET_F (label self-consistency)
  near_duplicate       another record with the same atom count, total energy (to 1e-4 eV) and sum|F| (to 1e-4)

Probe records (default: the structures with the largest model errors on the test split) are printed with
their flags purely for inspection; they play no role in any criterion.

Run from Local_MoE_MATPES_R2SCAN_Expanded_v1/:
    python qc_audit_pool.py --dataset data/expanded_holdout40_v1
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

CLOSE_A = 1.2      # angstrom
BOND_A = 2.0       # Si-O bond cutoff, angstrom
Z_LIMIT = 6.0
NET_F = 1e-3       # eV/A per atom
DEFAULT_PROBE = "own_0146,own_0147,external_0150,external_0149,external_0223,external_0090"


def robust_z(x):
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 0.6745 * (x - med) / max(mad, 1e-12)


def structure_metrics(z, pos, cell, pbc, energy, forces):
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    n = len(z)
    at = Atoms(numbers=z, positions=pos, cell=cell, pbc=pbc)
    i, j, d = neighbor_list("ijd", at, BOND_A)
    min_d = float(d.min()) if len(d) else float("nan")
    cn = np.zeros(n, int)
    for a, b, dist in zip(i, j, d):
        if dist < BOND_A and z[a] != z[b]:          # count Si-O pairs only
            cn[a] += 1
    wrong = ((z == 14) & (cn != 4)) | ((z == 8) & (cn != 2))
    fn = np.linalg.norm(forces, axis=1)
    return dict(n_atoms=n, E_per_atom_eV=energy / n, vol_per_atom_A3=abs(np.linalg.det(cell)) / n, min_dist_A=min_d,
                n_defect_atoms=int(wrong.sum()), max_force=float(fn.max()), rms_force=float(np.sqrt((forces ** 2).mean())),
                net_force_per_atom=float(np.linalg.norm(forces.sum(axis=0)) / n), E_total=energy, sum_abs_force=float(fn.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True, help="dataset root (contains manifest.csv)")
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--out", default="qc_audit_pool.csv")
    args = ap.parse_args()

    root = args.dataset.expanduser().resolve()
    man = pd.read_csv(root / "manifest.csv")
    rows = []
    for r in man.itertuples():
        with np.load(root / r.data_file, allow_pickle=False) as d:
            m = structure_metrics(np.asarray(d["atomic_numbers"]).reshape(-1), np.asarray(d["positions"], float),
                                  np.asarray(d["cell"], float), np.asarray(d["pbc"], bool).reshape(3),
                                  float(d["mace_energy_total_eV"]), np.asarray(d["mace_forces"], float))
        assert m["n_atoms"] == int(r.n_atoms), r.record_id
        rows.append(dict(record_id=r.record_id, source=r.source, family=r.family, split=r.split, **m))
    q = pd.DataFrame(rows)

    q["z_energy"] = robust_z(q.E_per_atom_eV)
    q["z_density"] = robust_z(q.vol_per_atom_A3)
    q["z_logmaxF"] = robust_z(np.log(q.max_force.clip(lower=1e-9)))
    sig = q.groupby(["n_atoms", q.E_total.round(4), q.sum_abs_force.round(4)]).record_id.transform("count")
    q["near_duplicate"] = sig > 1
    flags = pd.DataFrame({
        "close_contact": q.min_dist_A < CLOSE_A,
        "coordination_defect": q.n_defect_atoms > 0,
        "energy_outlier": q.z_energy.abs() > Z_LIMIT,
        "density_outlier": q.z_density.abs() > Z_LIMIT,
        "force_outlier": q.z_logmaxF.abs() > Z_LIMIT,
        "net_force": q.net_force_per_atom > NET_F,
        "near_duplicate": q.near_duplicate})
    q["qc_flags"] = flags.apply(lambda r: ",".join(k for k, v in r.items() if v), axis=1)
    q = pd.concat([q, flags.add_prefix("f_")], axis=1)
    q.to_csv(args.out, index=False)

    print(f"{len(q)} structures audited in {root.name} (thresholds: close<{CLOSE_A} A, bond<{BOND_A} A, |z|>{Z_LIMIT}, net force>{NET_F}; no model errors used)")
    print("\nflag counts by split:")
    print(flags.groupby(q.split).sum().astype(int).T.assign(total=flags.sum().astype(int)).to_string())
    print(f"\nrecords with at least one flag: {int((q['qc_flags'] != '').sum())}")
    hard = q[flags[["close_contact", "energy_outlier", "density_outlier", "force_outlier", "net_force", "near_duplicate"]].any(axis=1)]
    print(f"records with a flag other than coordination_defect: {len(hard)}")
    if len(hard):
        print(hard[["record_id", "source", "family", "split", "n_atoms", "qc_flags"]].to_string(index=False, max_colwidth=46))
    dup = q[q.near_duplicate].sort_values(["n_atoms", "E_total"])
    if len(dup):
        print("\nnear-duplicate groups (same atom count, E_total and sum|F|):")
        print(dup[["record_id", "source", "split", "n_atoms", "E_total", "sum_abs_force"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        span = dup.groupby([dup.n_atoms, dup.E_total.round(4), dup.sum_abs_force.round(4)]).split.nunique()
        print(f"near-duplicate groups spanning more than one split (leakage between splits): {int((span > 1).sum())} of {len(span)}")
    print("\ncoordination defects by split (count of structures with any):")
    print(q.groupby("split").apply(lambda g: pd.Series({"structures": len(g), "with_defects": int((g.n_defect_atoms > 0).sum()),
                                                       "median_defect_atoms": float(g.n_defect_atoms.median())})).to_string())
    probe = [p.strip() for p in args.probe.split(",") if p.strip()]
    pr = q[q.record_id.isin(probe)]
    if len(pr):
        print("\nPROBE records (inspection only; not part of any criterion):")
        print(pr[["record_id", "split", "n_atoms", "E_per_atom_eV", "z_energy", "z_logmaxF", "min_dist_A", "n_defect_atoms", "qc_flags"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        pct = lambda col, v: 100 * (q[col] < v).mean()
        print("   percentile in the pool of E/atom: " + ", ".join(f"{r.record_id} {pct('E_per_atom_eV', r.E_per_atom_eV):.0f}%" for r in pr.itertuples()))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
