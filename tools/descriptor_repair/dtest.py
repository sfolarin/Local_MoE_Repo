#!/usr/bin/env python3
"""Derivative test for the corrected descriptor, with all-image cutoff screening.
A central difference straddling r_cut is invalid (the cosine cutoff is C1 not C2),
so the step is chosen below the nearest pair-to-cutoff gap, or the centre is
EXEMPT and said to be.  CPU, float64, train/val records only."""
import csv, importlib.util, os, sys
import numpy as np, torch
from pathlib import Path

P = Path(os.environ["PROJ"]); Z2S = {14: "Si", 8: "O"}
NREC = int(os.environ.get("NREC", "2")); NPROBE = int(os.environ.get("NPROBE", "3"))

def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

co = load(os.environ["CORR"], "_co")
d = co.TorchSOAPDescriptor().to(torch.float64).eval()
RCUT = float(d.config.r_cut)
print("variant %s   r_cut %.3f A   float64 CPU" % (getattr(co, "DESCRIPTOR_VARIANT", "?"), RCUT))

def gap_all_images(pos, cell):
    """Smallest |distance - r_cut| over ALL periodic images, not minimum image."""
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list
    except ImportError:
        return None
    at = Atoms(positions=pos, cell=cell, pbc=True, symbols=["Si"] * len(pos))
    dd = neighbor_list("d", at, RCUT + 0.5)
    return None if len(dd) == 0 else float(np.min(np.abs(np.asarray(dd) - RCUT)))

def scalar(pos, cell, syms, grad=False):
    pt = torch.tensor(pos, dtype=torch.float64, requires_grad=grad)
    ct = torch.tensor(cell, dtype=torch.float64)
    s = d.compute_per_center(pt, ct, syms).sum()
    if not grad:
        return float(s.detach()), None
    g = torch.autograd.grad(s, pt)[0]
    return float(s.detach()), g.detach().cpu().numpy()

ST = P / "data" / "expanded_holdout40_v1_c1b_stage"
rows = [r for r in csv.DictReader(open(ST / "manifest.csv", newline=""))
        if r["split"] in ("train", "val")]
print("train+val rows %d (test never opened)\n" % len(rows))
rng = np.random.default_rng(5); nfail = nex = 0
for r0 in sorted(rows, key=lambda r: int(r["n_atoms"]))[:NREC]:
    with np.load(ST / r0["output_file"], allow_pickle=False) as z:
        pos = np.asarray(z["positions"], float); cell = np.asarray(z["cell"], float)
        syms = [Z2S[int(k)] for k in z["atomic_numbers"]]
        rid = str(z["record_id"]) if "record_id" in z else r0["output_file"]
    S0, G = scalar(pos, cell, syms, grad=True)
    gap = gap_all_images(pos, cell)
    h1 = 1e-3 if gap is None else min(1e-3, 0.4 * gap)
    print("  %-16s %4d atoms  S=%.6e  nearest pair %s A from r_cut  h=%.2e"
          % (rid[:16], len(syms), S0, ("%.3e" % gap) if gap else "n/a", h1))
    if h1 < 1e-5:
        print("      EXEMPT: no valid central-difference step exists"); nex += 1; continue
    floor = np.finfo(float).eps * abs(S0) / h1
    thr = max(1e-9, 20.0 * floor)
    worst = 0.0
    for i in rng.choice(len(pos), size=min(NPROBE, len(pos)), replace=False):
        for k in range(3):
            errs = []
            for h in (h1, h1 / 2):
                pp = pos.copy(); pp[i, k] += h
                pm = pos.copy(); pm[i, k] -= h
                fd = (scalar(pp, cell, syms)[0] - scalar(pm, cell, syms)[0]) / (2 * h)
                errs.append(fd)
            rich = (4 * errs[1] - errs[0]) / 3
            e = abs(rich - G[i, k]); worst = max(worst, e)
    ok = worst <= thr
    nfail += (not ok)
    print("      Richardson FD vs autograd: max err %.3e   threshold %.3e (floor %.2e)  -> %s"
          % (worst, thr, floor, "pass" if ok else "FAIL"))
print("\nderivative test: %d failed, %d exempt" % (nfail, nex))
raise SystemExit(1 if nfail else 0)
