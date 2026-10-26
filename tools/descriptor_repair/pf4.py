#!/usr/bin/env python3
"""pf4.py -- is the pre-NL module (which generated the paper features) equivalent
to the NL module (which the generator can patch)? And does each satisfy rotation
invariance? CPU, float64, read-only. Train/val records only."""
import csv, hashlib, importlib.util, os, sys, json
from pathlib import Path
import numpy as np, torch

P = Path(os.environ["PROJ"]).resolve()
K = int(os.environ.get("NREC", "3"))
Z2S = {14: "Si", 8: "O"}

def sh(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m

def desc(m):
    for n in dir(m):
        o = getattr(m, n)
        if isinstance(o, type) and hasattr(o, "compute_per_center"):
            return o, n
    raise SystemExit("no descriptor class with compute_per_center in %s" % m.__name__)

NLP = P/"scripts_nl"/"torchsoap_autograd_module_periodic_regnorm_nl.py"
OLP = P/"data"/"expanded_holdout40_v1_features"/"src_snapshot"/"torchsoap_autograd_module_periodic_regnorm.py"
print("=" * 88); print("PRE-NL vs NL EQUIVALENCE  +  rotation invariance of each"); print("=" * 88)
for lb, p in (("NL   (generator-compatible)", NLP), ("preNL (made the features)", OLP)):
    print("  %-28s %s  %s" % (lb, sh(p)[:14], p.name))

mn, mo = load(NLP, "_nl"), load(OLP, "_ol")
Cn, nn = desc(mn); Co, no = desc(mo)
print("  descriptor classes: NL.%s  preNL.%s" % (nn, no))
dn = Cn().to(torch.float64).eval(); do = Co().to(torch.float64).eval()

ST = P/"data"/"expanded_holdout40_v1_c1b_stage"
rows = [r for r in csv.DictReader(open(ST/"manifest.csv", newline=""))
        if r.get("split") in ("train", "val")]
print("\n  train+val rows in stage manifest: %d (test excluded before any read)" % len(rows))
rows = sorted(rows, key=lambda r: int(r["n_atoms"]))[:K]

def rot(seed=0):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0: q[:, 0] = -q[:, 0]
    return q

out = []
for r in rows:
    f = ST/r["output_file"]
    with np.load(f, allow_pickle=False) as d:
        pos = np.asarray(d["positions"], float); cell = np.asarray(d["cell"], float)
        syms = [Z2S[int(z)] for z in d["atomic_numbers"]]
        rid = str(d["record_id"]) if "record_id" in d else f.stem
    pt = torch.tensor(pos, dtype=torch.float64); ct = torch.tensor(cell, dtype=torch.float64)
    with torch.no_grad():
        a = dn.compute_per_center(pt, ct, syms).cpu().numpy()
        b = do.compute_per_center(pt, ct, syms).cpu().numpy()
    print("\n  %-18s %4d atoms  split=%s" % (rid[:18], len(syms), r["split"]))
    print("    NL features    %s" % (a.shape,)); print("    preNL features %s" % (b.shape,))
    rec = dict(record=rid, n_atoms=len(syms), split=r["split"],
               nl_shape=list(a.shape), prenl_shape=list(b.shape))
    if a.shape == b.shape:
        sc = max(float(np.abs(b).max()), 1e-30)
        rec["max_abs_diff"] = float(np.abs(a - b).max()); rec["max_rel_diff"] = rec["max_abs_diff"]/sc
        print("    max|NL - preNL| = %.6e   relative %.6e" % (rec["max_abs_diff"], rec["max_rel_diff"]))
        print("    -> %s" % ("IDENTICAL to machine precision" if rec["max_rel_diff"] < 1e-12
              else "DIFFERENT: swapping modules is NOT a no-op"))
    else:
        print("    -> shapes differ; the two modules are not interchangeable")
    R = rot(1)
    for lb, dd in (("NL", dn), ("preNL", do)):
        p2 = torch.tensor(pos @ R.T, dtype=torch.float64); c2 = torch.tensor(cell @ R.T, dtype=torch.float64)
        with torch.no_grad():
            x0 = dd.compute_per_center(pt, ct, syms).cpu().numpy()
            x1 = dd.compute_per_center(p2, c2, syms).cpu().numpy()
        e = float(np.abs(x1 - x0).max()) / max(float(np.abs(x0).max()), 1e-30)
        rec["rot_rel_%s" % lb] = e
        print("    %-5s rotation relative error %.6e  -> %s" % (lb, e,
              "invariant" if e < 1e-10 else "NOT invariant (defect present)"))
    out.append(rec)

Path("pf4.json").write_text(json.dumps(dict(
    nl_sha256=sh(NLP), prenl_sha256=sh(OLP), records=out), indent=2))
print("\nwrote pf4.json")
