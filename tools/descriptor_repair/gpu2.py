#!/usr/bin/env python3
"""gpu2.py -- step 2 of the authorized scope, in one allocation:
(a) bounded larger-structure validation of the corrected module on GPU,
(b) forward/backward time and peak allocated memory, corrected vs original.
Strict CUDA: fails rather than silently timing the CPU.  Train/val records only."""
import csv, hashlib, importlib.util, json, os, sys, time
import numpy as np, torch
from pathlib import Path

P = Path(os.environ["PROJ"]); Z2S = {14: "Si", 8: "O"}
DEV = torch.device(os.environ.get("DEVICE", "cuda"))
if DEV.type == "cuda" and not torch.cuda.is_available():
    raise SystemExit("FAIL: cuda requested but torch.cuda.is_available() is False. "
                     "Refusing to report a CPU timing as a GPU timing.")
SIZES = [int(x) for x in os.environ.get("SIZES", "432,900,1500,2232").split(",")]
REPS = int(os.environ.get("REPS", "5"))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
def sync():
    if DEV.type == "cuda": torch.cuda.synchronize(DEV)

CORR, ORIG = os.environ["CORR"], os.environ["ORIG"]
print("=" * 94); print("STEP 2: bounded GPU validation + cost measurement"); print("=" * 94)
print("corrected %s\n  %s" % (sha(CORR)[:14], CORR))
print("original  %s\n  %s" % (sha(ORIG)[:14], ORIG))
print("torch %s | numpy %s | python %s" % (torch.__version__, np.__version__, sys.version.split()[0]))
print("device %s%s | float64 | %d reps after 2 warm-up" % (DEV,
      (" (%s)" % torch.cuda.get_device_name(DEV)) if DEV.type == "cuda" else "", REPS))
print("sync: torch.cuda.synchronize() before and after each timed block"
      if DEV.type == "cuda" else "sync: n/a")

dc = load(CORR, "_c").TorchSOAPDescriptor().to(torch.float64).to(DEV).eval()
do = load(ORIG, "_o").TorchSOAPDescriptor().to(torch.float64).to(DEV).eval()

ST = P / "data" / "expanded_holdout40_v1_c1b_stage"
rows = [r for r in csv.DictReader(open(ST / "manifest.csv", newline=""))
        if r["split"] in ("train", "val")]
print("\ntrain+val rows %d (test never opened)" % len(rows))
pick = []
for t in SIZES:
    c = [r for r in rows if int(r["n_atoms"]) <= t]
    if not c: continue
    b = max(c, key=lambda r: int(r["n_atoms"]))
    if all(int(b["n_atoms"]) != int(x["n_atoms"]) for x in pick): pick.append(b)
print("selected sizes: %s" % [int(r["n_atoms"]) for r in pick])

def rot(s):
    q, r = np.linalg.qr(np.random.default_rng(s).standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0: q[:, 0] = -q[:, 0]
    return q

def timed(d, pos, cell, syms):
    p0 = torch.tensor(pos, dtype=torch.float64, device=DEV)
    ct = torch.tensor(cell, dtype=torch.float64, device=DEV)
    def once():
        p = p0.clone().requires_grad_(True)
        s = d.compute_per_center(p, ct, syms).sum()
        return torch.autograd.grad(s, p)[0]
    for _ in range(2): once()
    sync()
    if DEV.type == "cuda": torch.cuda.reset_peak_memory_stats(DEV)
    t0 = time.perf_counter()
    for _ in range(REPS): once()
    sync()
    dt = (time.perf_counter() - t0) / REPS
    pk = torch.cuda.max_memory_allocated(DEV) / 2**30 if DEV.type == "cuda" else float("nan")
    return dt, pk

print("\n%-16s %6s %10s %10s %7s %9s %9s %11s" % ("record","atoms","rot rel",
      "dense/nl","ratio","corr s","orig s","corr GiB"))
out = []
for r0 in pick:
    with np.load(ST / r0["output_file"], allow_pickle=False) as z:
        pos = np.asarray(z["positions"], float); cell = np.asarray(z["cell"], float)
        syms = [Z2S[int(k)] for k in z["atomic_numbers"]]
        rid = str(z["record_id"]) if "record_id" in z else r0["output_file"]
    R = rot(1)
    pt = torch.tensor(pos, dtype=torch.float64, device=DEV)
    ct = torch.tensor(cell, dtype=torch.float64, device=DEV)
    p2 = torch.tensor(pos @ R.T, dtype=torch.float64, device=DEV)
    c2 = torch.tensor(cell @ R.T, dtype=torch.float64, device=DEV)
    with torch.no_grad():
        x0 = dc.compute_per_center(pt, ct, syms); x1 = dc.compute_per_center(p2, c2, syms)
        sc = float(x0.abs().max().clamp(min=1e-30))
        rote = float((x1 - x0).abs().max()) / sc
        try:
            xd = dc.compute_per_center_dense(pt, ct, syms)
            dnl = float((xd - x0).abs().max()) / sc
        except Exception: dnl = float("nan")
        y0 = do.compute_per_center(pt, ct, syms); y1 = do.compute_per_center(p2, c2, syms)
        rote_o = float((y1 - y0).abs().max()) / max(float(y0.abs().max()), 1e-30)
    tc, pc = timed(dc, pos, cell, syms)
    to, po = timed(do, pos, cell, syms)
    print("%-16s %6d %10.3e %10.3e %7.3f %9.4f %9.4f %11.3f"
          % (rid[:16], len(syms), rote, dnl, tc/to if to else float("nan"), tc, to, pc))
    out.append(dict(record=rid, split=r0["split"], n_atoms=len(syms), rot_rel_corrected=rote,
                    rot_rel_original=rote_o, dense_nl_rel=dnl, corrected_s=tc, original_s=to,
                    ratio=tc/to if to else None, corrected_peak_GiB=pc, original_peak_GiB=po))

bad = [r for r in out if r["rot_rel_corrected"] > 1e-10]
good = [r for r in out if r["rot_rel_original"] > 1e-3]
print("\ncorrected invariant on all %d sizes: %s" % (len(out), "YES" if not bad else "NO"))
print("original FAILS the same test on %d of %d (as it must, or the test is inert)"
      % (len(good), len(out)))
rr = [r["ratio"] for r in out if r["ratio"]]
if rr:
    Rmax = max(rr)
    print("cost ratio corrected/original: min %.3f median %.3f max %.3f"
          % (min(rr), float(np.median(rr)), Rmax))
    print("  common net 3.20 h x %.3f = %.2f h     C1b 0.68 h x %.3f = %.2f h"
          % (Rmax, 3.20*Rmax, Rmax, 0.68*Rmax))
if DEV.type == "cuda":
    print("descriptor-only peak allocated: max %.3f GiB (NOT the 50.20 GiB full C1b step)"
          % max(r["corrected_peak_GiB"] for r in out))
Path("gpu2.json").write_text(json.dumps(dict(
    corrected_sha256=sha(CORR), original_sha256=sha(ORIG), torch=torch.__version__,
    numpy=np.__version__, python=sys.version.split()[0], device=str(DEV),
    gpu=torch.cuda.get_device_name(DEV) if DEV.type=="cuda" else None,
    dtype="float64", reps=REPS, warmup=2, synchronised=DEV.type=="cuda",
    records=out, ratio_max=max(rr) if rr else None), indent=2))
print("\nwrote gpu2.json")
