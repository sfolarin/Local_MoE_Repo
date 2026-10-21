#!/usr/bin/env python3
"""
test_torchsoap_neighbour_list.py

Regression and scaling test for the neighbour-list TorchSOAP
(torchsoap_autograd_module_periodic_regnorm_nl.py).

It answers three questions, in this order:

 1. TOPOLOGY.  Does the neighbour search return exactly the set of periodic
    pairs with |R_j - R_i + S @ cell + eps| < r_cut?  (Independent brute force,
    no TorchSOAP code involved.)
 2. SAME FUNCTION.  Do the new per-centre features, the forces (d/dpositions)
    and the cell gradient (d/dcell) agree with the dense implementation?
    Reference = the ORIGINAL module file if --original is given (recommended),
    otherwise the verbatim copy embedded in the new module
    (compute_per_center_dense).
 3. COST.  How do time and peak memory scale with N, dense vs neighbour list?
    (--scaling)

Nothing is written to disk except optional --json.  Read-only with respect to
all inputs.

Usage
-----
  python test_torchsoap_neighbour_list.py \
      --original $F \
      --structures path1.extxyz path2.extxyz ... \
      --scaling

Exit code 0 only if every check passes.
"""

from __future__ import annotations
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
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def synth(nx, ny, nz, tilt=(0.0, 0.0, 0.0), seed=0, a=2.47, jitter=0.5,
          unwrap=False):
    """Jittered-lattice Si/O box.  min distance >= a - 2*jitter*sqrt(3)/... is
    not enforced, but a=2.47, jitter=0.5 keeps every pair well above 0.5 A."""
    rng = np.random.default_rng(seed)
    lx, ly, lz = nx * a, ny * a, nz * a
    cell = np.array([
        [lx, 0.0, 0.0],
        [tilt[0] * lx, ly, 0.0],
        [tilt[1] * lx, tilt[2] * ly, lz],
    ])
    grid = np.array([[i, j, k] for i in range(nx) for j in range(ny)
                     for k in range(nz)], dtype=float)
    frac = (grid + 0.5) / np.array([nx, ny, nz])
    pos = frac @ cell + rng.uniform(-jitter, jitter, size=(len(grid), 3))
    if unwrap:  # push some atoms into other periodic images
        n_shift = rng.integers(-2, 3, size=(len(grid), 3))
        mask = rng.random(len(grid)) < 0.4
        pos = pos + (n_shift * mask[:, None]) @ cell
    n = len(grid)
    sym = np.where(rng.random(n) < 1.0 / 3.0, "Si", "O")
    return pos, cell, list(sym)


def brute_force_pair_count(pos, cell, r_cut, eps):
    """Independent count of (i, j, S) with |d + eps| < r_cut, home self excluded.
    The shift range is derived from the actual fractional spread of the atoms
    (unwrapped inputs need a wider range) plus the cutoff in plane spacings."""
    n = len(pos)
    inv = np.linalg.inv(cell)
    frac = pos @ inv
    spread = np.ceil(frac.max(axis=0) - frac.min(axis=0)).astype(int)
    h = 1.0 / np.linalg.norm(inv, axis=0)
    ext = spread + np.ceil(r_cut / h).astype(int) + 1
    rx, ry, rz = (range(-int(e), int(e) + 1) for e in ext)
    shifts = np.array([[a, b, c] for a in rx for b in ry for c in rz], float)
    shift_cart = shifts @ cell
    d0 = pos[None, :, :] - pos[:, None, :]            # (N, N, 3) = R_j - R_i
    total = 0
    for s, sc in zip(shifts, shift_cart):
        d = d0 + sc
        r = np.linalg.norm(d + eps, axis=-1)
        m = r < r_cut
        if not s.any():
            m &= ~np.eye(n, dtype=bool)
        total += int(m.sum())
    return total


def eval_fn(fn, pos_np, cell_np, symbols, w, device):
    pos = torch.tensor(pos_np, dtype=torch.float64, device=device, requires_grad=True)
    cell = torch.tensor(cell_np, dtype=torch.float64, device=device, requires_grad=True)
    d = fn(pos, cell, symbols)
    loss = (d * w).sum()
    gp, gc = torch.autograd.grad(loss, (pos, cell))
    return d.detach().cpu().numpy(), gp.cpu().numpy(), gc.cpu().numpy()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_and_mem(fn, pos_np, cell_np, symbols, w, device, reps):
    """forward + backward, median wall time and peak allocated GiB (CUDA only)."""
    def once():
        pos = torch.tensor(pos_np, dtype=torch.float64, device=device, requires_grad=True)
        cell = torch.tensor(cell_np, dtype=torch.float64, device=device)
        d = fn(pos, cell, symbols)
        torch.autograd.grad((d * w).sum(), pos)

    once()  # warm-up
    sync(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    ts = []
    for _ in range(reps):
        sync(device)
        t0 = time.perf_counter()
        once()
        sync(device)
        ts.append(time.perf_counter() - t0)
    peak = float("nan")
    if device.type == "cuda":
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    return float(np.median(ts)), peak


def fit_exponent(ns, ts):
    ns = np.asarray(ns, float)
    ts = np.asarray(ts, float)
    ok = np.isfinite(ts) & (ts > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log(ns[ok]), np.log(ts[ok]), 1)[0])


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--module", type=Path,
                    default=REPO_ROOT / 'src/local_moe/torchsoap_autograd_module_periodic_regnorm_nl.py',
                    help="the neighbour-list module under test")
    ap.add_argument("--original", type=Path, default=None,
                    help="the ORIGINAL dense module file; used as the reference")
    ap.add_argument("--structures", nargs="*", default=[],
                    help="ASE-readable structure files (last frame is used)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "matscipy", "ase"])
    ap.add_argument("--tol-feat", type=float, default=1e-10)
    ap.add_argument("--tol-grad", type=float, default=1e-8)
    ap.add_argument("--max-dense-atoms", type=int, default=None,
                    help="largest N on which the dense path is run (default 5000 on "
                         "cuda, 1000 on cpu)")
    ap.add_argument("--scaling", action="store_true", help="run the scaling table")
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[4, 6, 8, 10, 12, 14, 16],
                    help="edge lengths n of n x n x n jittered boxes for --scaling")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.max_dense_atoms is None:
        args.max_dense_atoms = 5000 if device.type == "cuda" else 1000

    new = load_module(args.module, "tso_new")
    cfg = new.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4,
                              nl_backend=args.backend)
    nl = new.TorchSOAPDescriptor(cfg).to(device)
    nl._nl_info_printed = True  # keep the log quiet
    nl._periodic_shift_info_printed = True
    soap_dim = nl.soap_dim

    ref_name = "embedded dense copy (compute_per_center_dense)"
    ref_fn = nl.compute_per_center_dense
    orig = None
    if args.original is not None:
        orig = load_module(args.original, "tso_orig")
        ocfg = orig.TorchSOAPConfig(species=("Si", "O"), r_cut=5.0, n_max=6, l_max=4)
        od = orig.TorchSOAPDescriptor(ocfg).to(device)
        od._periodic_shift_info_printed = True
        ref_fn = od.compute_per_center
        ref_name = f"ORIGINAL file {args.original.name}"

    print(f"device={device} | backend request={args.backend} | reference = {ref_name}")
    print(f"tolerances: features {args.tol_feat:g} | gradients {args.tol_grad:g}")

    # ---------------- case list ----------------
    cases = []
    for label, (nx, ny, nz), tilt, unwrap in [
        ("2-atom 1x1x2 cell (125 shifts)", (1, 1, 2), (0.0, 0.0, 0.0), False),
        ("8-atom 2x2x2",                   (2, 2, 2), (0.0, 0.0, 0.0), False),
        ("27-atom 3x3x3",                  (3, 3, 3), (0.0, 0.0, 0.0), False),
        ("64-atom 4x4x4, atoms unwrapped", (4, 4, 4), (0.0, 0.0, 0.0), True),
        ("125-atom triclinic 5x5x5",       (5, 5, 5), (0.31, -0.27, 0.22), False),
        ("thin slab 8x8x2 (z = 4.9 A)",    (8, 8, 2), (0.1, 0.0, 0.0), False),
        ("216-atom triclinic, unwrapped",  (6, 6, 6), (-0.2, 0.15, 0.35), True),
        ("512-atom 8x8x8 (single shift)",  (8, 8, 8), (0.0, 0.0, 0.0), False),
    ]:
        pos, cell, sym = synth(nx, ny, nz, tilt=tilt, seed=len(cases), unwrap=unwrap)
        cases.append((label, pos, cell, sym))

    for p in args.structures:
        at = read(p)
        cases.append((f"file {Path(p).name} (N={len(at)})",
                      np.array(at.positions, float), np.array(at.cell.array, float),
                      list(at.get_chemical_symbols())))

    results = []
    all_ok = True
    print("\n=== 1+2: topology and same-function checks ===")
    hdr = (f"{'case':44s} {'N':>5s} {'edges':>8s} {'topo':>5s} "
           f"{'feat|d|':>9s} {'dF|d|':>9s} {'dCell|d|':>9s}  verdict")
    print(hdr)
    for label, pos, cell, sym in cases:
        n = len(pos)
        # topology (independent brute force vs the module's own search)
        i_, j_, S_, used = nl._neighbour_edges(torch.tensor(pos, device=device),
                                               torch.tensor(cell, device=device))
        i_, j_, S_ = i_.cpu().numpy(), j_.cpu().numpy(), S_.cpu().numpy()
        d = pos[j_] - pos[i_] + S_ @ cell
        r = np.linalg.norm(d + cfg.eps, axis=-1)
        n_nl = int((r < cfg.r_cut).sum())
        if n <= 300:
            n_bf = brute_force_pair_count(pos, cell, cfg.r_cut, cfg.eps)
            topo_ok = (n_nl == n_bf)
            topo_txt = "ok" if topo_ok else f"BAD {n_nl}!={n_bf}"
        else:
            topo_ok, topo_txt = True, "skip"

        rng = np.random.default_rng(1234)
        w = torch.tensor(rng.normal(size=(n, soap_dim)), dtype=torch.float64, device=device)

        if n <= args.max_dense_atoms:
            f_ref, g_ref, c_ref = eval_fn(ref_fn, pos, cell, sym, w, device)
            f_new, g_new, c_new = eval_fn(nl.compute_per_center, pos, cell, sym, w, device)
            df = float(np.max(np.abs(f_new - f_ref)))
            dg = float(np.max(np.abs(g_new - g_ref)))
            dc = float(np.max(np.abs(c_new - c_ref)))
            same_ok = (df <= args.tol_feat) and (dg <= args.tol_grad) and (dc <= args.tol_grad)
            if orig is not None:
                f_emb, g_emb, c_emb = eval_fn(nl.compute_per_center_dense, pos, cell, sym, w, device)
                emb = float(np.max(np.abs(f_emb - f_ref)))
                same_ok = same_ok and (emb <= args.tol_feat)
        else:
            df = dg = dc = float("nan")
            same_ok = True

        # edge SET must agree between every available backend (i, j, rounded S)
        bk_txt = ""
        if n <= 600:
            sets = {}
            for bk in ("torch", "matscipy", "ase"):
                try:
                    a, b, c, _u = nl._neighbour_edges(torch.tensor(pos, device=device),
                                                      torch.tensor(cell, device=device), backend=bk)
                except ImportError:
                    continue
                a, b, c = a.cpu().numpy(), b.cpu().numpy(), c.cpu().numpy()
                dd = pos[b] - pos[a] + c @ cell
                m = np.linalg.norm(dd + cfg.eps, axis=-1) < cfg.r_cut
                sets[bk] = set(zip(a[m].tolist(), b[m].tolist(),
                                   map(tuple, np.rint(c[m]).astype(int).tolist())))
            ref_set = sets.get("torch")
            bad = [k for k, v in sets.items() if v != ref_set]
            bk_txt = "+".join(sets) + (" MISMATCH " + ",".join(bad) if bad else "")
            topo_ok = topo_ok and not bad
            topo_txt = topo_txt if not bad else topo_txt + "/XBK"
            if bk_txt:
                print(f"    backends compared: {bk_txt}")

        ok = topo_ok and same_ok
        all_ok &= ok
        results.append(dict(case=label, n=n, edges=n_nl, topo=topo_txt,
                            d_feat=df, d_force=dg, d_cell=dc, ok=bool(ok)))
        print(f"{label:44s} {n:5d} {n_nl:8d} {topo_txt:>5s} "
              f"{df:9.2e} {dg:9.2e} {dc:9.2e}  {'PASS' if ok else 'FAIL'}")

    scaling = []
    if args.scaling:
        print("\n=== 3: scaling (forward + backward, float64) ===")
        print(f"{'N':>6s} {'edges':>8s} {'t_dense[s]':>11s} {'t_nl[s]':>9s} {'speedup':>8s} "
              f"{'mem_dense[GiB]':>15s} {'mem_nl[GiB]':>12s}")
        for nn in args.sizes:
            pos, cell, sym = synth(nn, nn, nn, seed=100 + nn)
            n = len(pos)
            rng = np.random.default_rng(7)
            w = torch.tensor(rng.normal(size=(n, soap_dim)), dtype=torch.float64, device=device)
            i_, j_, S_, _ = nl._neighbour_edges(torch.tensor(pos, device=device),
                                                torch.tensor(cell, device=device))
            t_nl, m_nl = time_and_mem(nl.compute_per_center, pos, cell, sym, w, device, args.reps)
            if n <= args.max_dense_atoms:
                try:
                    t_d, m_d = time_and_mem(ref_fn, pos, cell, sym, w, device, args.reps)
                except RuntimeError as exc:  # OOM etc.
                    t_d, m_d = float("nan"), float("nan")
                    print(f"   dense failed at N={n}: {str(exc)[:80]}")
            else:
                t_d, m_d = float("nan"), float("nan")
            scaling.append(dict(n=n, edges=int(len(i_)), t_dense=t_d, t_nl=t_nl,
                                mem_dense=m_d, mem_nl=m_nl))
            sp = t_d / t_nl if np.isfinite(t_d) else float("nan")
            print(f"{n:6d} {len(i_):8d} {t_d:11.4f} {t_nl:9.4f} {sp:8.1f} "
                  f"{m_d:15.3f} {m_nl:12.3f}")
        ns = [s["n"] for s in scaling]
        print(f"\nfitted exponents (time ~ N^p):   dense p = "
              f"{fit_exponent(ns, [s['t_dense'] for s in scaling]):.2f}   "
              f"neighbour-list p = {fit_exponent(ns, [s['t_nl'] for s in scaling]):.2f}")
        if device.type == "cuda":
            print(f"fitted exponents (mem ~ N^p):    dense p = "
                  f"{fit_exponent(ns, [s['mem_dense'] for s in scaling]):.2f}   "
                  f"neighbour-list p = {fit_exponent(ns, [s['mem_nl'] for s in scaling]):.2f}")

    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    if args.json:
        args.json.write_text(json.dumps(dict(ok=bool(all_ok), cases=results,
                                             scaling=scaling), indent=2))
        print(f"saved {args.json}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
