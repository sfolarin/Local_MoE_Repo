#!/usr/bin/env python3
"""diag_fd.py -- diagnose the assembled-model finite-difference failure by
scanning h.  Roundoff-dominated error scales as 1/h; truncation error as h^2.
The slope decides which, and the minimum locates the best achievable accuracy.
Also measures the effective machine epsilon of the energy evaluation."""
import hashlib, importlib.util, inspect, os, sys, json
from pathlib import Path
import numpy as np, torch

def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

TR, CORR, STAGE = os.environ["TRAINER"], os.environ["CORR"], os.environ["STAGE"]
DEV = torch.device(os.environ.get("DEVICE", "cpu"))
t = load(TR, "_tr"); soap = t.import_torchsoap(Path(CORR))
desc = soap.TorchSOAPDescriptor().to(torch.float64).to(DEV).eval()
RCUT = float(desc.config.r_cut)
cm, cmean, cstd, e_ref, _ = t.load_common_checkpoint(Path(os.environ["COMMON"]), DEV)
rr = t.load_routing_checkpoint(Path(os.environ["ROUTER"]), DEV)
rck = torch.load(os.environ["RESIDUAL"], map_location="cpu", weights_only=False)
res = t.MLP(input_dim=int(rck["input_dim"]),
            hidden_dims=[int(x) for x in rck["residual_hidden_dims"]],
            output_dim=len(rck["expert_names"]), dropout=0.0).to(DEV)
res.load_state_dict(rck["residual_model_state_dict"]); res.eval()
for p in res.parameters(): p.requires_grad_(False)
pool = dict(record=None, descriptor=desc, common_model=cm, common_mean=cmean,
            common_std=cstd, routing_model=rr[0], routing_mean=rr[1], routing_std=rr[2],
            residual_model=res, residual_mean=cmean, residual_std=cstd, e_ref=e_ref)
fn = t.model_terms; sig = inspect.signature(fn)

def terms(rec, pos, cell, grad=False):
    p = torch.tensor(pos, dtype=torch.float64, device=DEV, requires_grad=grad)
    c = torch.tensor(cell, dtype=torch.float64, device=DEV)
    pool["record"] = rec
    out = fn(positions=p, cell=c, **{k: pool[k] for k in sig.parameters if k in pool})
    E = out["total_energy"]
    if not grad: return float(E.detach()), None, out
    return float(E.detach()), torch.autograd.grad(E, p)[0].detach().cpu().numpy(), out

recs = t.load_records(Path(STAGE))
rec = sorted(recs["train"], key=lambda r: r.n_atoms)[0]
pos0 = np.asarray(rec.positions_np, float); cell0 = np.asarray(rec.cell_np, float)
E0, G0, out = terms(rec, pos0, cell0, grad=True); F0 = -G0

print("=" * 92); print("FINITE-DIFFERENCE DIAGNOSIS: %s, %d atoms" % (rec.source_file, rec.n_atoms))
print("=" * 92)
print("E = %.12f eV" % E0)
print("dtypes returned by model_terms:")
for k, v in out.items():
    if isinstance(v, torch.Tensor):
        print("  %-24s %s %s" % (k, str(v.dtype).replace("torch.", ""), tuple(v.shape)))
print("descriptor dtype float64; the above shows where single precision enters.")

from ase import Atoms
from ase.neighborlist import neighbor_list
at = Atoms(positions=pos0, cell=cell0, pbc=True, symbols=rec.symbols)
d = neighbor_list("d", at, RCUT + 0.5)
gap = float(np.min(np.abs(d - RCUT)))
print("\nnearest pair to r_cut: %.4e A  (h must stay below 0.4*gap = %.3e)" % (gap, 0.4 * gap))

i, k = int(np.argmax(np.abs(F0).max(axis=1))), int(np.argmax(np.abs(F0[np.argmax(np.abs(F0).max(axis=1))])))
print("probing atom %d axis %d, analytic force %.9f eV/A" % (i, k, F0[i, k]))
print("\n%-12s %18s %14s %10s" % ("h", "central difference", "abs error", "err*h"))
rows = []
for h in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
    if h > 0.4 * gap:
        print("%-12.0e %18s" % (h, "skipped: straddles r_cut")); continue
    pp = pos0.copy(); pp[i, k] += h
    pm = pos0.copy(); pm[i, k] -= h
    fd = -(terms(rec, pp, cell0)[0] - terms(rec, pm, cell0)[0]) / (2 * h)
    err = abs(fd - F0[i, k])
    print("%-12.0e %18.9f %14.3e %10.2e" % (h, fd, err, err * h))
    rows.append((h, fd, err))
if len(rows) >= 2:
    hs = np.array([r[0] for r in rows]); es = np.array([r[2] for r in rows])
    good = es > 0
    slope = np.polyfit(np.log10(hs[good]), np.log10(es[good]), 1)[0]
    print("\nlog-log slope of error vs h : %+.2f" % slope)
    print("  -1 => roundoff dominated (precision limited, forces fine)")
    print("  +2 => truncation dominated (forces would be wrong if error stays large)")
    eps_eff = float(np.median(es[good] * hs[good]) / max(abs(E0), 1e-30))
    print("effective epsilon of the energy: %.3e   (float32 %.2e, float64 %.2e)"
          % (eps_eff, np.finfo(np.float32).eps, np.finfo(np.float64).eps))
    best = rows[int(np.argmin(es))]
    print("best achievable: err %.3e eV/A at h %.0e" % (best[2], best[0]))
    print("\nVERDICT: %s" % ("roundoff-limited; the force is consistent with the energy "
          "gradient to the precision the evaluation allows" if slope < -0.5 else
          "NOT roundoff-limited; investigate the forces"))
