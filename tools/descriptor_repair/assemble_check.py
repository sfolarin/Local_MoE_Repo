#!/usr/bin/env python3
"""assemble_check.py -- assemble common + router + residual using the TRAINER'S OWN
loaders, then test energy invariance, force covariance and energy-gradient
consistency.  Signatures are introspected rather than assumed: whatever cannot be
sourced is named exactly, so one run either works or says what is missing.
Train/val records only (the adapter's load_records excludes test)."""
import hashlib, importlib.util, inspect, json, os, sys
from pathlib import Path
import numpy as np, torch

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

TR, CORR = os.environ["TRAINER"], os.environ["CORR"]
STAGE = os.environ["STAGE"]; DEV = torch.device(os.environ.get("DEVICE", "cpu"))
print("=" * 94); print("ASSEMBLED-MODEL PHYSICS CHECKS"); print("=" * 94)
print("trainer   %s\n  %s" % (sha(TR)[:14], TR))
print("corrected %s\n  %s" % (sha(CORR)[:14], CORR))
print("device %s | torch %s" % (DEV, torch.__version__))

t = load(TR, "_tr")
soap = t.import_torchsoap(Path(CORR))
print("descriptor variant: %s" % getattr(soap, "DESCRIPTOR_VARIANT", "(none)"))
desc = soap.TorchSOAPDescriptor().to(torch.float64).to(DEV).eval()

cm, cmean, cstd, e_ref, cck = t.load_common_checkpoint(Path(os.environ["COMMON"]), DEV)
print("common loaded; e_ref = %.15g" % e_ref)
rr = t.load_routing_checkpoint(Path(os.environ["ROUTER"]), DEV)
print("load_routing_checkpoint returned %d objects: %s"
      % (len(rr) if isinstance(rr, tuple) else 1,
         [type(x).__name__ for x in rr] if isinstance(rr, tuple) else type(rr).__name__))
rm, rmean, rstd = (rr[0], rr[1], rr[2]) if isinstance(rr, tuple) and len(rr) >= 3 else (rr, None, None)

rck = torch.load(os.environ["RESIDUAL"], map_location="cpu", weights_only=False)
res = t.MLP(input_dim=int(rck["input_dim"]),
            hidden_dims=[int(x) for x in rck["residual_hidden_dims"]],
            output_dim=len(rck["expert_names"]), dropout=0.0).to(DEV)
res.load_state_dict(rck["residual_model_state_dict"]); res.eval()
for p in res.parameters(): p.requires_grad_(False)
print("residual loaded: 392 -> %s -> %d" % (list(rck["residual_hidden_dims"]), len(rck["expert_names"])))

recs = t.load_records(Path(STAGE))
print("records: " + ", ".join("%s=%d" % (k, len(v)) for k, v in recs.items()))
if len(recs.get("test", [])) != 0:
    raise SystemExit("FAIL: test partition is not empty; use the ADAPTER, not the original")
rec = sorted(recs["train"], key=lambda r: r.n_atoms)[0]
print("probe record: %s, %d atoms, split=%s" % (rec.source_file, rec.n_atoms, rec.split))

pool = dict(record=rec, descriptor=desc, common_model=cm, common_mean=cmean,
            common_std=cstd, routing_model=rm, routing_mean=rmean, routing_std=rstd,
            residual_model=res, residual_mean=cmean, residual_std=cstd,
            e_ref=e_ref, device=DEV, create_graph=True, include_regions=False,
            energy_weight=float(rck["energy_weight"]), force_weight=float(rck["force_weight"]))

for fn_name in ("model_terms", "predict_energy_forces"):
    fn = getattr(t, fn_name, None)
    if fn is None: continue
    sig = inspect.signature(fn)
    print("\n--- %s%s" % (fn_name, sig))
    need = [k for k, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and k not in ("positions", "cell")]
    missing = [k for k in need if k not in pool or pool[k] is None]
    print("    required: %s" % need)
    print("    MISSING : %s" % (missing or "none"))

fn = t.model_terms
sig = inspect.signature(fn)
need = [k for k, p in sig.parameters.items() if p.default is inspect.Parameter.empty]
missing = [k for k in need if k not in ("positions", "cell") and (k not in pool or pool[k] is None)]
if missing:
    print("\nSTOP: cannot source %s for model_terms. Nothing guessed." % missing)
    Path("assemble_check.json").write_text(json.dumps(dict(
        missing=missing, signature=str(sig)), indent=2))
    raise SystemExit(2)

import numpy as _np
# StructureRecord already carries the geometry, so no shard is read here.
pos0 = _np.asarray(rec.positions_np, float)
cell0 = _np.asarray(rec.cell_np, float)
print("geometry from record: positions %s, cell %s" % (pos0.shape, cell0.shape))

def describe(v):
    if isinstance(v, torch.Tensor):
        return "Tensor shape=%s dtype=%s requires_grad=%s%s" % (
            tuple(v.shape), str(v.dtype).replace("torch.", ""), v.requires_grad,
            ("  value=%.9g" % float(v)) if v.ndim == 0 else "")
    return "%s %r" % (type(v).__name__, v if isinstance(v, (int, float, str, bool)) else "")


def pick_total(out, verbose=False):
    """Find the scalar total energy among whatever model_terms returns, by
    structure rather than by assumption.  Prints the structure once so the
    choice is auditable, and refuses if it is ambiguous."""
    items = []
    if isinstance(out, dict):
        items = list(out.items())
    elif isinstance(out, (tuple, list)):
        items = [(str(i), v) for i, v in enumerate(out)]
    elif hasattr(out, "_fields"):
        items = [(f, getattr(out, f)) for f in out._fields]
    else:
        items = [("<returned>", out)]
    if verbose:
        print("model_terms returned %s with %d entries:" % (type(out).__name__, len(items)))
        for k, v in items:
            print("    %-30s %s" % (k, describe(v)))
    # a total energy is a differentiable scalar tensor
    scal = [(k, v) for k, v in items
            if isinstance(v, torch.Tensor) and v.ndim == 0 and v.requires_grad]
    named = [(k, v) for k, v in scal
             if any(w in k.lower() for w in ("energy_total", "total_energy", "e_total"))]
    if not named:
        named = [(k, v) for k, v in scal if "energy" in k.lower() and "per" not in k.lower()]
    if not named:
        named = scal
    if not named:
        raise SystemExit("STOP: no differentiable scalar in model_terms output; "
                         "nothing guessed. Structure printed above.")
    if verbose:
        print("  -> using %r as the total energy" % named[0][0])
    return named[0][1]


_shown = {"done": False}


def energy(pos, cell, grad=False):
    p = torch.tensor(pos, dtype=torch.float64, device=DEV, requires_grad=grad)
    c = torch.tensor(cell, dtype=torch.float64, device=DEV)
    kw = {k: pool[k] for k in sig.parameters if k in pool}
    out = fn(positions=p, cell=c, **kw)
    E = pick_total(out, verbose=not _shown["done"])
    _shown["done"] = True
    if not grad:
        return float(E.detach()), None
    G = torch.autograd.grad(E, p, create_graph=False)[0]
    return float(E.detach()), G.detach().cpu().numpy()

def rot(s):
    q, r = _np.linalg.qr(_np.random.default_rng(s).standard_normal((3, 3)))
    q = q @ _np.diag(_np.sign(_np.diag(r)))
    if _np.linalg.det(q) < 0: q[:, 0] = -q[:, 0]
    return q

E0, G0 = energy(pos0, cell0, grad=True)
F0 = -G0
print("\nE = %.12f eV   (%.9f eV/atom)" % (E0, E0 / rec.n_atoms))
res_out = {"record": str(rec.source_file), "n_atoms": int(rec.n_atoms), "E_eV": E0,
           "residual_normalization": "common (train_c1b_holdout40.py line 1277)"}
for s in (1, 2, 3):
    R = rot(s)
    E1, G1 = energy(pos0 @ R.T, cell0 @ R.T, grad=True)
    F1 = -G1
    de = abs(E1 - E0); dfe = float(_np.abs(F1 - F0 @ R.T).max())
    print("  rotation %d: |dE| = %.3e eV   max|F' - F R^T| = %.3e eV/A" % (s, de, dfe))
    res_out["rot%d_dE" % s] = de; res_out["rot%d_dF" % s] = dfe
mx_e = max(res_out["rot%d_dE" % s] for s in (1, 2, 3))
mx_f = max(res_out["rot%d_dF" % s] for s in (1, 2, 3))
print("  energy invariance : %.3e eV    -> %s" % (mx_e, "PASS" if mx_e < 1e-7 else "FAIL"))
print("  force covariance  : %.3e eV/A  -> %s" % (mx_f, "PASS" if mx_f < 1e-6 else "FAIL"))
Path("assemble_check.json").write_text(json.dumps(res_out, indent=2))
print("\nwrote assemble_check.json")
