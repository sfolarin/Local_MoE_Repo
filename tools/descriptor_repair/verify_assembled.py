#!/usr/bin/env python3
"""
verify_assembled.py -- physics verification of the assembled Local-MoE model,
with a nonzero exit status on any failure.

Supersedes assemble_check.py, which printed PASS/FAIL but always exited 0 -- in a
batch job that is indistinguishable from success.

Checks, per record:
  1. finiteness of the energy, every returned term, the routing weights and the
     autograd forces;
  2. energy invariance under proper rotation;
  3. force covariance  F(Rx) == F(x) R^T;
  4. energy-gradient consistency: a CUTOFF-AWARE central difference of the
     ASSEMBLED energy against its autograd forces.  weights.requires_grad only
     shows graph participation; this is what verifies the forces are the gradient
     of the energy that is reported.

The finite-difference step is screened against the nearest pair-to-cutoff
distance over ALL periodic images, because a stencil straddling r_cut is invalid
(the cosine cutoff is C1 but not C2, so the h^2 expansion fails and the order
estimate collapses).  A centre with no valid step is marked EXEMPT and counted,
never silently skipped.  The tolerance tracks the roundoff floor eps*|E|/h rather
than being a flat constant.

Exit: 0 all checks passed; 1 a tolerance or finiteness check failed;
      2 assembly could not be completed (nothing guessed).

  TRAINER=<c1b adapter> CORR=<corrected module> COMMON=.. ROUTER=.. RESIDUAL=..
  STAGE=<stage root> [DEVICE=cpu|cuda] [NREC=2] [NPROBE=3] python3 verify_assembled.py
"""
import hashlib, importlib.util, inspect, json, os, sys
from pathlib import Path
import numpy as np
import torch

TOL_E = float(os.environ.get("TOL_E", "1e-7"))      # eV, rotation invariance
TOL_F = float(os.environ.get("TOL_F", "1e-6"))      # eV/A, force covariance
NREC = int(os.environ.get("NREC", "2"))
NPROBE = int(os.environ.get("NPROBE", "3"))
DEV = torch.device(os.environ.get("DEVICE", "cpu"))
if DEV.type == "cuda" and not torch.cuda.is_available():
    raise SystemExit("FAIL: cuda requested but torch.cuda.is_available() is False")

fails, exempts, results = [], 0, []


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    sys.modules[n] = m
    s.loader.exec_module(m)
    return m


def rot(seed):
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    assert abs(np.linalg.det(q) - 1.0) < 1e-12
    return q


def gap_all_images(pos, cell, rcut, syms):
    """Nearest |distance - r_cut| over ALL periodic images (not minimum image)."""
    try:
        from ase import Atoms
        from ase.neighborlist import neighbor_list
    except ImportError:
        return None
    at = Atoms(positions=np.asarray(pos), cell=np.asarray(cell), pbc=True, symbols=syms)
    d = neighbor_list("d", at, float(rcut) + 0.5)
    return None if len(d) == 0 else float(np.min(np.abs(np.asarray(d) - float(rcut))))


# ---------------------------------------------------------------- assembly
TR, CORR, STAGE = os.environ["TRAINER"], os.environ["CORR"], os.environ["STAGE"]
print("=" * 96)
print("ASSEMBLED-MODEL PHYSICS VERIFICATION (nonzero exit on failure)")
print("=" * 96)
print("trainer   %s  %s" % (sha(TR)[:14], TR))
print("corrected %s  %s" % (sha(CORR)[:14], CORR))
print("device %s | torch %s | numpy %s | python %s"
      % (DEV, torch.__version__, np.__version__, sys.version.split()[0]))

t = load(TR, "_tr")
soap = t.import_torchsoap(Path(CORR))
variant = getattr(soap, "DESCRIPTOR_VARIANT", None)
print("descriptor variant: %r" % variant)
if variant != "rotinv_v1":
    raise SystemExit("FAIL(2): descriptor is not the corrected module (variant %r)" % variant)
desc = soap.TorchSOAPDescriptor().to(torch.float64).to(DEV).eval()
RCUT = float(desc.config.r_cut)

cm, cmean, cstd, e_ref, cck = t.load_common_checkpoint(Path(os.environ["COMMON"]), DEV)
rr = t.load_routing_checkpoint(Path(os.environ["ROUTER"]), DEV)
rm, rmean, rstd = rr[0], rr[1], rr[2]
rck = torch.load(os.environ["RESIDUAL"], map_location="cpu", weights_only=False)
res = t.MLP(input_dim=int(rck["input_dim"]),
            hidden_dims=[int(x) for x in rck["residual_hidden_dims"]],
            output_dim=len(rck["expert_names"]), dropout=0.0).to(DEV)
res.load_state_dict(rck["residual_model_state_dict"])
res.eval()
for p in res.parameters():
    p.requires_grad_(False)
print("e_ref = %.15g   r_cut = %.3f A" % (e_ref, RCUT))
print("checkpoint hashes: common %s  router %s  residual %s"
      % (sha(os.environ["COMMON"])[:14], sha(os.environ["ROUTER"])[:14],
         sha(os.environ["RESIDUAL"])[:14]))
# residual normalisation IS the common one: train_c1b_holdout40.py line 1277
pool = dict(record=None, descriptor=desc, common_model=cm, common_mean=cmean,
            common_std=cstd, routing_model=rm, routing_mean=rmean, routing_std=rstd,
            residual_model=res, residual_mean=cmean, residual_std=cstd, e_ref=e_ref)
print("residual normalisation = common (train_c1b_holdout40.py line 1277)")

fn = t.model_terms
sig = inspect.signature(fn)
need = [k for k, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty and k not in ("positions", "cell")]
missing = [k for k in need if pool.get(k, "MISSING") == "MISSING" or
           (k != "record" and pool.get(k) is None)]
if missing:
    raise SystemExit("FAIL(2): cannot source %s for model_terms; nothing guessed" % missing)

recs = t.load_records(Path(STAGE))
print("records: " + ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(recs.items())))
if len(recs.get("test", [])) != 0:
    raise SystemExit("FAIL(2): test partition is not empty (%d); use the ADAPTER"
                     % len(recs["test"]))


def terms(rec, pos, cell, grad=False):
    p = torch.tensor(pos, dtype=torch.float64, device=DEV, requires_grad=grad)
    c = torch.tensor(cell, dtype=torch.float64, device=DEV)
    pool["record"] = rec
    out = fn(positions=p, cell=c, **{k: pool[k] for k in sig.parameters if k in pool})
    E = out["total_energy"]
    if not grad:
        return float(E.detach()), None, out
    G = torch.autograd.grad(E, p, create_graph=False)[0]
    return float(E.detach()), G.detach().cpu().numpy(), out


# ---------------------------------------------------------------- per record
chosen = sorted(recs["train"], key=lambda r: r.n_atoms)[:max(1, NREC - 1)]
if recs.get("val"):
    chosen = chosen + sorted(recs["val"], key=lambda r: r.n_atoms)[:1]
for rec in chosen:
    pos0 = np.asarray(rec.positions_np, float)
    cell0 = np.asarray(rec.cell_np, float)
    E0, G0, out = terms(rec, pos0, cell0, grad=True)
    F0 = -G0
    R = dict(record=str(rec.source_file), split=rec.split, n_atoms=int(rec.n_atoms),
             E_eV=E0, E_per_atom=E0 / rec.n_atoms)
    print("\n%s  (%s, %d atoms)" % (rec.source_file, rec.split, rec.n_atoms))
    print("  E = %.12f eV   (%.9f eV/atom)" % (E0, E0 / rec.n_atoms))

    # 1. finiteness
    bad = []
    if not np.isfinite(E0):
        bad.append("total_energy")
    if not np.isfinite(F0).all():
        bad.append("forces")
    for k, v in (out.items() if isinstance(out, dict) else []):
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
            bad.append(k)
    R["nonfinite"] = bad
    print("  finiteness: %s" % ("all finite" if not bad else "NONFINITE in %s" % bad))
    if bad:
        fails.append("%s nonfinite %s" % (rec.source_file, bad))

    # 2 & 3. invariance and covariance
    de = dfe = 0.0
    for s in (1, 2, 3):
        Rm = rot(s)
        E1, G1, _ = terms(rec, pos0 @ Rm.T, cell0 @ Rm.T, grad=True)
        de = max(de, abs(E1 - E0))
        dfe = max(dfe, float(np.abs((-G1) - F0 @ Rm.T).max()))
    R["rot_dE_eV"], R["rot_dF_eV_A"] = de, dfe
    ok_e, ok_f = de <= TOL_E, dfe <= TOL_F
    print("  energy invariance : %.3e eV    tol %.0e  -> %s" % (de, TOL_E, "pass" if ok_e else "FAIL"))
    print("  force covariance  : %.3e eV/A  tol %.0e  -> %s" % (dfe, TOL_F, "pass" if ok_f else "FAIL"))
    if not ok_e:
        fails.append("%s energy invariance %.3e > %.0e" % (rec.source_file, de, TOL_E))
    if not ok_f:
        fails.append("%s force covariance %.3e > %.0e" % (rec.source_file, dfe, TOL_F))

    # 4. cutoff-aware finite difference of the ASSEMBLED energy vs its forces
    gap = gap_all_images(pos0, cell0, RCUT, rec.symbols)
    # Roundoff dominates here (error ~ 1/h), so the step should be as LARGE as
    # validity allows rather than capped at an arbitrary 1e-3.  The binding
    # constraint is the cutoff gap; 5e-3 A is a loose ceiling that keeps
    # truncation small.  The best-of-three probe below also samples h/3 and
    # h/10, which covers the truncation-dominated side.
    h1 = 5e-3 if gap is None else min(5e-3, 0.4 * gap)
    R["cutoff_gap_A"], R["fd_step"] = gap, h1
    print("  nearest pair to r_cut: %s A   FD step %.3e"
          % (("%.3e" % gap) if gap is not None else "n/a", h1))
    if h1 < 1e-5:
        exempts += 1
        R["fd"] = "EXEMPT"
        R["gradient_consistency_verified"] = False
        print("  gradient consistency: EXEMPT (no valid central-difference step)")
    else:
        # The energy is float64 but the networks evaluate in float32 (weights and
        # centered_delta come back float32), so the float32-limited quantity is
        # the RESIDUAL part: N*e_ref is exact float64 arithmetic and carries no
        # network noise.  diag_fd.py measured the effective epsilon of the energy
        # at 1.5e-09 with a ~42 eV residual against a ~654 eV total, consistent
        # with eps32 acting on the residual scale.  Building the floor from the
        # total energy with float64 epsilon -- as the first version did -- was
        # wrong in both the epsilon and the scale, by about nine orders.
        resid = sum(abs(float(out[k].detach())) for k in
                    ("common_residual_total", "moe_residual_total") if k in out)
        eps32 = float(np.finfo(np.float32).eps)
        floor = eps32 * max(resid, 1.0) / h1
        thr = max(1e-4, 20.0 * floor)
        fmax = float(np.abs(F0).max())
        R.update(fd_threshold=thr, fd_roundoff_floor=floor, fd_force_max=fmax,
                 fd_residual_scale_eV=resid)
        # A threshold that cannot resolve a 5% force error carries no information.
        # Saying so is honest; returning a green pass from a powerless test is not.
        if thr > 0.05 * max(fmax, 1e-12):
            exempts += 1
            R["fd"] = "EXEMPT_resolution"
            R["gradient_consistency_verified"] = False
            print("  gradient consistency: EXEMPT -- threshold %.3e eV/A exceeds 5%% of "
                  "max|F|=%.3e" % (thr, fmax))
            print("      h is capped at %.2e by the %.3e A cutoff gap; residual scale "
                  "%.1f eV" % (h1, gap if gap is not None else float("nan"), resid))
        else:
            rng = np.random.default_rng(5)
            worst, detail = 0.0, []
            for i in rng.choice(len(pos0), size=min(NPROBE, len(pos0)), replace=False):
                for k in range(3):
                    # Roundoff-dominated error falls as 1/h, so Richardson
                    # extrapolation amplifies noise instead of removing it.
                    # Take the best of several valid steps instead.
                    errs = []
                    for h in (h1, h1 / 3.0, h1 / 10.0):
                        pp = pos0.copy(); pp[i, k] += h
                        pm = pos0.copy(); pm[i, k] -= h
                        fd = -(terms(rec, pp, cell0)[0] - terms(rec, pm, cell0)[0]) / (2 * h)
                        errs.append(abs(fd - F0[i, k]))
                    e = min(errs)
                    detail.append(dict(atom=int(i), axis=int(k), best_abs_error=e))
                    worst = max(worst, e)
            ok = worst <= thr
            R.update(fd_max_abs_error=worst, fd="pass" if ok else "FAIL", fd_probes=detail,
                     gradient_consistency_verified=bool(ok))
            print("  gradient consistency: %.3e eV/A  tol %.3e (floor %.2e, residual "
                  "%.1f eV)  -> %s" % (worst, thr, floor, resid, "pass" if ok else "FAIL"))
            if not ok:
                fails.append("%s gradient consistency %.3e > %.3e"
                             % (rec.source_file, worst, thr))
    results.append(R)

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 96)
unverified = [r["record"] for r in results if not r.get("gradient_consistency_verified")]
print("records checked %d | failures %d" % (len(results), len(fails)))
print("gradient consistency VERIFIED   on %d record(s)"
      % sum(1 for r in results if r.get("gradient_consistency_verified")))
print("gradient consistency NOT VERIFIED on %d record(s): %s"
      % (len(unverified), unverified or "none"))
if unverified:
    print("  An exemption is NOT a pass.  Those records' forces are unverified by")
    print("  finite difference; rotation invariance and covariance still hold for them.")
for f in fails:
    print("  FAIL: %s" % f)
print("NOTE: with checkpoints trained on UNCORRECTED features these energies are")
print("      interface validation only, never an accuracy result.")
Path(os.environ.get("JSON", "verify_assembled.json")).write_text(json.dumps(dict(
    trainer_sha256=sha(TR), corrected_sha256=sha(CORR), variant=variant,
    common_sha256=sha(os.environ["COMMON"]), router_sha256=sha(os.environ["ROUTER"]),
    residual_sha256=sha(os.environ["RESIDUAL"]), e_ref=e_ref, r_cut=RCUT,
    device=str(DEV), torch=torch.__version__, numpy=np.__version__,
    python=sys.version.split()[0], tol_energy=TOL_E, tol_force=TOL_F,
    residual_normalization="common (train_c1b_holdout40.py line 1277)",
    records=results, exempt_centres=exempts, failures=fails,
    gradient_consistency_verified_records=[
        r["record"] for r in results if r.get("gradient_consistency_verified")],
    gradient_consistency_unverified_records=unverified,
    verdict=("FAIL" if fails else
             ("PASS" if not unverified else "PASS_WITH_UNVERIFIED_GRADIENTS"))), indent=2))
print("wrote %s" % os.environ.get("JSON", "verify_assembled.json"))
verdict = ("FAIL" if fails else
           ("PASS" if not unverified else "PASS_WITH_UNVERIFIED_GRADIENTS"))
print("RESULT: %s%s" % (verdict, (" (%d failures)" % len(fails)) if fails else ""))
# 0 = everything verified; 1 = a check failed; 3 = passed but some gradients
# could not be verified, so the caller must not treat this as a full pass.
raise SystemExit(1 if fails else (0 if not unverified else 3))
