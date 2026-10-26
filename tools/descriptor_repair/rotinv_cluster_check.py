#!/usr/bin/env python3
"""
rotinv_cluster_check.py -- self-contained rotational-invariance check on GENUINE
training/validation configurations.  One file, no archive, no extraction.

WHAT THIS IS, PRECISELY
  It imports the PRODUCTION descriptor module read-only and defines a SUBCLASS
  that applies the corrected angular treatment:
    C1  homogeneous angular basis (restores the r^2 terms of the real solid
        harmonics, which the production code had specialised to r^2 = 1)
    C2  exact per-m weights w_lm = 1/<B_lm^2>, derived in exact rational
        arithmetic; every weight is a dyadic rational, so exact in float32/64
    C4  isotropic, strictly positive unit-vector denominator
  C3 (replacing torch.linalg.norm(d + eps) with the isotropic
  sqrt(|d|^2 + eps^2) at the three sites inside compute_per_center) is NOT
  applied here, because those are long methods this subclass does not override.
  C3 IS present in the shipped corrected module
  (torchsoap_autograd_module_periodic_regnorm_rotinv_v1.py, sha256
  0afd9a1b...), which remains the deliverable.  Its absence here leaves a
  residual O(eps/r) ~ 1e-12 relative anisotropy in the radial/cutoff factors,
  so expect ~1e-12 rather than ~1e-15 below -- still ~100x inside tolerance.
  This script is a cluster-environment check on real structures, not a
  replacement for that module.

  Nothing is written to the production tree.  No checkpoint is loaded.  The
  energy model is a fixed-seed random MLP ("diagnostic"): it is not physical
  and says nothing about accuracy.

TRAIN/VALIDATION ONLY -- three independent guards
  1. the manifest is read first and filtered to train/val before any structure
     file is opened;
  2. a positive allow-list of the exact paths named by those rows; anything
     outside it is refused.  This is the guard that matters, because this
     project's stage keeps every split in ONE flat geometry/ directory, so
     filenames carry no split marker;
  3. each shard's own "split" field must agree, catching a manifest that
     mislabels a test row.
  Label keys (energy/forces/...) are never read.  The run fails with exit code
  1 if the requested real structures cannot be loaded.
"""
import argparse, csv, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

# C2: exact weights w_lm = 1/<B_lm^2>_sphere, in the module's own m order.
W = ((1.0,), (3.0,) * 3, (15.0, 15.0, 5.0, 15.0, 15.0),
     (35.0/8, 105.0, 21.0/8, 7.0/4, 21.0/8, 105.0/4, 35.0/8),
     (315.0/4, 315.0/8, 45.0/4, 45.0/8, 9.0/64, 45.0/8, 45.0/16, 315.0/8, 315.0/64))
TOL = {"desc64": 1e-10, "e64": 1e-9, "f64": 1e-9, "s64": 1e-7,
       "desc32": 1e-4, "e32": 1e-2, "f32": 1e-4}
GPA = 160.21766208


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def load_production(path):
    spec = importlib.util.spec_from_file_location("_prod_torchsoap", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_prod_torchsoap"] = m          # register BEFORE exec, for dataclasses
    spec.loader.exec_module(m)
    return m


def make_corrected_class(prod):
    base = prod.TorchSOAPDescriptor

    class Corrected(base):
        """Production descriptor with C1 + C2 + C4 applied.  Production class untouched."""
        VARIANT = "rotinv_subclass_C1C2C4"

        def _w(self, l, dtype, device):
            return torch.tensor(W[l], dtype=dtype, device=device)

        def _real_sph_harm_basis(self, dcart, r):
            cfg = self.config
            eps = float(cfg.eps)
            # C4: isotropic, strictly positive denominator (|u| = 1 to machine
            # precision).  Recomputed from dcart so the passed-in anisotropic r
            # does not enter the ANGULAR part at all.
            r_iso = torch.sqrt((dcart * dcart).sum(-1) + eps * eps)
            x = dcart[..., 0] / r_iso
            y = dcart[..., 1] / r_iso
            z = dcart[..., 2] / r_iso
            one = torch.ones_like(x)
            # C1: homogeneous basis.  q = |u|^2 restores the r^2 terms, making
            # every channel homogeneous of degree l and harmonic.
            q = x * x + y * y + z * z
            out = [one[..., None]]
            if cfg.l_max >= 1:
                out.append(torch.stack([x, y, z], -1))
            if cfg.l_max >= 2:
                out.append(torch.stack([x*y, y*z, 0.5*(3*z*z - q), x*z,
                                        0.5*(x*x - y*y)], -1))
            if cfg.l_max >= 3:
                out.append(torch.stack([y*(3*x*x - y*y), x*y*z, y*(5*z*z - q),
                                        z*(5*z*z - 3*q), x*(5*z*z - q),
                                        z*(x*x - y*y), x*(x*x - 3*y*y)], -1))
            if cfg.l_max >= 4:
                out.append(torch.stack([x*y*(x*x - y*y), y*z*(3*x*x - y*y),
                                        x*y*(7*z*z - q), y*z*(7*z*z - 3*q),
                                        35*z**4 - 30*z*z*q + 3*q*q,
                                        x*z*(7*z*z - 3*q), (x*x - y*y)*(7*z*z - q),
                                        x*z*(x*x - 3*y*y),
                                        x**4 - 6*x*x*y*y + y**4], -1))
            return out

        def _power_spectrum(self, coeff_by_l):
            cfg = self.config
            parts = []
            for l, c in enumerate(coeff_by_l):
                ci, cj = c[:, self.pair_i, :], c[:, self.pair_j, :]
                # C2: w_lm * <B_lm^2> is then constant over m, which is exactly
                # the condition for sum_m w c c to depend on direction only
                # through u.v.  Equal weights already satisfy it for l<=1 and
                # fail for l>=2.
                parts.append(torch.sum(self._w(l, ci.dtype, ci.device) * ci * cj, -1))
            d = torch.cat(parts, -1)
            if cfg.normalize_power:
                n = torch.linalg.norm(d, dim=-1, keepdim=True)
                d = d / torch.sqrt(n * n + d.new_tensor(float(cfg.power_norm_lambda)) ** 2)
            return d
    return Corrected


# ------------------------------------------------------- geometry, train/val --
LABEL = ("energy", "energies", "force", "forces", "stress", "virial", "label",
         "target", "reference", "dft", "matpes", "r2scan", "expert", "teacher")


def norm_split(v):
    v = str(v).strip().lower()
    return {"train": "train", "training": "train", "val": "val",
            "validation": "val", "valid": "val", "test": "test"}.get(v)


def geom_hash(z, p, c):
    h = hashlib.sha256()
    h.update(np.asarray(z, dtype=np.int64).tobytes())
    h.update(np.round(np.asarray(p, float), 8).tobytes())
    h.update(np.round(np.asarray(c, float), 8).tobytes())
    return h.hexdigest()


def load_real(stage_root, per_split, max_atoms, seed, splits=("train", "val")):
    from ase import Atoms
    stage = Path(stage_root)
    man = stage / "manifest.csv"
    if not man.exists():
        raise SystemExit("FAIL: no manifest.csv under %s" % stage)
    rows = list(csv.DictReader(open(man, newline="")))
    cols = list(rows[0].keys()) if rows else []
    sc = next((c for c in cols if sum(norm_split(r.get(c, "")) is not None
                                      for r in rows[:300]) > len(rows[:300]) // 2), None)
    ic = next((c for c in ("source_file", "record_id", "id") if c in cols), None)
    pc = next((c for c in ("output_file", "geometry_file", "data_file") if c in cols), None)
    if not (sc and ic):
        raise SystemExit("FAIL: could not identify split/id columns in %s; columns=%s"
                         % (man, cols))
    by = {}
    for r in rows:
        s = norm_split(r.get(sc, ""))
        if s:
            by.setdefault(s, []).append(r)
    counts = {k: len(v) for k, v in sorted(by.items())}

    # GUARD 1 + 2: filter to train/val, then build the allow-list, before any open
    allowed, sel = set(), []
    for s in splits:
        for r in by.get(s, []):
            sel.append((s, r))
            if pc and r.get(pc):
                q = Path(str(r[pc]))
                q = q if q.is_absolute() else man.parent / q
                if q.exists():
                    allowed.add(q.resolve())
    print("  manifest        : %s" % man)
    print("  columns         : split=%r id=%r path=%r" % (sc, ic, pc))
    print("  manifest counts : %s" % counts)
    print("  allow-list      : %d files from train/val rows; %d test rows excluded "
          "before any file was opened" % (len(allowed), counts.get("test", 0)))
    print("  guards          : (1) manifest filter (2) path allow-list "
          "(3) each shard's own split field")
    print("  requested       : %d PER SPLIT (not a total) -> %d in all"
          % (per_split, per_split * len(splits)))

    rng = np.random.default_rng(seed)
    out, recs = [], []
    for s in splits:
        pool = [r for ss, r in sel if ss == s]
        got, errs = 0, []
        for k in rng.permutation(len(pool)):
            if got >= per_split:
                break
            r = pool[int(k)]
            rid = str(r.get(ic))
            try:
                q = Path(str(r[pc])) if pc and r.get(pc) else None
                q = (q if q and q.is_absolute() else man.parent / q) if q else None
                if q is None or not q.exists():
                    errs.append("%s: no shard path" % rid); continue
                if allowed and q.resolve() not in allowed:
                    errs.append("%s: not in allow-list" % rid); continue
                with np.load(q, allow_pickle=False) as f:
                    keys = set(f.files)
                    kz = next((k for k in ("atomic_numbers", "numbers", "Z") if k in keys), None)
                    kp = next((k for k in ("positions", "positions_np", "pos") if k in keys), None)
                    kc = next((k for k in ("cell", "cell_np", "lattice") if k in keys), None)
                    if not (kz and kp and kc):
                        errs.append("%s: geometry keys missing; present=%s"
                                    % (rid, sorted(keys))); continue
                    for k in (kz, kp, kc):
                        if any(w in k.lower() for w in LABEL):
                            raise SystemExit("FAIL: %r looks like a label key" % k)
                    z = f[kz].astype(int).ravel()
                    pos = np.asarray(f[kp], float).reshape(-1, 3)
                    cell = np.asarray(f[kc], float).reshape(3, 3)
                    # GUARD 3: the shard's own split must agree
                    if "split" in keys:
                        ds = norm_split(np.asarray(f["split"]).ravel()[0])
                        if ds == "test":
                            raise SystemExit("FAIL: %s declares split=test but the "
                                             "manifest listed it as %s" % (rid, s))
                        if ds != s:
                            raise SystemExit("FAIL: split disagreement for %s: "
                                             "manifest %s, shard %s" % (rid, s, ds))
                    ish = str(np.asarray(f["geometry_sha256"]).ravel()[0]) \
                        if "geometry_sha256" in keys else ""
            except SystemExit:
                raise
            except Exception as e:
                errs.append("%s: %s" % (rid, e)); continue
            if len(pos) != len(z):
                errs.append("%s: %d pos vs %d Z" % (rid, len(pos), len(z))); continue
            if len(z) > max_atoms:
                continue
            mh = str(r.get("geometry_sha256", "") or "")
            if mh and ish and mh != ish:
                raise SystemExit("FAIL: geometry hash disagreement for %s" % rid)
            at = Atoms(numbers=z, positions=pos, cell=cell, pbc=True)
            out.append(("REAL %s %s N=%d" % (s, rid, len(at)), at))
            recs.append(dict(record_id=rid, split=s, n_atoms=int(len(at)),
                             formula=at.get_chemical_formula(),
                             computed_geometry_sha256=geom_hash(z, pos, cell),
                             manifest_geometry_sha256=mh or None,
                             shard_declared_split=s, shard=str(q),
                             cellpar=[round(float(v), 3) for v in at.get_cell().cellpar()]))
            got += 1
        if got < per_split:
            print("  LOAD FAILED for split %r: asked %d, got %d. First errors: %s"
                  % (s, per_split, got, errs[:3] or ["none; the remainder exceeded --max-atoms"]))
            raise SystemExit(1)
    print("  loaded per split: %s (total %d)"
          % ({s: sum(1 for r in recs if r["split"] == s) for s in splits}, len(recs)))
    print("  %-22s %-6s %6s  %-16s %-18s %s"
          % ("record id", "split", "atoms", "formula", "computed geom sha", "manifest geom sha"))
    for r in recs:
        print("  %-22s %-6s %6d  %-16s %-18s %s"
              % (r["record_id"][:22], r["split"], r["n_atoms"], r["formula"][:16],
                 r["computed_geometry_sha256"][:16], (r["manifest_geometry_sha256"] or "-")[:18]))
    return out, dict(manifest=str(man), split_column=sc, id_column=ic, path_column=pc,
                     manifest_counts=counts, allow_list_size=len(allowed),
                     test_rows_excluded=counts.get("test", 0),
                     per_split_requested=per_split, records=recs, n_loaded=len(recs))


# ------------------------------------------------------------------ physics --
class Diag(torch.nn.Module):
    """Fixed-seed random MLP. NOT trained, NOT physical. Only to make a scalar."""
    def __init__(self, dim, seed=12345, dtype=torch.float64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                       torch.nn.Linear(64, 1))
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.35, 0.35, generator=g))
        self.to(dtype)

    def forward(self, pc):
        return self.net(pc).sum()


def rand_rot(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def axis_rot(ax, deg):
    t = np.deg2rad(deg); u = np.asarray(ax, float); u /= np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def efs(desc, model, atoms, dtype, dev, net_dtype=None, stress=True):
    p0 = torch.tensor(atoms.get_positions(), dtype=dtype, device=dev, requires_grad=True)
    c0 = torch.tensor(atoms.get_cell()[:], dtype=dtype, device=dev)
    e = torch.zeros((3, 3), dtype=dtype, device=dev, requires_grad=True)
    D = torch.eye(3, dtype=dtype, device=dev) + 0.5 * (e + e.T)
    pc = desc.compute_per_center(p0 @ D, c0 @ D, atoms.get_chemical_symbols())
    if net_dtype is not None:
        pc = pc.to(net_dtype)
    en = model(pc).to(dtype)
    gs = torch.autograd.grad(en, [p0, e] if stress else [p0])
    s = None
    if stress:
        g = gs[1].detach().cpu().numpy(); g = 0.5 * (g + g.T)
        s = g / float(torch.det(c0).abs()) * GPA
    return float(en.detach()), -gs[0].detach().cpu().numpy(), s


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--stage-root", required=True)
    a.add_argument("--module", required=True, help="production descriptor module (read only)")
    a.add_argument("--per-split", type=int, default=3, help="PER SPLIT, not a total")
    a.add_argument("--max-atoms", type=int, default=400)
    a.add_argument("--nrot", type=int, default=12)
    a.add_argument("--seed", type=int, default=20261025)
    a.add_argument("--device", default="cpu")
    a.add_argument("--json", default=None)
    args = a.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to cpu"); dev = torch.device("cpu")
    print("=" * 112)
    print("ROTATIONAL-INVARIANCE CHECK ON GENUINE TRAIN/VAL CONFIGURATIONS")
    print("  descriptor + DIAGNOSTIC random MLP only; no checkpoint is loaded")
    print("  corrected via SUBCLASS: C1 homogeneous basis, C2 exact weights, C4 isotropic")
    print("  C3 (isotropic distance inside compute_per_center) NOT active here ->")
    print("  expect a ~1e-12 floor rather than ~1e-15; C3 is in the shipped module")
    print("=" * 112)
    print("production module : %s" % args.module)
    print("  sha256          : %s" % sha(args.module))
    print("this script sha256: %s" % sha(__file__))
    print("torch %s | numpy %s | python %s | device %s"
          % (torch.__version__, np.__version__, sys.version.split()[0], dev))
    print("seed %d | %d random rotations + 6 named" % (args.seed, args.nrot))
    print("tolerances: " + ", ".join("%s<%.0e" % (k, v) for k, v in TOL.items()))

    prod = load_production(args.module)
    C = make_corrected_class(prod)
    d64 = C().to(torch.float64).to(dev).eval()
    d32 = C().to(torch.float32).to(dev).eval()
    o64 = prod.TorchSOAPDescriptor().to(torch.float64).to(dev).eval()
    print("per-centre dim %d | global dim %d | variant %s"
          % (d64.soap_dim, d64.output_dim, C.VARIANT))
    for l in range(d64.config.l_max + 1):
        print("  w(l=%d) = %s" % (l, list(W[l])))
    m64 = Diag(d64.soap_dim, args.seed, torch.float64).to(dev).eval()
    m32 = Diag(d64.soap_dim, args.seed, torch.float32).to(dev).eval()

    print("\n" + "=" * 112)
    print("REAL STRUCTURES (train/val only; the test partition is never opened)")
    print("=" * 112)
    cells, rep = load_real(args.stage_root, args.per_split, args.max_atoms, args.seed)

    rng = np.random.default_rng(args.seed)
    rots = [("random%02d" % i, rand_rot(rng)) for i in range(args.nrot)] + \
           [("Rz(30)", axis_rot([0, 0, 1], 30)), ("Rz(90)", axis_rot([0, 0, 1], 90)),
            ("Rx(90)", axis_rot([1, 0, 0], 90)), ("Ry(90)", axis_rot([0, 1, 0], 90)),
            ("Rx(37.4)", axis_rot([1, 0, 0], 37.4)),
            ("R(111,54.7)", axis_rot([1, 1, 1], 54.7356))]
    rows, nfail = [], 0

    def chk(test, name, what, val, tol, unit, note="", invert=False):
        nonlocal nfail
        ok = (val > tol) if invert else (val <= tol)
        if not ok:
            nfail += 1
        rows.append(dict(test=test, structure=name, quantity=what, value=float(val),
                         tol=float(tol), unit=unit, passed=bool(ok), control=invert, note=note))
        print("    %-4s %-34s %-9s %10.3e  %s %7.1e  %s %s"
              % (test, what[:34], unit, val, ">" if invert else "<", tol,
                 "ok  " if ok else "FAIL", note))

    print("\n" + "=" * 112)
    for name, at in cells:
        print("\n  -- %s  %s  cellpar=%s" % (name, at.get_chemical_formula(),
                                             np.round(at.get_cell().cellpar(), 2)))
        sym = at.get_chemical_symbols(); n = len(at)
        P = torch.tensor(at.get_positions(), dtype=torch.float64, device=dev)
        Cl = torch.tensor(at.get_cell()[:], dtype=torch.float64, device=dev)
        with torch.no_grad():
            b0 = d64.compute_per_center(P, Cl, sym).cpu().numpy()
            b0b = d64.compute_per_center(P, Cl, sym).cpu().numpy()
        den = np.where(np.linalg.norm(b0, axis=-1) > 0, np.linalg.norm(b0, axis=-1), 1)
        print("    noise floor (same input twice): %.3e rel"
              % float((np.linalg.norm(b0b - b0, axis=-1) / den).max()))

        wmx = 0.0
        for rn, R in rots:
            with torch.no_grad():
                bb = d64.compute_per_center(
                    torch.tensor(at.get_positions() @ R.T, dtype=torch.float64, device=dev),
                    torch.tensor(at.get_cell()[:] @ R.T, dtype=torch.float64, device=dev),
                    sym).cpu().numpy()
            wmx = max(wmx, float((np.linalg.norm(bb - b0, axis=-1) / den).max()))
        chk("T1", name, "per-centre features, max rel", wmx, TOL["desc64"], "rel",
            "%d rotations" % len(rots))

        with torch.no_grad():
            a32 = d32.compute_per_center(
                torch.tensor(at.get_positions(), dtype=torch.float32, device=dev),
                torch.tensor(at.get_cell()[:], dtype=torch.float32, device=dev), sym).cpu().numpy()
            d2 = np.where(np.linalg.norm(a32, axis=-1) > 0, np.linalg.norm(a32, axis=-1), 1)
            w32 = 0.0
            for rn, R in rots:
                bb = d32.compute_per_center(
                    torch.tensor(at.get_positions() @ R.T, dtype=torch.float32, device=dev),
                    torch.tensor(at.get_cell()[:] @ R.T, dtype=torch.float32, device=dev),
                    sym).cpu().numpy()
                w32 = max(w32, float((np.linalg.norm(bb - a32, axis=-1) / d2).max()))
        chk("T1", name, "per-centre features f32", w32, TOL["desc32"], "rel")

        e0, f0, s0 = efs(d64, m64, at, torch.float64, dev)
        de = df = ds = 0.0
        for rn, R in rots:
            r2 = at.copy(); r2.set_positions(at.get_positions() @ R.T)
            r2.set_cell(at.get_cell()[:] @ R.T); r2.set_pbc(True)
            e1, f1, s1 = efs(d64, m64, r2, torch.float64, dev)
            de = max(de, abs(e1 - e0) / n * 1000)
            df = max(df, float(np.abs(f1 - f0 @ R.T).max()))
            ds = max(ds, float(np.abs(s1 - R @ s0 @ R.T).max()))
        chk("T2", name, "diagnostic energy |dE|", de, TOL["e64"], "meV/atom")
        chk("T2", name, "force covariance max|dF|", df, TOL["f64"], "eV/A",
            "rms|F|=%.3f" % float(np.sqrt((f0 ** 2).mean())))
        chk("T4", name, "stress covariance max|dS|", ds, TOL["s64"], "GPa")

        em, fm, _ = efs(d64, m32, at, torch.float64, dev, torch.float32, False)
        dem = dfm = 0.0
        for rn, R in rots[:6]:
            r2 = at.copy(); r2.set_positions(at.get_positions() @ R.T)
            r2.set_cell(at.get_cell()[:] @ R.T); r2.set_pbc(True)
            e1, f1, _ = efs(d64, m32, r2, torch.float64, dev, torch.float32, False)
            dem = max(dem, abs(e1 - em) / n * 1000)
            dfm = max(dfm, float(np.abs(f1 - fm @ R.T).max()))
        chk("T2", name, "mixed precision |dE|", dem, TOL["e32"], "meV/atom", "desc f64, net f32")
        chk("T2", name, "mixed precision max|dF|", dfm, TOL["f32"], "eV/A")

        sh = at.copy(); sh.set_positions(at.get_positions() + np.array([0.37, -1.23, 2.71]))
        e1, f1, _ = efs(d64, m64, sh, torch.float64, dev, stress=False)
        chk("T3", name, "translation |dE|", abs(e1 - e0) / n * 1000, TOL["e64"], "meV/atom")
        chk("T3", name, "translation max|dF|", float(np.abs(f1 - f0).max()), TOL["f64"], "eV/A")
        print("    net force |sum F| = %.2e eV/A" % float(np.abs(f0.sum(0)).max()))

        pm = rng.permutation(n); pa = at[list(pm)]
        e1, f1, _ = efs(d64, m64, pa, torch.float64, dev, stress=False)
        with torch.no_grad():
            bp = d64.compute_per_center(
                torch.tensor(pa.get_positions(), dtype=torch.float64, device=dev),
                torch.tensor(pa.get_cell()[:], dtype=torch.float64, device=dev),
                pa.get_chemical_symbols()).cpu().numpy()
        chk("T3", name, "permutation features", float(np.linalg.norm(bp - b0[pm]) /
                                                      max(np.linalg.norm(b0), 1e-30)),
            TOL["desc64"], "rel")
        chk("T3", name, "permutation max|dF|", float(np.abs(f1 - f0[pm]).max()), TOL["f64"], "eV/A")

        lat = at.get_cell()[:]; tr = at.copy()
        q = at.get_positions().copy(); q[0] += lat[0] + 2 * lat[2]; tr.set_positions(q)
        e1, f1, _ = efs(d64, m64, tr, torch.float64, dev, stress=False)
        chk("T7", name, "lattice translation |dE|", abs(e1 - e0) / n * 1000, TOL["e64"], "meV/atom")

        # T6, step size chosen FROM THE GEOMETRY.
        # A central-difference stencil that straddles r_cut is invalid: the
        # cosine cutoff is C1 but not C2 there, so the h^2 Taylor expansion
        # does not hold and the order estimate collapses (measured on a real
        # val structure: order -1.82, Richardson 1.9e-7, while every component
        # away from the cutoff gave order 2.00 and ~1e-11).  So for each probed
        # atom, take the distance of its nearest neighbour pair to r_cut over
        # ALL periodic images and keep the largest step strictly below it.  The
        # pass threshold is the larger of 1e-9 and 20x the central-difference
        # roundoff floor eps*|E|/h, because FD cannot do better than that.
        from ase.neighborlist import neighbor_list
        rc_ = float(d64.config.r_cut)
        nb_i, _, nb_d = neighbor_list("ijd", at, rc_ + 0.05)
        gap = np.full(n, np.inf)
        for a_, gv in zip(nb_i, np.abs(nb_d - rc_)):
            gap[a_] = min(gap[a_], gv)
        for i in rng.choice(n, min(2, n), replace=False):
            i = int(i)
            gi = float(gap[i])
            h1 = min(1.0e-3, 0.4 * gi)
            if h1 < 1.0e-5:
                print("    T6   atom %-5d EXEMPT: nearest pair is %.2e A from r_cut, so no "
                      "step is both below the gap and above the roundoff floor" % (i, gi))
                rows.append(dict(test="T6", structure=name,
                                 quantity="atom %d exempt (pair %.2e A from r_cut)" % (i, gi),
                                 value=float(gi), tol=None, unit="A", passed=None,
                                 control=False, note="stencil would straddle the cutoff"))
                continue
            h2 = h1 / 2.0
            floor = 2.22e-16 * abs(e0) / h2
            tol_eff = max(1.0e-9, 20.0 * floor)
            for k in range(3):
                fdv = {}
                for h in (h1, h2):
                    v = []
                    for sg in (1, -1):
                        cc = at.copy(); pp = at.get_positions().copy()
                        pp[i, k] += sg * h; cc.set_positions(pp)
                        v.append(efs(d64, m64, cc, torch.float64, dev, stress=False)[0])
                    fdv[h] = -(v[0] - v[1]) / (2 * h)
                rich = abs((4 * fdv[h2] - fdv[h1]) / 3 - f0[i, k])
                chk("T6", name, "force Richardson, atom %d %s" % (i, "xyz"[k]), rich,
                    tol_eff, "eV/A",
                    "h=%.1e/%.1e, gap %.1e A, floor %.1e" % (h1, h2, gi, floor))

        with torch.no_grad():
            ob = o64.compute_per_center(P, Cl, sym).cpu().numpy()
            od = np.where(np.linalg.norm(ob, axis=-1) > 0, np.linalg.norm(ob, axis=-1), 1)
            wo = 0.0
            for rn, R in rots:
                bb = o64.compute_per_center(
                    torch.tensor(at.get_positions() @ R.T, dtype=torch.float64, device=dev),
                    torch.tensor(at.get_cell()[:] @ R.T, dtype=torch.float64, device=dev),
                    sym).cpu().numpy()
                wo = max(wo, float((np.linalg.norm(bb - ob, axis=-1) / od).max()))
        chk("T8", name, "ORIGINAL features, max rel", wo, TOL["desc64"], "rel",
            "control: MUST exceed tolerance", invert=True)

    print("\n" + "=" * 112)
    print("SUMMARY BY TEST")
    print("=" * 112)
    names = {"T1": "descriptor invariance under proper rotations",
             "T2": "diagnostic energy invariance and force covariance",
             "T3": "translation and permutation",
             "T4": "stress covariance",
             "T6": "force finite differences (Richardson)",
             "T7": "periodic images",
             "T8": "control: the ORIGINAL descriptor must fail"}
    allok = True
    for t in sorted(names):
        rr = [r for r in rows if r["test"] == t]
        bad = [r for r in rr if not r["passed"]]
        allok = allok and not bad
        print("%-4s %-54s %5d checks %5d failed  %s"
              % (t, names[t], len(rr), len(bad), "PASS" if not bad else "FAIL"))
    gate = rep["n_loaded"] >= 2 * args.per_split
    print("-" * 112)
    print("real train/val structures loaded: %d (requested %d per split)  %s"
          % (rep["n_loaded"], args.per_split, "PASS" if gate else "FAIL"))
    allok = allok and gate
    print("OVERALL: %s   (%d checks, %d failed)" % ("PASS" if allok else "FAIL",
                                                    len(rows), nfail))
    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            script=__file__, script_sha256=sha(__file__),
            production_module=args.module, production_sha256=sha(args.module),
            variant=C.VARIANT, c3_applied=False,
            torch=torch.__version__, numpy=np.__version__,
            python=sys.version.split()[0], device=str(dev), seed=args.seed,
            nrot=args.nrot, tolerances=TOL, weights={str(l): list(W[l]) for l in range(5)},
            real=rep, rows=rows, real_gate_passed=bool(gate),
            overall_pass=bool(allok)), indent=2))
        print("wrote %s" % args.json)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
