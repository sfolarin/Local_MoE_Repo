#!/usr/bin/env python3
"""
test_torchsoap_rotation.py

Measure how the TorchSOAP descriptor used by Local-MoE behaves under rotation.

READ-ONLY with respect to production code: the descriptor module is imported
from the path you give with --module and is never modified.  Nothing here is a
model input.  The "positive control" at the end is a *diagnostic subclass built
inside this script* (it rescales the angular basis so the harness can be shown
to give ~0 when the basis is correctly weighted); it is NOT the descriptor the
trained models were fitted with.

What is measured
----------------
A  Single-neighbour angular norms  sum_m B_lm(u)^2  for l = 0..4 (the exact
   quantity ChatGPT flagged), along x, y, z, (111) and over the whole sphere.
B  Per-centre power spectrum and the global 783-feature vector under
   (i) random proper rotations applied to positions AND cell together,
   (ii) rigid translation (control, should be ~0),
   (iii) dense vs neighbour-list agreement (if the module has both paths).
C  Positive control: same harness, angular basis rescaled to the correct
   real-harmonic weights (diagnostic only).

Convention: ASE row vectors.  pos' = pos @ R.T ,  cell' = cell @ R.T ,
R in SO(3) (det = +1).  Atom order is unchanged, so per-atom features of the
rotated structure are compared with the same atom's features before rotation.

Usage
-----
  python test_torchsoap_rotation.py --module PATH/torchsoap_..._nl.py \
      [--dtype float64|float32] [--nrot 20] [--seed 12345] \
      [--structures a.extxyz b.extxyz ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

L_MAX = 4


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("torchsoap_under_test", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["torchsoap_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Haar-random proper rotation (det = +1) via QR of a Gaussian matrix."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def rot_axis(axis: str, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fmt(x: float) -> str:
    return f"{x:.3e}"


# --------------------------------------------------------------------------
# test structures (built here so the test is self-contained)
# --------------------------------------------------------------------------
def build_structures(seed: int):
    from ase import Atoms
    from ase.spacegroup import crystal

    rng = np.random.default_rng(seed)
    out = []

    quartz = crystal(
        ["Si", "O"],
        basis=[(0.4697, 0.0, 0.0), (0.4135, 0.2669, 0.1191)],
        spacegroup=152,
        cellpar=[4.9134, 4.9134, 5.4052, 90, 90, 120],
    )
    out.append(("alpha-quartz (9 atoms, hexagonal cell)", quartz))

    crist = crystal(
        ["Si", "O"],
        basis=[(0.0, 0.0, 0.0), (0.125, 0.125, 0.125)],
        spacegroup=227,
        cellpar=[7.16, 7.16, 7.16, 90, 90, 90],
    )
    crist = crist.repeat((1, 1, 1))
    c2 = crist.copy()
    c2.rattle(stdev=0.12, seed=seed)
    strain = np.array([[1.02, 0.03, -0.01], [0.0, 0.98, 0.04], [0.02, 0.0, 1.01]])
    c2.set_cell(c2.get_cell()[:] @ strain, scale_atoms=True)
    out.append((f"beta-cristobalite rattled+sheared ({len(c2)} atoms, triclinic)", c2))

    # amorphous-like: random placement with a minimum distance, triclinic cell
    n_f = 24
    cell = np.array([[10.2, 0.0, 0.0], [1.1, 9.9, 0.0], [-0.7, 0.9, 10.4]])
    pos, sym = [], []
    tries = 0
    while len(pos) < 3 * n_f and tries < 200000:
        tries += 1
        f = rng.random(3)
        p = f @ cell
        ok = True
        for q in pos:
            d = p - q
            df = np.linalg.solve(cell.T, d)
            df -= np.round(df)
            if np.linalg.norm(df @ cell) < (1.35 if len(sym) else 1.35):
                ok = False
                break
        if ok:
            pos.append(p)
            sym.append("Si" if len(pos) <= n_f else "O")
    amorph = Atoms(sym, positions=np.array(pos), cell=cell, pbc=True)
    out.append((f"random SiO2-like ({len(amorph)} atoms, triclinic)", amorph))
    return out


def load_structures(paths, seed):
    if not paths:
        return build_structures(seed)
    from ase.io import read

    out = []
    for p in paths:
        frames = read(p, index=":")
        for k, a in enumerate(frames):
            out.append((f"{Path(p).name}[{k}] ({len(a)} atoms)", a))
    return out


def to_t(atoms, dtype):
    pos = torch.tensor(atoms.get_positions(), dtype=dtype)
    cell = torch.tensor(np.asarray(atoms.get_cell()), dtype=dtype)
    return pos, cell, atoms.get_chemical_symbols(), atoms.get_masses()


def rotated(atoms, R):
    a = atoms.copy()
    a.set_cell(np.asarray(atoms.get_cell()) @ R.T, scale_atoms=False)
    a.set_positions(atoms.get_positions() @ R.T)
    return a


# --------------------------------------------------------------------------
# A. single-neighbour angular norms and the correct weights
# --------------------------------------------------------------------------
def sphere_quadrature(n_theta=96, n_phi=192):
    xg, wg = np.polynomial.legendre.leggauss(n_theta)  # in cos(theta)
    ph = np.arange(n_phi) * 2 * np.pi / n_phi
    ct = xg[:, None] * np.ones((1, n_phi))
    st = np.sqrt(1 - ct**2)
    x = (st * np.cos(ph)[None, :]).ravel()
    y = (st * np.sin(ph)[None, :]).ravel()
    z = ct.ravel()
    w = (wg[:, None] * np.ones((1, n_phi)) * (2 * np.pi / n_phi)).ravel() / (4 * np.pi)
    return np.stack([x, y, z], axis=1), w


def basis_blocks(desc, u, dtype=torch.float64):
    """Return list over l of (n, 2l+1) arrays of the module's own angular basis."""
    ut = torch.tensor(u, dtype=dtype)
    r = torch.ones(ut.shape[0], dtype=dtype)
    blocks = desc._real_sph_harm_basis(ut, r)
    return [b.detach().cpu().numpy() for b in blocks]


def control_weights(mod):
    """Per-(l,m) weights 1/<B_m^2> that make sum_m w_m B_m(u)^2 direction independent (diagnostic only)."""
    desc = mod.TorchSOAPDescriptor()
    U, w = sphere_quadrature()
    bl = basis_blocks(desc, U)
    return {l: 1.0 / np.diag((b * w[:, None]).T @ b) for l, b in enumerate(bl)}


def part_A(mod, report):
    desc = mod.TorchSOAPDescriptor()
    print("\n" + "=" * 100)
    print("A. SINGLE-NEIGHBOUR ANGULAR NORM   S_l(u) = sum_m B_lm(u)^2   (module's own _real_sph_harm_basis, r = 1)")
    print("=" * 100)
    dirs = {
        "x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1],
        "(110)/sqrt2": [1, 1, 0], "(111)/sqrt3": [1, 1, 1],
    }
    hdr = f"{'direction':14s}" + "".join(f"{'l=' + str(l):>12s}" for l in range(L_MAX + 1))
    print(hdr)
    tabA = {}
    for name, v in dirs.items():
        u = np.array([v], dtype=float)
        u /= np.linalg.norm(u)
        bl = basis_blocks(desc, u)
        s = [float(np.sum(b**2)) for b in bl]
        tabA[name] = s
        print(f"{name:14s}" + "".join(f"{x:12.6f}" for x in s))
    # sphere extrema
    U, w = sphere_quadrature()
    bl = basis_blocks(desc, U)
    S = [np.sum(b**2, axis=1) for b in bl]
    print(f"{'min over sphere':14s}" + "".join(f"{s.min():12.6f}" for s in S))
    print(f"{'max over sphere':14s}" + "".join(f"{s.max():12.6f}" for s in S))
    print(f"{'max/min':14s}" + "".join(f"{s.max() / max(s.min(), 1e-300):12.4f}" for s in S))
    print(f"{'mean (4pi avg)':14s}" + "".join(f"{float(np.sum(w * s)):12.6f}" for s in S))
    # Gram matrix of the polynomial basis on the sphere
    print("\nGram matrix G_mm' = <B_m B_m'> on the sphere (should be diagonal if each B is one real harmonic):")
    weights = {}
    gram_info = {}
    for l, b in enumerate(bl):
        G = (b * w[:, None]).T @ b
        diag = np.diag(G).copy()
        off = np.max(np.abs(G - np.diag(diag)))
        gram_info[l] = {"diag": diag.tolist(), "max_offdiag": float(off)}
        weights[l] = 1.0 / diag
        print(f"  l={l}: diag = {np.array2string(diag, precision=5, separator=', ')}   max|offdiag| = {off:.2e}")
        # normalised so that weighted sum is (2l+1)
        chk = (b**2 * weights[l][None, :]).sum(axis=1)
        print(f"        weights 1/diag = {np.array2string(weights[l], precision=4, separator=', ')} ;"
              f" weighted sum over m = {chk.min():.12f} .. {chk.max():.12f}  (target 2l+1 = {2 * l + 1})")
    report["A"] = {"table": tabA,
                   "sphere_min": [float(s.min()) for s in S],
                   "sphere_max": [float(s.max()) for s in S],
                   "gram": gram_info}
    return weights


# --------------------------------------------------------------------------
# B. descriptor invariance
# --------------------------------------------------------------------------
def make_desc(mod, normalize, cls=None):
    cfg = mod.TorchSOAPConfig(normalize_power=normalize)
    return (cls or mod.TorchSOAPDescriptor)(cfg).eval()


def rel_l2(a, b, axis=-1):
    return np.linalg.norm(a - b, axis=axis) / np.maximum(np.linalg.norm(b, axis=axis), 1e-300)


def per_l_blocks(raw):  # raw: (N, 390) l-major, 78 per l
    return [raw[:, k * 78:(k + 1) * 78] for k in range(L_MAX + 1)]


def invariance_for(desc_norm, desc_raw, atoms, LRs, dtype, path="compute_per_center"):
    pos, cell, sym, masses = to_t(atoms, dtype)
    f = getattr(desc_norm, path)
    fr = getattr(desc_raw, path)
    with torch.no_grad():
        p0 = f(pos, cell, sym).double().numpy()
        r0 = fr(pos, cell, sym).double().numpy()
        g0 = desc_norm(pos, cell, sym, masses).double().numpy()
    res = {"labels": [], "atom_rel_l2": [], "atom_max_abs": [], "perl_rel": [[] for _ in range(L_MAX + 1)],
           "glob_rel": [], "glob_max_abs": [], "scalar_max_abs": []}
    for label, R in LRs:
        res["labels"].append(label)
        a2 = rotated(atoms, R)
        pos2, cell2, sym2, _ = to_t(a2, dtype)
        with torch.no_grad():
            p1 = f(pos2, cell2, sym2).double().numpy()
            r1 = fr(pos2, cell2, sym2).double().numpy()
            g1 = desc_norm(pos2, cell2, sym2, masses).double().numpy()
        res["atom_rel_l2"].append(rel_l2(p1, p0))
        res["atom_max_abs"].append(np.max(np.abs(p1 - p0), axis=1))
        for k, (b1, b0) in enumerate(zip(per_l_blocks(r1), per_l_blocks(r0))):
            res["perl_rel"][k].append(rel_l2(b1, b0))
        res["glob_rel"].append(float(np.linalg.norm(g1[:-3] - g0[:-3]) / np.linalg.norm(g0[:-3])))
        res["glob_max_abs"].append(float(np.max(np.abs(g1[:-3] - g0[:-3]))))
        res["scalar_max_abs"].append(float(np.max(np.abs(g1[-3:] - g0[-3:]))))
    return res, p0


def summarize(res, select=None):
    idx = list(range(len(res["labels"]))) if select is None else [i for i, l in enumerate(res["labels"]) if select(l)]
    ar = np.concatenate([res["atom_rel_l2"][i] for i in idx])
    am = np.concatenate([res["atom_max_abs"][i] for i in idx])
    pl = [np.concatenate([res["perl_rel"][k][i] for i in idx]) for k in range(L_MAX + 1)]
    res = {"glob_rel": [res["glob_rel"][i] for i in idx], "glob_max_abs": [res["glob_max_abs"][i] for i in idx],
           "scalar_max_abs": [res["scalar_max_abs"][i] for i in idx]}
    return {
        "atom_rel_l2_max": float(ar.max()), "atom_rel_l2_mean": float(ar.mean()),
        "atom_maxabs_max": float(am.max()),
        "perl_rel_max": [float(x.max()) for x in pl],
        "perl_rel_mean": [float(x.mean()) for x in pl],
        "glob_rel_max": float(max(res["glob_rel"])),
        "glob_maxabs_max": float(max(res["glob_max_abs"])),
        "scalar_maxabs_max": float(max(res["scalar_max_abs"])),
    }


def print_summary(tag, s):
    print(f"  {tag}")
    print(f"     per-centre (as seen by the models, 390 normalised features):"
          f" max|dp|={fmt(s['atom_maxabs_max'])}  rel-L2 max={fmt(s['atom_rel_l2_max'])} mean={fmt(s['atom_rel_l2_mean'])}")
    print("     per-l raw power spectrum rel-L2 (max over atoms x rotations): "
          + "  ".join(f"l={k}:{fmt(v)}" for k, v in enumerate(s["perl_rel_max"])))
    print(f"     global 780 descriptor (Si-mean, O-mean): rel-L2 max={fmt(s['glob_rel_max'])}  max|d|={fmt(s['glob_maxabs_max'])};"
          f"  3 scalars max|d|={fmt(s['scalar_maxabs_max'])}")


def context_spread(p0):
    """Typical relative distance between different atoms' features (scale reference)."""
    n = p0.shape[0]
    if n < 2:
        return float("nan")
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            d.append(np.linalg.norm(p0[i] - p0[j]) / max(np.linalg.norm(p0[j]), 1e-300))
    return float(np.median(d))


def part_B(mod, structures, args, dtype, report, label, cls=None, do_dense=True):
    desc_n = make_desc(mod, True, cls)
    desc_r = make_desc(mod, False, cls)
    rng = np.random.default_rng(args.seed)
    LRs = [(f"random{k:02d}", random_rotation(rng)) for k in range(args.nrot)]
    LRs += [("Rz(30)", rot_axis("z", 30)), ("Rz(90)", rot_axis("z", 90)), ("Rx(90)", rot_axis("x", 90)),
            ("Ry(90)", rot_axis("y", 90)), ("Rz(180)", rot_axis("z", 180)), ("Rx(180)", rot_axis("x", 180))]
    out = {}
    print("\n" + "=" * 100)
    print(f"B. DESCRIPTOR vs ROTATION  [{label}]   dtype={args.dtype}   rotations/structure={len(LRs)} "
          f"({args.nrot} Haar-random SO(3) + Rz30,Rz90,Rx90,Ry90,Rz180,Rx180)")
    print("=" * 100)
    for name, atoms in structures:
        print(f"\n- {name}")
        res, p0 = invariance_for(desc_n, desc_r, atoms, LRs, dtype)
        s = summarize(res)
        s["median_between_atom_rel_distance"] = context_spread(p0)
        s["by_rotation"] = {}
        for lab in [l for l in res["labels"] if not l.startswith("random")]:
            s["by_rotation"][lab] = summarize(res, select=lambda l, lab=lab: l == lab)
        s["random_only"] = summarize(res, select=lambda l: l.startswith("random"))
        print_summary("neighbour-list path (compute_per_center), ALL rotations", s)
        print_summary(f"   random SO(3) only (n={args.nrot})", s["random_only"])
        print("     named rotations, per-centre rel-L2 max / global-780 rel-L2:  " + "  ".join(
            f"{k}: {fmt(v['atom_rel_l2_max'])}/{fmt(v['glob_rel_max'])}" for k, v in s["by_rotation"].items()))
        print(f"     scale reference: median relative distance between DIFFERENT atoms' features = {fmt(s['median_between_atom_rel_distance'])}")
        out[name] = {"nl": s}
        if do_dense and hasattr(desc_n, "compute_per_center_dense") and len(atoms) <= args.max_dense_atoms:
            resd, _ = invariance_for(desc_n, desc_r, atoms, LRs, dtype, path="compute_per_center_dense")
            sd = summarize(resd)
            print_summary("dense path (compute_per_center_dense, original code)", sd)
            out[name]["dense"] = sd
        # translation control
        pos, cell, sym, masses = to_t(atoms, dtype)
        with torch.no_grad():
            p0t = desc_n.compute_per_center(pos, cell, sym).double().numpy()
            p1t = desc_n.compute_per_center(pos + torch.tensor([0.37, -1.91, 2.43], dtype=dtype), cell, sym).double().numpy()
        tr = float(np.max(np.abs(p1t - p0t)))
        print(f"     translation control (rigid shift, no wrapping): max|dp| = {fmt(tr)}")
        out[name]["translation_maxabs"] = tr
        # improper operations (det = -1): inversion and mirror z -> -z, positions and cell together
        imp = {}
        for tag, M in (("inversion", -np.eye(3)), ("mirror z->-z", np.diag([1.0, 1.0, -1.0]))):
            Mt = torch.tensor(M, dtype=dtype)
            with torch.no_grad():
                pm = desc_n.compute_per_center(pos @ Mt.T, cell @ Mt.T, sym).double().numpy()
            imp[tag] = float(np.max(np.abs(pm - p0t)))
        out[name]["improper_maxabs"] = imp
        print("     improper operations (pos and cell together): " + "  ".join(f"{k}: max|dp| = {fmt(v)}" for k, v in imp.items()))
        # dense vs nl on identical input
        if hasattr(desc_n, "compute_per_center_dense") and len(atoms) <= args.max_dense_atoms:
            with torch.no_grad():
                pd = desc_n.compute_per_center_dense(pos, cell, sym).double().numpy()
            dn = float(np.max(np.abs(pd - p0t)))
            print(f"     dense vs neighbour-list on the same input: max|dp| = {fmt(dn)}")
            out[name]["dense_vs_nl_maxabs"] = dn
    tol = args.tol if args.tol else (1e-9 if args.dtype == "float64" else 1e-4)
    worst_atom = max(v["nl"]["random_only"]["atom_rel_l2_max"] for v in out.values())
    worst_glob = max(v["nl"]["random_only"]["glob_rel_max"] for v in out.values())
    worst_l = [max(v["nl"]["random_only"]["perl_rel_max"][k] for v in out.values()) for k in range(L_MAX + 1)]
    print(f"\n  VERDICT [{label}] (tolerance {tol:g} relative, random SO(3) rotations, worst case over structures)")
    print(f"     per-centre 390-vector rel-L2: {fmt(worst_atom)}  -> {'PASS' if worst_atom < tol else 'FAIL'}")
    print(f"     global 780-vector rel-L2    : {fmt(worst_glob)}  -> {'PASS' if worst_glob < tol else 'FAIL'}")
    print("     raw power spectrum by l      : " + "  ".join(
        f"l={k}: {fmt(x)} {'PASS' if x < tol else 'FAIL'}" for k, x in enumerate(worst_l)))
    report.setdefault("B", {})[label] = out
    report.setdefault("verdict", {})[label] = {"tol": tol, "per_centre": worst_atom, "global": worst_glob, "per_l": worst_l}
    return out


# --------------------------------------------------------------------------
# C. positive control (diagnostic subclass; NOT a model input)
# --------------------------------------------------------------------------
def make_control_class(mod, weights):
    base = mod.TorchSOAPDescriptor

    class CorrectlyWeightedDiagnostic(base):
        """DIAGNOSTIC ONLY. Multiplies B_lm by sqrt(w_lm) so equal-weight
        contraction equals the rotation-invariant weighted contraction."""

        def _real_sph_harm_basis(self, dcart, r):
            blocks = super()._real_sph_harm_basis(dcart, r)
            out = []
            for l, b in enumerate(blocks):
                w = torch.tensor(np.sqrt(weights[l]), dtype=b.dtype, device=b.device)
                out.append(b * w)
            return out

    return CorrectlyWeightedDiagnostic


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--nrot", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--structures", nargs="*", default=None)
    ap.add_argument("--max-dense-atoms", type=int, default=128)
    ap.add_argument("--tol", type=float, default=None, help="relative tolerance for PASS/FAIL (default 1e-9 float64, 1e-4 float32)")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    mp = Path(args.module).resolve()
    mod = load_module(mp)
    report = {"module": str(mp), "module_sha256": sha256_file(mp),
              "script_sha256": sha256_file(Path(__file__).resolve()),
              "torch": torch.__version__, "numpy": np.__version__, "dtype": args.dtype,
              "seed": args.seed, "nrot": args.nrot}
    print("module        :", mp)
    print("module sha256 :", report["module_sha256"])
    print("script sha256 :", report["script_sha256"])
    print("torch / numpy :", torch.__version__, np.__version__, "| dtype:", args.dtype, "| seed:", args.seed)
    print("module has compute_per_center_dense:", hasattr(mod.TorchSOAPDescriptor, "compute_per_center_dense"))
    print("rotation convention: pos' = pos @ R.T ; cell' = cell @ R.T ; R in SO(3)")
    print("eps in module config:", mod.TorchSOAPConfig().eps)

    weights = part_A(mod, report)
    structures = load_structures(args.structures, args.seed)
    part_B(mod, structures, args, dtype, report, "AS IMPLEMENTED (production module)")
    if not args.no_control:
        Ctl = make_control_class(mod, weights)
        part_B(mod, structures, args, dtype, report,
               "POSITIVE CONTROL: correctly weighted angular basis (diagnostic subclass, NOT the trained descriptor)",
               cls=Ctl, do_dense=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print("\nwrote", args.json)


if __name__ == "__main__":
    main()
