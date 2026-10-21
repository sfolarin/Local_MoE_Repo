#!/usr/bin/env python3
"""
local_moe_reference_numpy.py

A PyTorch-free, line-by-line reference implementation of the Local-MoE energy
(holdout-40, stage C1b).  It exists to be READ and ported: every step of the model
is a few lines of numpy, in the order the C++ code has to do them.  It reads only
what export_local_moe_spec.py wrote (spec/local_moe_spec.{npz,json}).

    E = N*e_ref + sum_i e0_i + sum_i sum_k w_ik * delta_ik

  1. edges   all (i, j, image) with |R_j - R_i + S.cell + eps| < r_cut, fully periodic,
             the home-cell self pair excluded                    (any neighbour search)
  2. D_i     TorchSOAP power spectrum, 390 numbers per atom, float64
  3. x_i     [D_i (cast to float32), isSi, isO]                  392, float32
  4. e0_i    common MLP on (x - mean_c)/std_c
  5. w_ik    softmax over the 7 outputs of the routing MLP on (x - mean_r)/std_r
  6. delta   residual MLP on (x - mean_c)/std_c, minus its mean over the 7 outputs
  7. E       sums in float64

Forces are F = -dE/dR with the weights w_ik depending on R.  This script does not
differentiate analytically; --fd checks the conservative force of selected atoms by
central finite differences in float64 (which includes dw/dR), and --case compares
every intermediate against the PyTorch model's own numbers.

Examples
--------
  # compare against a stored reference case (energy, per-atom pieces, descriptor)
  python local_moe_reference_numpy.py --spec port_spec_v1/spec/local_moe_spec \
      --case port_spec_v1/reference/case_00_N60.npz

  # finite-difference forces of three atoms against the stored PyTorch forces
  python local_moe_reference_numpy.py --spec ... --case ... --fd 0,1,2
"""
import argparse
import json
from pathlib import Path

import numpy as np

NEG_EPS = 1.0e-12


# ---------------------------------------------------------------------------
# 1. edges
# ---------------------------------------------------------------------------
def edges(positions, cell, r_cut):
    """(i, j, d) with d = R_j - R_i + S @ cell, fully periodic, within r_cut (+ tiny margin)."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    atoms = Atoms(numbers=np.ones(len(positions), dtype=int), positions=positions, cell=cell, pbc=True)
    i, j, d = neighbor_list("ijD", atoms, r_cut + 1.0e-6)
    return np.asarray(i), np.asarray(j), np.asarray(d, dtype=np.float64)


# ---------------------------------------------------------------------------
# 2. TorchSOAP power spectrum
# ---------------------------------------------------------------------------
def angular_basis(d, r, eps, l_max=4):
    """25 real-harmonic-like polynomials of the unit vector, in the order l=0 | l=1 | ... | l=4."""
    x = d[:, 0] / (r + eps)
    y = d[:, 1] / (r + eps)
    z = d[:, 2] / (r + eps)
    cols = [np.ones_like(x)]                                                        # l = 0
    cols += [x, y, z]                                                               # l = 1
    cols += [x * y, y * z, 0.5 * (3.0 * z * z - 1.0), x * z, 0.5 * (x * x - y * y)]  # l = 2
    cols += [y * (3.0 * x * x - y * y), x * y * z, y * (5.0 * z * z - 1.0), z * (5.0 * z * z - 3.0),
             x * (5.0 * z * z - 1.0), z * (x * x - y * y), x * (x * x - 3.0 * y * y)]  # l = 3
    cols += [x * y * (x * x - y * y), y * z * (3.0 * x * x - y * y), x * y * (7.0 * z * z - 1.0),
             y * z * (7.0 * z * z - 3.0), 35.0 * z ** 4 - 30.0 * z * z + 3.0, x * z * (7.0 * z * z - 3.0),
             (x * x - y * y) * (7.0 * z * z - 1.0), x * z * (x * x - 3.0 * y * y),
             x ** 4 - 6.0 * x * x * y * y + y ** 4]                                  # l = 4
    return np.stack(cols, axis=1)                                                   # (E, 25)


def soap_descriptor(positions, cell, numbers, spec, meta):
    c = meta["descriptor"]
    r_cut, n_max, sigma, eps, lam = c["r_cut"], c["n_max"], c["radial_sigma"], c["eps"], c["power_norm_lambda"]
    n_atoms = len(positions)
    n_species = len(meta["species_order"])

    i, j, d = edges(positions, cell, r_cut)
    r = np.sqrt(np.sum((d + eps) ** 2, axis=1))                  # eps is added to each component, as in the model
    keep = r < r_cut
    i, j, d, r = i[keep], j[keep], d[keep], r[keep]

    centers = spec["radial_centers"]                              # exported from the module (float64)
    fc = 0.5 * (np.cos(np.pi * np.clip(r / r_cut, 0.0, 1.0)) + 1.0)           # cosine cutoff
    g = np.exp(-0.5 * ((r[:, None] - centers[None, :]) / sigma) ** 2) * fc[:, None]   # (E, n_max)
    ylm = angular_basis(d, r, eps)                                # (E, 25)

    z_to_species = {int(z): k for k, z in enumerate(meta["atomic_numbers"])}
    sp = np.array([z_to_species[int(z)] for z in numbers])        # species of every atom (neighbour species = sp[j])

    # c[i, s*n_max + n, lm] = sum over neighbours j of species s of g_n(r_ij) * Y_lm(rhat_ij)
    flat = i * n_species + sp[j]
    acc = np.zeros((n_atoms * n_species, n_max, 25))
    for n in range(n_max):
        for m in range(25):
            acc[:, n, m] = np.bincount(flat, weights=g[:, n] * ylm[:, m], minlength=n_atoms * n_species)
    coeff = acc.reshape(n_atoms, n_species * n_max, 25)           # channel = species*n_max + n

    pair_i, pair_j = spec["pair_i"], spec["pair_j"]               # 78 upper-triangular channel pairs
    offs = [0, 1, 4, 9, 16, 25]
    power = [np.einsum("ipm,ipm->ip", coeff[:, pair_i, offs[l]:offs[l + 1]], coeff[:, pair_j, offs[l]:offs[l + 1]])
             for l in range(c["l_max"] + 1)]
    desc = np.concatenate(power, axis=1)                          # (N, 390), l-major
    if c["normalize_power"]:
        norm = np.linalg.norm(desc, axis=1, keepdims=True)
        desc = desc / np.sqrt(norm * norm + lam ** 2)
    return desc, int(len(i))


# ---------------------------------------------------------------------------
# 3-6. networks
# ---------------------------------------------------------------------------
def run_net(ops, spec, h, dtype):
    for op in ops:
        if op["op"] == "linear":
            h = h @ spec[op["W"]].astype(dtype).T + spec[op["b"]].astype(dtype)
        elif op["op"] == "layernorm":
            mu = h.mean(axis=-1, keepdims=True)
            var = ((h - mu) ** 2).mean(axis=-1, keepdims=True)              # biased variance, as torch
            h = (h - mu) / np.sqrt(var + dtype(op["eps"])) * spec[op["w"]].astype(dtype) + spec[op["b"]].astype(dtype)
        elif op["op"] == "silu":
            with np.errstate(over="ignore"):
                h = h / (dtype(1.0) + np.exp(-h))
        elif op["op"] == "dropout":
            pass                                                            # identity at inference
        else:
            raise ValueError(op)
    return h


def local_moe(positions, cell, numbers, spec, meta, nets_dtype=np.float32):
    """Returns total energy and all per-atom pieces."""
    n = len(positions)
    desc, n_edges = soap_descriptor(positions, cell, numbers, spec, meta)
    onehot = np.stack([(numbers == z).astype(np.float64) for z in meta["atomic_numbers"]], axis=1)
    x = np.concatenate([desc, onehot], axis=1).astype(nets_dtype)           # (N, 392)

    cm, cs = spec["common.feature_mean"].astype(nets_dtype), spec["common.feature_std"].astype(nets_dtype)
    rm, rs = spec["routing.feature_mean"].astype(nets_dtype), spec["routing.feature_std"].astype(nets_dtype)
    xc = (x - cm) / cs
    xr = (x - rm) / rs

    nets = meta["networks"]
    e0 = run_net(nets["common"]["ops"], spec, xc, nets_dtype)[:, 0]
    logits = run_net(nets["routing"]["ops"], spec, xr, nets_dtype)
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w = w / w.sum(axis=1, keepdims=True)
    raw = run_net(nets["residual"]["ops"], spec, xc, nets_dtype)             # residual head: COMMON normalisation
    delta = raw - raw.mean(axis=1, keepdims=True)

    e_ref = float(spec["e_ref_eV_per_atom"])
    e_total = n * e_ref + float(np.sum(e0.astype(np.float64))) + float(np.sum((w * delta).astype(np.float64)))
    return dict(E=e_total, e0=e0, w=w, delta=delta, x=x, soap=desc, n_edges=n_edges,
                E_common=n * e_ref + float(np.sum(e0.astype(np.float64))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="prefix of local_moe_spec (without .npz / .json)")
    ap.add_argument("--case", required=True, help="reference case .npz written by the exporter")
    ap.add_argument("--fd", default="", help="comma list of atom indices for finite-difference forces")
    ap.add_argument("--fd-step", type=float, default=1.0e-4)
    args = ap.parse_args()

    spec = dict(np.load(args.spec + ".npz"))
    meta = json.load(open(args.spec + ".json"))
    case = np.load(args.case, allow_pickle=False)
    pos, cell, numbers = case["positions"], case["cell"], case["numbers"]
    n = len(pos)
    print(f"case {Path(args.case).name}: N = {n}, record {case['record_id']}")

    r = local_moe(pos, cell, numbers, spec, meta, np.float32)
    print(f"edges inside r_cut: {r['n_edges']} ({r['n_edges'] / n:.1f} per atom)")

    def cmp(name, a, b, unit=""):
        print(f"  {name:34s} max|diff| = {np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))):.3e} {unit}")

    cmp("descriptor (float64, 390)", r["soap"], case["soap64"])
    if "x" in case.files:
        cmp("feature vector x (float32, 392)", r["x"], case["x"])
    cmp("e0 per atom (eV)", r["e0"], case["e0"], "eV")
    cmp("routing weights w_ik", r["w"], case["w"])
    cmp("centred residual delta_ik (eV)", r["delta"], case["delta"], "eV")
    dE = abs(r["E"] - float(case["E_full"]))
    print(f"  {'total energy E (eV)':34s} numpy {r['E']:+.8f}  torch {float(case['E_full']):+.8f}  "
          f"|diff| = {dE:.3e} eV = {dE / n:.2e} eV/atom")
    print(f"  {'common-only energy (eV)':34s} |diff| = {abs(r['E_common'] - float(case['E_common'])):.3e} eV")
    print("  note: PyTorch sums the per-atom float32 energies in float32 before casting; this script sums in float64, "
          "so totals agree to ~1e-7 relative, not bit for bit.")

    if args.fd.strip():
        atoms = [int(a) for a in args.fd.split(",") if a.strip()]
        h = args.fd_step
        print(f"\nfinite-difference forces (float64 networks, central, h = {h:g} A) vs stored PyTorch forces:")
        worst = 0.0
        for a in atoms:
            f_fd = np.zeros(3)
            for k in range(3):
                pp, pm = pos.copy(), pos.copy()
                pp[a, k] += h
                pm[a, k] -= h
                f_fd[k] = -(local_moe(pp, cell, numbers, spec, meta, np.float64)["E"]
                            - local_moe(pm, cell, numbers, spec, meta, np.float64)["E"]) / (2 * h)
            f_t = case["F_full"][a]
            worst = max(worst, float(np.max(np.abs(f_fd - f_t))))
            print(f"  atom {a:5d}  FD {np.array2string(f_fd, precision=6)}  torch {np.array2string(f_t, precision=6)}")
        print(f"  worst component difference = {worst:.3e} eV/A")


if __name__ == "__main__":
    main()
