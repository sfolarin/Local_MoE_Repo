#!/usr/bin/env python3
"""
decompose_gamma_offset.py        (CPU only, read-only)

Where does the interface-energy error come from?

    gamma = (E_AB - E_A - E_B) / (2 area) * 16.02176634            [J/m^2]

With dE_X = E_X(model) - E_X(MATPes) for X in {AB, A, B} the error is EXACTLY

    d_gamma = k * (dE_AB - dE_A - dE_B),        k = 16.02176634 / (2 area)

so it splits into three additive terms  c_AB = k dE_AB,  c_A = -k dE_A,  c_B = -k dE_B.
In per-atom terms (eps_X = dE_X / N_X) the same error is

    d_gamma = k N_AB * ( eps_AB - eps_comp ),    eps_comp = (N_A eps_A + N_B eps_B) / N_AB

i.e. only the DIFFERENCE between the per-atom error of the interface cell and that of its two components
matters, multiplied by the amplification  k N_AB  (J/m^2 per eV/atom).

Inputs (all already on disk): the evaluator's gamma CSV (model energies E_AB_eV, E_A_eV, E_B_eV), the manifest
(paths), and the MATPes energies stored in the relaxed structure files. Two checks stop the script if they fail:
the reconstructed model gamma must equal the CSV's, and the MATPes gamma recomputed from the files must equal
the manifest value.

Run from External_Benchmark_Oishi2025/:
    python decompose_gamma_offset.py --gamma-csv gamma66_common_holdout40_ew10000_v1.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read

CONV = 16.02176634


def matpes_energy(atoms):
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        for k in ("energy", "REF_energy", "total_energy"):
            if k in atoms.info:
                return float(atoms.info[k])
    raise ValueError("no reference energy stored in the structure file")


def decompose(g, man, ext):
    rows = []
    for r in g.itertuples():
        m = man[man.case == r.case]
        if len(m) != 1:
            raise ValueError(f"{r.case}: {len(m)} manifest rows")
        m = m.iloc[0]
        ab, a, b = (read(ext / m[k], index=-1) for k in ("ab_path", "a_path", "b_path"))
        if len(a) + len(b) != len(ab):
            raise ValueError(f"{r.case}: atom counts do not close")
        eab, ea, eb = matpes_energy(ab), matpes_energy(a), matpes_energy(b)
        area = float(np.linalg.norm(np.cross(ab.cell[0], ab.cell[1])))
        k = CONV / (2.0 * area)
        if abs(k * (eab - ea - eb) - float(m.gamma_MATPES_J_m2)) > 1e-6:
            raise ValueError(f"{r.case}: MATPes gamma from the files ({k * (eab - ea - eb):.6f}) differs from the manifest "
                             f"({float(m.gamma_MATPES_J_m2):.6f}); the energies read from the files are not the ones used before")
        dab, da, db = r.E_AB_eV - eab, r.E_A_eV - ea, r.E_B_eV - eb
        n_ab, n_a, n_b = len(ab), len(a), len(b)
        d_gamma = k * (dab - da - db)
        if abs(d_gamma - (r.gamma_common_J_m2 - float(m.gamma_MATPES_J_m2))) > 1e-6:
            raise ValueError(f"{r.case}: decomposition does not reproduce the stored gamma error")
        eps_ab, eps_a, eps_b = dab / n_ab, da / n_a, db / n_b
        eps_comp = (da + db) / n_ab
        rows.append(dict(case=r.case, new_split=r.new_split, N_AB=n_ab, N_A=n_a, N_B=n_b, area_A2=area,
                         dgamma=d_gamma, c_AB=k * dab, c_A=-k * da, c_B=-k * db,
                         eps_AB_meV=1000 * eps_ab, eps_A_meV=1000 * eps_a, eps_B_meV=1000 * eps_b,
                         eps_comp_meV=1000 * eps_comp, diff_meV=1000 * (eps_ab - eps_comp), amp_J_per_eV_atom=k * n_ab))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma-csv", required=True)
    ap.add_argument("--manifest", default="gamma66_manifest.csv")
    ap.add_argument("--ext", default=".")
    ap.add_argument("--out", default="gamma_offset_decomposition.csv")
    args = ap.parse_args()
    ext = Path(args.ext)
    t = decompose(pd.read_csv(ext / args.gamma_csv), pd.read_csv(ext / args.manifest), ext)
    t.to_csv(args.out, index=False)
    print(f"{len(t)} rows; decomposition reproduces every stored gamma error to 1e-6 J/m2; MATPes gamma re-derived from the files\n")

    cols = ["dgamma", "c_AB", "c_A", "c_B", "eps_AB_meV", "eps_comp_meV", "diff_meV", "amp_J_per_eV_atom"]
    out = []
    for name, g in [("held-out (test)", t[t.new_split == "test"]), ("in training", t[t.new_split == "train"]), ("all", t)]:
        d = {"set": name, "N": len(g)}
        d.update({c: float(g[c].mean()) for c in cols})
        d["sem_dgamma"] = float(g.dgamma.std(ddof=1) / np.sqrt(len(g))) if len(g) > 1 else float("nan")
        out.append(d)
    S = pd.DataFrame(out)
    print("MEAN error terms (J/m2) and per-atom errors (meV/atom). dgamma = c_AB + c_A + c_B")
    print(S[["set", "N", "dgamma", "sem_dgamma", "c_AB", "c_A", "c_B"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nper-atom view (mean over rows):  diff = eps_AB - eps_comp,  amplification = k*N_AB")
    print(S[["set", "eps_AB_meV", "eps_comp_meV", "diff_meV", "amp_J_per_eV_atom"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    for name in ("held-out (test)", "all"):
        g = t[t.new_split == "test"] if name.startswith("held") else t
        need = g.dgamma.mean() / g.amp_J_per_eV_atom.mean() * 1000
        print(f"\n{name}: a mean error of {g.dgamma.mean():+.4f} J/m2 needs only a {need:+.2f} meV/atom per-atom difference between the "
              f"interface cell and its components (cf. network RMSE of 3-5 meV/atom)")
    te = t[t.new_split == "test"]
    print(f"\nheld-out rows where each term is positive: c_AB {int((te.c_AB > 0).sum())}/{len(te)}, "
          f"c_A {int((te.c_A > 0).sum())}/{len(te)}, c_B {int((te.c_B > 0).sum())}/{len(te)}, total error {int((te.dgamma > 0).sum())}/{len(te)}")
    print(f"correlation of the held-out error with the amplification k*N_AB: {np.corrcoef(te.dgamma, te.amp_J_per_eV_atom)[0, 1]:+.2f}; "
          f"with eps_AB: {np.corrcoef(te.dgamma, te.eps_AB_meV)[0, 1]:+.2f}; with eps_comp: {np.corrcoef(te.dgamma, te.eps_comp_meV)[0, 1]:+.2f}")
    print("\nlargest held-out errors and their terms:")
    print(te.reindex(te.dgamma.abs().sort_values(ascending=False).index).head(8)[
        ["case", "N_AB", "dgamma", "c_AB", "c_A", "c_B", "eps_AB_meV", "eps_comp_meV"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
