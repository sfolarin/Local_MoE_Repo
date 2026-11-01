#!/usr/bin/env python3
"""
finalize_gamma66_ep13.py -- complete the deliverable for the gamma66 epoch-13 evaluation.

eval_gamma66_fullmoe.py writes the per-family CSV and prints its statistics, but two
things the handoff asks for are missing: per-structure atom counts N_A and N_B (only
N_AB is written), and a machine-readable summary.  Both are derived from the existing
output rather than by editing the evaluation script, which must stay byte-identical to
what produced the numbers.

N_A and N_B are read back from the same geometry files the evaluation used, named in the
manifest, and the closure N_A + N_B == N_AB is re-checked here independently of the
check the evaluation already performed.  If any row fails, nothing is written.

The summary carries, for all / train / test and for both models:
    N, MAE, RMSE, signed bias, Pearson r, and the fitted slope
plus, per split, the residual's contribution to gamma and its effect on absolute error:
how many families it helped, hurt or left unchanged, and the mean change.

If the pre-repair stand-alone common evaluation is present it is included as a third
model, labelled as superseded.  The evaluation script compares against it and prints a
warning when they disagree; they disagree because that file predates the descriptor
repair, and the summary should record that rather than leave a reader to rediscover it.

    python3 finalize_gamma66_ep13.py --csv <gamma66_..._ep13_v1.csv> --out-dir <dir>
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-4          # the tolerance the evaluation script uses for "the residual helped"


def metrics(x, y):
    e = np.asarray(y) - np.asarray(x)
    d = {"N": len(e), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
         "bias": float(e.mean())}
    if len(e) >= 8 and np.ptp(x) > 0:
        d["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        d["slope"] = float(np.polyfit(x, y, 1)[0])
    else:
        d["pearson_r"] = np.nan
        d["slope"] = np.nan
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-family CSV from eval_gamma66_fullmoe.py")
    ap.add_argument("--manifest", default=None,
                    help="default: <External_Benchmark_Oishi2025>/gamma66_manifest.csv")
    ap.add_argument("--prerepair", default=None,
                    help="stand-alone common CSV that predates the descriptor repair")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="gamma66_ep13_v1")
    args = ap.parse_args()

    from ase.io import read as ase_read

    EXT = Path(os.environ["PROJ"]).parent / "External_Benchmark_Oishi2025"
    man_p = Path(args.manifest) if args.manifest else EXT / "gamma66_manifest.csv"
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(Path(args.csv).expanduser().resolve())
    man = pd.read_csv(man_p)
    assert len(d) == 66, "expected 66 families, found %d" % len(d)
    assert d.case.is_unique, "duplicate case in the evaluation output"

    # --- atom counts, read back from the geometries the evaluation used -------
    counts, bad = [], []
    for r in man.itertuples():
        na = len(ase_read(str(EXT / r.a_path), index=-1))
        nb = len(ase_read(str(EXT / r.b_path), index=-1))
        nab = len(ase_read(str(EXT / r.ab_path), index=-1))
        if na + nb != nab:
            bad.append((r.case, na, nb, nab))
        counts.append(dict(case=r.case, N_A=na, N_B=nb, N_AB_check=nab))
    if bad:
        print("FAIL: atom counts do not close; nothing written")
        for c, na, nb, nab in bad:
            print("  %-28s %d + %d != %d" % (c, na, nb, nab))
        return 1
    cdf = pd.DataFrame(counts)
    d = d.merge(cdf, on="case", how="left")
    mism = d[d.N_AB != d.N_AB_check]
    if len(mism):
        print("FAIL: N_AB in the evaluation output disagrees with the geometry files")
        print(mism[["case", "N_AB", "N_AB_check"]].to_string(index=False))
        return 1
    d = d.drop(columns=["N_AB_check"])
    print("atom counts verified on all %d families: N_A + N_B == N_AB" % len(d))

    cols = ["case", "origin", "pair", "new_split", "old_pool_split",
            "N_AB", "N_A", "N_B", "area_A2",
            "full_AB_eV", "full_A_eV", "full_B_eV",
            "common_part_AB_eV", "common_part_A_eV", "common_part_B_eV",
            "residual_part_AB_eV", "residual_part_A_eV", "residual_part_B_eV",
            "gamma_MATPES_J_m2", "gamma_full_J_m2", "gamma_common_part_J_m2",
            "gamma_residual_part_J_m2", "err_full", "err_common_part", "abs_err_change"]
    cols = [c for c in cols if c in d.columns] + [c for c in d.columns if c not in cols]
    d = d[cols]
    per_family = out / f"{args.stem}_per_family.csv"
    d.to_csv(per_family, index=False)

    # --- summary --------------------------------------------------------------
    series = [("full Local-MoE", "gamma_full_J_m2", "current"),
              ("common part only", "gamma_common_part_J_m2", "current")]
    pre = None
    if args.prerepair and Path(args.prerepair).is_file():
        pre = pd.read_csv(args.prerepair)[["case", "gamma_common_J_m2"]]
        d = d.merge(pre, on="case", how="left")
        series.append(("common, pre-repair descriptor", "gamma_common_J_m2", "superseded"))

    rows = []
    for split in ("all", "train", "test"):
        g = d if split == "all" else d[d.new_split == split]
        for name, col, status in series:
            if col not in g.columns or g[col].isna().all():
                continue
            rows.append(dict(split=split, model=name, status=status,
                             **metrics(g.gamma_MATPES_J_m2.values, g[col].values)))
    s = pd.DataFrame(rows)
    summary = out / f"{args.stem}_summary.csv"
    s.to_csv(summary, index=False)

    print("\nGAMMA vs MATPes-r2SCAN, J/m^2")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- what the residual actually did ---------------------------------------
    print("\nresidual contribution to gamma, and its effect on |error|")
    res = []
    for split in ("all", "train", "test"):
        g = d if split == "all" else d[d.new_split == split]
        ch = g.abs_err_change.to_numpy()
        r = dict(split=split, N=len(g),
                 residual_gamma_mean=float(g.gamma_residual_part_J_m2.mean()),
                 residual_gamma_mean_abs=float(g.gamma_residual_part_J_m2.abs().mean()),
                 residual_gamma_max_abs=float(g.gamma_residual_part_J_m2.abs().max()),
                 helped=int((ch < -TOL).sum()), hurt=int((ch > TOL).sum()),
                 unchanged=int((np.abs(ch) <= TOL).sum()),
                 mean_abs_err_change=float(ch.mean()))
        res.append(r)
        print("  %-5s n=%2d  residual gamma mean %+.4f  mean|.| %.4f  max|.| %.4f  |  "
              "helped %2d  hurt %2d  unchanged %2d  |  mean d|err| %+.5f"
              % (split, r["N"], r["residual_gamma_mean"], r["residual_gamma_mean_abs"],
                 r["residual_gamma_max_abs"], r["helped"], r["hurt"], r["unchanged"],
                 r["mean_abs_err_change"]))
    pd.DataFrame(res).to_csv(out / f"{args.stem}_residual_effect.csv", index=False)

    neg = int((d.gamma_full_J_m2 < 0).sum())
    print("\nfamilies retained: %d (none dropped); negative predicted gamma: %d" % (len(d), neg))
    print("identity max |gamma_full - gamma_common - gamma_residual| = %.2e J/m2"
          % (d.gamma_full_J_m2 - d.gamma_common_part_J_m2
             - d.gamma_residual_part_J_m2).abs().max())
    print("\nwrote %s" % per_family)
    print("      %s" % summary)
    print("      %s" % (out / f"{args.stem}_residual_effect.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
