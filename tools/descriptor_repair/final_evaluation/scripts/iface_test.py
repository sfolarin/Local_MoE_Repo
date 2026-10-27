#!/usr/bin/env python3
"""
iface_test.py -- paired test on the interface-atom force improvement, the one place a
directionally consistent signal survives on held-out data.

The frozen comparison reported, over the 22 test structures carrying region labels:

    full MoE c1b: lower on 19, higher on 3 of 22; mean change -0.00054 eV/A

19 of 22 is consistent in direction, but the effect size is about 0.3 % of a 0.168
eV/A base, and no p-value or interval was attached.  This script supplies them.

It uses the comparison script's own definition, read from its source rather than
reinvented, so the numbers here agree with the table printed beside them:

    per-structure region force RMSE = sqrt(sumsq_<region> / (3 * n_<region>))
    atom-weighted (AW)  = sqrt(sum_i sumsq_i / sum_i 3 n_i)
    structure-balanced (SB) = sqrt(mean_i (sumsq_i / (3 n_i)))
    labelled structures = those with n_interface > 0
    paired difference  = per-structure RMSE(MoE) - per-structure RMSE(common)

Before testing anything it REPRODUCES the published 19/3 split, the mean change, and
the AW/SB values in the regions CSV.  If those do not match, the definition has been
misread and no test result should be believed.

    python3 iface_test.py            # reads $REPAIR from the environment
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

BOOT = 20000
SEED = 20260811
MOE, COMMON = "c1b", "common_only"
REGIONS = ("interface", "phase1_bulk", "phase2_bulk")


def sign_test(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def boot_mean(d, rng):
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(a, b, rng, weights=None):
    """Bootstrap CI for the RMSE ratio; weights given -> atom-weighted."""
    n = len(a)
    idx = rng.integers(0, n, size=(BOOT, n))
    if weights is None:
        ra = np.sqrt(a[idx].mean(axis=1))
        rb = np.sqrt(b[idx].mean(axis=1))
    else:
        w = weights[idx]
        ra = np.sqrt((a[idx] * w).sum(axis=1) / w.sum(axis=1))
        rb = np.sqrt((b[idx] * w).sum(axis=1) / w.sum(axis=1))
    r = ra / rb
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> int:
    R = Path(os.environ["REPAIR"])
    ps = R / "final" / "compare" / "local_moe_vs_common_per_structure.csv"
    rg = R / "final" / "compare" / "local_moe_vs_common_regions.csv"
    rng = np.random.default_rng(SEED)

    d = pd.read_csv(ps)
    m = d[d.model == MOE].set_index("source_file").sort_index()
    c = d[d.model == COMMON].set_index("source_file").sort_index()
    if list(m.index) != list(c.index):
        print("FAIL: the two models cover different structures")
        return 1
    if not (m.n_atoms.values == c.n_atoms.values).all():
        print("FAIL: atom counts differ between models")
        return 1

    regions = pd.read_csv(rg).set_index("region")
    out = {"n_test_structures": int(len(m))}
    ok = True

    for region in REGIONS:
        nk, sk = "n_" + region, "sumsq_" + region
        mask = (m[nk].values > 0)
        if not (mask == (c[nk].values > 0)).all():
            print("FAIL: %s labelling differs between models" % region)
            return 1
        n = m.loc[mask, nk].values.astype(float)
        if not (n == c.loc[mask, nk].values.astype(float)).all():
            print("FAIL: %s atom counts differ between models" % region)
            return 1

        sm, sc = m.loc[mask, sk].values, c.loc[mask, sk].values
        mse_m, mse_c = sm / (3.0 * n), sc / (3.0 * n)        # per component
        rms_m, rms_c = np.sqrt(mse_m), np.sqrt(mse_c)        # per structure
        diff = rms_m - rms_c                                 # the script's quantity

        aw_m = math.sqrt(sm.sum() / (3.0 * n).sum())
        aw_c = math.sqrt(sc.sum() / (3.0 * n).sum())
        sb_m, sb_c = math.sqrt(mse_m.mean()), math.sqrt(mse_c.mean())

        print("\n" + "=" * 78)
        print("%s -- %d labelled test structures, %d atoms"
              % (region.upper(), mask.sum(), int(n.sum())))
        print("=" * 78)

        # ---- reproduce the published numbers before testing ------------------
        if region in regions.index:
            r = regions.loc[region]
            checks = [("common AW", aw_c, float(r["common AW"])),
                      ("common SB", sb_c, float(r["common SB"])),
                      ("MoE AW", aw_m, float(r["full MoE c1b AW"])),
                      ("MoE SB", sb_m, float(r["full MoE c1b SB"]))]
            worst = max(abs(a - b) for _, a, b in checks)
            print("  reproduction of the regions CSV: max |difference| = %.3e  %s"
                  % (worst, "ok" if worst < 1e-12 else "MISMATCH"))
            if worst >= 1e-12:
                for lbl, a, b in checks:
                    print("    %-10s recomputed %.12f  published %.12f" % (lbl, a, b))
                ok = False

        lower, higher = int((diff < 0).sum()), int((diff > 0).sum())
        print("  lower on %d, higher on %d of %d; mean change %+.5f eV/A, "
              "largest decrease %+.5f, largest increase %+.5f"
              % (lower, higher, len(diff), diff.mean(), diff.min(), diff.max()))

        # ---- the statistics that were missing --------------------------------
        p = sign_test(lower, higher)
        dlo, dhi = boot_mean(diff, rng)
        sb_lo, sb_hi = boot_ratio(mse_m, mse_c, rng)
        aw_lo, aw_hi = boot_ratio(mse_m, mse_c, rng, weights=3.0 * n)
        rel = 100.0 * (sb_m - sb_c) / sb_c

        print("  exact two-sided sign test        p = %.4g" % p)
        print("  mean paired change               %+.6f eV/A  95%% CI [%+.6f, %+.6f]  %s"
              % (diff.mean(), dlo, dhi,
                 "excludes 0" if dhi < 0 or dlo > 0 else "SPANS 0"))
        print("  SB RMSE   common %.5f -> MoE %.5f  (%+.2f %%)  ratio CI [%.4f, %.4f]  %s"
              % (sb_c, sb_m, rel, sb_lo, sb_hi,
                 "excludes 1.0" if not (sb_lo <= 1.0 <= sb_hi) else "SPANS 1.0"))
        print("  AW RMSE   common %.5f -> MoE %.5f           ratio CI [%.4f, %.4f]  %s"
              % (aw_c, aw_m, aw_lo, aw_hi,
                 "excludes 1.0" if not (aw_lo <= 1.0 <= aw_hi) else "SPANS 1.0"))

        out[region] = dict(
            n_structures=int(mask.sum()), n_atoms=int(n.sum()),
            lower=lower, higher=higher, sign_p=p,
            mean_paired_change=float(diff.mean()), mean_paired_change_ci=[dlo, dhi],
            sb_common=sb_c, sb_moe=sb_m, sb_rel_percent=rel,
            sb_ratio_ci=[sb_lo, sb_hi], sb_ratio_ci_spans_one=bool(sb_lo <= 1.0 <= sb_hi),
            aw_common=aw_c, aw_moe=aw_m,
            aw_ratio_ci=[aw_lo, aw_hi], aw_ratio_ci_spans_one=bool(aw_lo <= 1.0 <= aw_hi),
        )

    # ---- energy on the labelled subset, for context -------------------------
    mask = (m.n_interface.values > 0)
    em = m.loc[mask, "dE_eV_per_atom"].values
    ec = c.loc[mask, "dE_eV_per_atom"].values
    better, worse = int((em ** 2 < ec ** 2).sum()), int((em ** 2 > ec ** 2).sum())
    p = sign_test(better, worse)
    lo, hi = boot_ratio(em ** 2, ec ** 2, rng)
    print("\n" + "=" * 78)
    print("ENERGY on the same %d labelled structures (context)" % mask.sum())
    print("=" * 78)
    print("  RMSE  common %.5f -> MoE %.5f meV/atom"
          % (1000 * math.sqrt((ec ** 2).mean()), 1000 * math.sqrt((em ** 2).mean())))
    print("  MoE better on %d of %d; sign test p = %.4g; ratio CI [%.4f, %.4f]  %s"
          % (better, mask.sum(), p, lo, hi,
             "excludes 1.0" if not (lo <= 1.0 <= hi) else "SPANS 1.0"))
    out["energy_labelled_subset"] = dict(
        n=int(mask.sum()), better=better, worse=worse, sign_p=p,
        rmse_common_meV=1000 * math.sqrt((ec ** 2).mean()),
        rmse_moe_meV=1000 * math.sqrt((em ** 2).mean()),
        ratio_ci=[lo, hi], ratio_ci_spans_one=bool(lo <= 1.0 <= hi))

    dest = R / "final" / "iface_test.json"
    dest.write_text(json.dumps(dict(result=out, bootstrap=BOOT, seed=SEED,
                                    reproduced_published_numbers=bool(ok)), indent=2))
    print("\nwrote %s" % dest)
    print("REPRODUCTION:", "published numbers reproduced" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
