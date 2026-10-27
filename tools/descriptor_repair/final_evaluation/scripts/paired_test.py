#!/usr/bin/env python3
"""
paired_test.py -- is the full Local-MoE distinguishable from the common network on
the held-out test partition?

The frozen evaluation found, on 105 held-out configurations:

    energy RMSE   common 2.4603  ->  Local-MoE 2.5735 meV/atom   (+4.6 %, worse)
    force SB-RMSE common 0.05765 ->  Local-MoE 0.05765 eV/A      ( 0.0 %)

An RMSE ratio on 105 samples is a point estimate with no error bar, so "+4.6 % worse"
could be a real regression or resampling noise.  Both readings lead to different
papers, so the difference needs a test rather than a sentence.

Because both models were evaluated on IDENTICAL configurations, the comparison is
paired, which is far more sensitive than comparing two independent RMSEs.  This
script runs, per partition and for energy and forces separately:

  * an exact two-sided sign test on how many structures each model fits better.
    Distribution-free; makes no assumption about the error distribution.
  * a bootstrap 95 % CI over structures for the RMSE ratio (MoE / common).  A CI
    containing 1.0 means the partition cannot distinguish the two models.
  * the paired mean difference in squared error, with a bootstrap CI, which is the
    quantity RMSE is built from.

It first proves the pairing is real: the record id sets must be equal, and the
REFERENCE energies and forces must be bit-identical between the two export paths.
If they are not, the two sets are not the same configurations and no comparison is
meaningful.

    python3 paired_test.py                 # reads $REPAIR from the environment
"""
import json
import math
import os
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")
BOOT = 20000
SEED = 20260811


def load(raw_root: Path, split: str) -> dict:
    out = {}
    for f in sorted((raw_root / split).glob("*.npz")):
        with np.load(f, allow_pickle=False) as d:
            rid = str(d["record_id"])
            n = int(d["n_atoms"])
            fr = np.asarray(d["reference_forces"], float)
            fp = np.asarray(d["predicted_forces"], float)
            out[rid] = dict(
                n=n,
                e_ref=float(d["E_reference_eV"]),
                de=(float(d["E_predicted_eV"]) - float(d["E_reference_eV"])) / n,
                fmse=float(np.mean((fp - fr) ** 2)),
                f_ref_sum=float(np.abs(fr).sum()),
            )
    return out


def sign_test(better: int, worse: int) -> float:
    """Exact two-sided binomial p under p=0.5, ties excluded."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def boot_ratio(a: np.ndarray, b: np.ndarray, rng) -> tuple:
    """Bootstrap CI for sqrt(mean a)/sqrt(mean b) over paired structures."""
    n = len(a)
    idx = rng.integers(0, n, size=(BOOT, n))
    ra = np.sqrt(a[idx].mean(axis=1))
    rb = np.sqrt(b[idx].mean(axis=1))
    r = ra / rb
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def boot_mean(d: np.ndarray, rng) -> tuple:
    n = len(d)
    idx = rng.integers(0, n, size=(BOOT, n))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    R = Path(os.environ["REPAIR"])
    moe_root = R / "final" / "moe" / "raw_predictions"
    com_root = R / "pilot" / "common" / "raw_predictions"

    rng = np.random.default_rng(SEED)
    report = {}
    overall_ok = True

    for split in SPLITS:
        moe, com = load(moe_root, split), load(com_root, split)

        # ---- the pairing must be real before anything is compared --------------
        if set(moe) != set(com):
            print("%s: FAIL -- record id sets differ (%d vs %d, %d common)"
                  % (split, len(moe), len(com), len(set(moe) & set(com))))
            overall_ok = False
            continue
        ids = sorted(moe)
        bad = [r for r in ids
               if moe[r]["n"] != com[r]["n"]
               or moe[r]["e_ref"] != com[r]["e_ref"]
               or moe[r]["f_ref_sum"] != com[r]["f_ref_sum"]]
        if bad:
            print("%s: FAIL -- %d records differ in reference data, e.g. %s"
                  % (split, len(bad), bad[:3]))
            overall_ok = False
            continue

        de_m = np.array([moe[r]["de"] for r in ids])
        de_c = np.array([com[r]["de"] for r in ids])
        fm_m = np.array([moe[r]["fmse"] for r in ids])
        fm_c = np.array([com[r]["fmse"] for r in ids])

        print("\n" + "=" * 78)
        print("%s -- %d configurations, pairing verified "
              "(identical ids, atom counts and reference data)" % (split.upper(), len(ids)))
        print("=" * 78)

        block = {"n": len(ids)}
        for label, m, c, unit, scale in (
            ("energy |dE| per atom", de_m ** 2, de_c ** 2, "meV/atom", 1000.0),
            ("force RMSE per component", fm_m, fm_c, "eV/A", 1.0),
        ):
            rm, rc = scale * math.sqrt(m.mean()), scale * math.sqrt(c.mean())
            better = int((m < c).sum())
            worse = int((m > c).sum())
            ties = len(ids) - better - worse
            p = sign_test(better, worse)
            lo, hi = boot_ratio(m, c, rng)
            dlo, dhi = boot_mean(m - c, rng)
            dmean = float((m - c).mean())
            spans1 = lo <= 1.0 <= hi

            print("\n  %s" % label)
            print("    common %.5f  ->  Local-MoE %.5f  %s   ratio %.4f"
                  % (rc, rm, unit, rm / rc))
            print("    Local-MoE better on %d of %d structures (worse on %d, tied %d)"
                  % (better, len(ids), worse, ties))
            print("    exact two-sided sign test      p = %.4g" % p)
            print("    bootstrap 95%% CI on the ratio  [%.4f, %.4f]  %s"
                  % (lo, hi, "SPANS 1.0 -- not distinguishable"
                     if spans1 else "excludes 1.0 -- distinguishable"))
            print("    paired mean d(squared error)   %+.4e  95%% CI [%+.4e, %+.4e]"
                  % (dmean, dlo, dhi))

            block[label] = dict(
                common=rc, local_moe=rm, unit=unit, ratio=rm / rc,
                better=better, worse=worse, ties=ties, sign_p=p,
                ratio_ci=[lo, hi], ratio_ci_spans_one=bool(spans1),
                paired_mean_dsq=dmean, paired_mean_dsq_ci=[dlo, dhi],
            )
        report[split] = block

    out = R / "final" / "paired_test.json"
    out.write_text(json.dumps(dict(report=report, bootstrap=BOOT, seed=SEED,
                                   pairing_verified=bool(overall_ok)), indent=2))
    print("\nwrote %s" % out)
    print("PAIRING:", "verified on every partition" if overall_ok else "FAILED")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
