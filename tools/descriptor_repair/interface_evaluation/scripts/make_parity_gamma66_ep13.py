#!/usr/bin/env python3
"""
make_parity_gamma66_ep13.py -- the Local-MoE interface-energy parity figure.

Self-contained replacement for the original generator, which could not be recovered.
The script packaged alongside the first version of this deliverable,
make_parity_plot.py, targets the GULP/Oishi comparison and does not produce this figure;
it contains no CSV writer, so it cannot have written the _rows.csv and _stats.csv that
accompanied the figure either.  Rather than guess at the original, this reads the
committed per-family CSV and nothing else.

    gamma_MATPES_J_m2  (x)  vs  gamma_full_J_m2  (y),  all 66 families,
    split by new_split: 26 training as blue circles, 40 test as green squares.

Every annotation is computed here from the CSV and then CHECKED against the committed
summary CSV.  A disagreement beyond --tol aborts without writing a figure: a parity plot
whose printed statistics differ from the table it accompanies is worse than no plot.

It also prints the residual's effect on each metric -- full versus common-only, per
split, as both the absolute change and the percentage -- because that comparison is the
substantive result and should come from the data rather than from arithmetic done by
hand in a report.

CPU only.  No model, no checkpoint, no GPU.

    python3 make_parity_gamma66_ep13.py \
        --csv     Gamma66_EP13_Deliverable_v1/gamma66_ep13_v1_per_family.csv \
        --summary Gamma66_EP13_Deliverable_v1/gamma66_ep13_v1_summary.csv \
        --out-dir Gamma66_EP13_Deliverable_v1 --stem parity_gamma66_ep13_v1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF_LABEL = r"$\gamma_{\mathrm{MACE\text{-}MatPES\text{-}r^2SCAN\text{-}0}}$ (J m$^{-2}$)"
MOE_LABEL = r"$\gamma_{\mathrm{Local\text{-}MoE}}$ (J m$^{-2}$)"

TRAIN_COLOR = "#1f4e9c"      # blue circles
TEST_COLOR = "#2e8b4a"       # green squares

N_EXPECTED = {"all": 66, "train": 26, "test": 40}


def metrics(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    e = y - x
    d = {"N": len(e), "MAE": float(np.abs(e).mean()),
         "RMSE": float(np.sqrt((e ** 2).mean())), "bias": float(e.mean())}
    if len(e) >= 8 and np.ptp(x) > 0:
        d["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        d["slope"] = float(np.polyfit(x, y, 1)[0])
    else:
        d["pearson_r"] = float("nan")
        d["slope"] = float("nan")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="committed per-family CSV")
    ap.add_argument("--summary", default=None,
                    help="committed summary CSV; annotations are checked against it")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="parity_gamma66_ep13_v1")
    ap.add_argument("--tol", type=float, default=5e-6,
                    help="max allowed |difference| vs the committed summary")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the summary cross-check (not for published figures)")
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(Path(args.csv).expanduser().resolve())

    need = ["gamma_MATPES_J_m2", "gamma_full_J_m2", "gamma_common_part_J_m2", "new_split"]
    missing = [c for c in need if c not in d.columns]
    if missing:
        print("FAIL: the CSV is missing %s" % missing)
        return 1
    for c in need[:3]:
        d[c] = pd.to_numeric(d[c], errors="raise")
        if not np.isfinite(d[c].to_numpy(float)).all():
            print("FAIL: non-finite values in %s" % c)
            return 1

    groups = {"all": d, "train": d[d.new_split == "train"], "test": d[d.new_split == "test"]}
    for k, n in N_EXPECTED.items():
        if len(groups[k]) != n:
            print("FAIL: expected %d %s families, found %d -- the partition in this CSV "
                  "is not the frozen one" % (n, k, len(groups[k])))
            return 1
    print("66 families: 26 training, 40 test. All retained.")

    # ---- statistics, computed here ------------------------------------------
    full = {k: metrics(g.gamma_MATPES_J_m2, g.gamma_full_J_m2) for k, g in groups.items()}
    comm = {k: metrics(g.gamma_MATPES_J_m2, g.gamma_common_part_J_m2)
            for k, g in groups.items()}

    print("\nfull Local-MoE vs MACE-MatPES-r2SCAN-0, J/m^2")
    for k in ("all", "train", "test"):
        m = full[k]
        print("  %-5s n=%2d  MAE %.6f  RMSE %.6f  bias %+.6f  r %.4f  slope %.4f"
              % (k, m["N"], m["MAE"], m["RMSE"], m["bias"], m["pearson_r"], m["slope"]))

    # ---- cross-check against the committed summary ---------------------------
    if args.summary and not args.no_check:
        s = pd.read_csv(Path(args.summary).expanduser().resolve())
        col = {"MAE": "MAE", "RMSE": "RMSE", "bias": "bias", "pearson_r": "pearson_r"}
        bad = []
        for k in ("all", "train", "test"):
            row = s[(s.split == k) & (s.model == "full Local-MoE")]
            if len(row) != 1:
                bad.append("%s: %d matching summary row(s)" % (k, len(row)))
                continue
            row = row.iloc[0]
            for name, c in col.items():
                if c not in s.columns or not np.isfinite(float(row[c])):
                    continue
                dv = abs(float(row[c]) - full[k][name])
                if dv > args.tol:
                    bad.append("%s %s: summary %.8f vs recomputed %.8f (|d| %.2e)"
                               % (k, name, float(row[c]), full[k][name], dv))
        if bad:
            print("\nFAIL: the recomputed annotations disagree with the committed "
                  "summary; no figure written")
            for b in bad:
                print("  - %s" % b)
            return 1
        print("\nannotations agree with the committed summary to within %.1e" % args.tol)

    # ---- what the residual does, per metric ----------------------------------
    print("\nresidual effect (full vs common-only), the substantive comparison")
    eff_rows = []
    for k in ("all", "train", "test"):
        print("  %s" % k)
        for name, better in (("MAE", "lower"), ("RMSE", "lower"),
                             ("bias", "nearer zero"), ("pearson_r", "higher")):
            c, f = comm[k][name], full[k][name]
            pct = 100.0 * (f - c) / abs(c) if c else float("nan")
            if name == "pearson_r":
                verdict = "better" if f > c else "worse"
            elif name == "bias":
                verdict = "better" if abs(f) < abs(c) else "worse"
            else:
                verdict = "better" if f < c else "worse"
            print("    %-9s common %+.6f -> full %+.6f   %+6.2f%%   %s (%s is better)"
                  % (name, c, f, pct, verdict, better))
            eff_rows.append(dict(split=k, metric=name, common_only=c, full=f,
                                 change=f - c, pct_change=pct, verdict=verdict))
    pd.DataFrame(eff_rows).to_csv(out / f"{args.stem}_residual_metrics.csv", index=False)

    # ---- figure ---------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 11, "figure.dpi": 150, "savefig.dpi": 600,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.9,
    })
    fig, ax = plt.subplots(figsize=(4.6, 4.6))

    v = np.concatenate([d.gamma_MATPES_J_m2.to_numpy(float),
                        d.gamma_full_J_m2.to_numpy(float)])
    lo, hi = float(v.min()), float(v.max())
    pad = 0.08 * (hi - lo) if hi > lo else 0.01
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], "-", color="0.35", lw=0.9, zorder=1,
            label="_nolegend_")

    tr, te = groups["train"], groups["test"]
    ax.scatter(tr.gamma_MATPES_J_m2, tr.gamma_full_J_m2, s=38, marker="o",
               facecolors="none", edgecolors=TRAIN_COLOR, linewidths=1.25,
               zorder=3, label="Training")
    ax.scatter(te.gamma_MATPES_J_m2, te.gamma_full_J_m2, s=34, marker="s",
               facecolors="none", edgecolors=TEST_COLOR, linewidths=1.25,
               zorder=3, label="Test")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(REF_LABEL)
    ax.set_ylabel(MOE_LABEL)

    txt = "\n".join([
        r"$N$ = %d (26 training, 40 test)" % full["all"]["N"],
        "Training:  MAE %.4f   RMSE %.4f   $r$ %.3f"
        % (full["train"]["MAE"], full["train"]["RMSE"], full["train"]["pearson_r"]),
        "Test:        MAE %.4f   RMSE %.4f   $r$ %.3f"
        % (full["test"]["MAE"], full["test"]["RMSE"], full["test"]["pearson_r"]),
    ])
    ax.text(0.035, 0.965, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8.4, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="white",
                      edgecolor="0.75", linewidth=0.7))
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, edgecolor="0.75",
              fontsize=9.5, handletextpad=0.5, borderpad=0.6)
    ax.tick_params(direction="in", top=True, right=True, length=4, width=0.9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out / f"{args.stem}.{ext}"
        fig.savefig(p, dpi=600, bbox_inches="tight")
        print("saved %s" % p.name)
    plt.close(fig)
    print("wrote %s_residual_metrics.csv" % args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
