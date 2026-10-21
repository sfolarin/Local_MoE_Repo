#!/usr/bin/env python3
"""Interface-energy parity: MACE-MatPES-r2SCAN (x) versus Local-MoE (y).

Reads a per-interface CSV, computes agreement statistics with bootstrap
confidence intervals, and draws (a) a parity plot with y = x and (b) the signed
error against the reference gamma.  One or more model columns can be given;
each is drawn with its own colour AND marker, so the figure survives greyscale
and colour-vision deficiency.

Example (full model and common network side by side):
    python make_gamma_parity_figure.py \
        --rows-csv oishi_gamma_holdout40_rows.csv \
        --ref-col matpes_gamma_J_m2 \
        --pred "Local-MoE=localmoe_gamma_J_m2" \
        --pred "Common network=common_gamma_J_m2" \
        --label-col interface \
        --out-prefix gamma_parity_localmoe_vs_matpes

With no --ref-col / --pred the script tries to find the columns itself and, if
that is ambiguous, prints the column names and stops.  Nothing is dropped
silently: rows with a missing value are listed before they are excluded.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated colour-vision-safe categorical slots 1-2 (see the dataviz palette check).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]


def pick_columns(df: pd.DataFrame, ref_col, preds):
    cols = list(df.columns)
    if ref_col is None:
        cand = [c for c in cols if re.search(r"matpes|mace|ref", c, re.I) and re.search(r"gamma|γ|j_m2", c, re.I)]
        if len(cand) != 1:
            sys.exit(f"Cannot pick the reference column automatically. Use --ref-col. Columns: {cols}")
        ref_col = cand[0]
    if not preds:
        cand = [c for c in cols
                if c != ref_col and re.search(r"common|local|moe|full|c1b|pred", c, re.I)
                and re.search(r"gamma|γ|j_m2", c, re.I)]
        if not cand:
            sys.exit(f"Cannot pick the model column(s) automatically. Use --pred 'Label=column'. Columns: {cols}")
        preds = [f"{c}={c}" for c in cand]
    out = []
    for p in preds:
        label, _, col = p.partition("=")
        if not col:
            label, col = p, p
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Columns: {cols}")
        out.append((label, col))
    if ref_col not in df.columns:
        sys.exit(f"Column '{ref_col}' not found. Columns: {cols}")
    return ref_col, out


def stats(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict:
    e = y - x
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    mae_b = np.abs(e[idx]).mean(axis=1)
    bias_b = e[idx].mean(axis=1)
    rho_b = np.array([spearmanr(x[i], y[i]).statistic for i in idx])
    rho_b = rho_b[np.isfinite(rho_b)]
    slope, intercept = np.polyfit(x, y, 1)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return dict(
        N=n,
        MAE=float(np.abs(e).mean()), MAE_lo=ci(mae_b)[0], MAE_hi=ci(mae_b)[1],
        RMSE=float(np.sqrt((e ** 2).mean())),
        bias=float(e.mean()), bias_lo=ci(bias_b)[0], bias_hi=ci(bias_b)[1],
        n_over=int((e > 0).sum()), n_under=int((e < 0).sum()), n_exact=int((e == 0).sum()),
        pearson=float(np.corrcoef(x, y)[0, 1]),
        spearman=float(spearmanr(x, y).statistic), spearman_lo=ci(rho_b)[0], spearman_hi=ci(rho_b)[1],
        slope=float(slope), intercept=float(intercept),
        MAE_after_bias_removed=float(np.abs(e - e.mean()).mean()),
        MAE_const_predictor=float(np.abs(x - x.mean()).mean()),
        mean_ref=float(x.mean()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-csv", required=True)
    ap.add_argument("--ref-col")
    ap.add_argument("--pred", action="append", default=[], help="'Label=column' (repeatable, max 4)")
    ap.add_argument("--query", help="pandas query that selects the rows to plot, e.g. 'new_split == \"test\"'")
    ap.add_argument("--label-col", help="column with interface names, used to label the worst outliers")
    ap.add_argument("--label-worst", type=int, default=3, help="label this many largest |error| points of the first model")
    ap.add_argument("--ref-name", default="MACE-MatPES-r$^2$SCAN")
    ap.add_argument("--out-prefix", default="gamma_parity")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.rows_csv)
    if args.query:
        n_all = len(df)
        df = df.query(args.query, engine="python").reset_index(drop=True)
        print(f"--query {args.query!r}: kept {len(df)} of {n_all} rows")
    ref_col, preds = pick_columns(df, args.ref_col, args.pred)
    if len(preds) > 4:
        sys.exit("At most 4 models per figure.")

    need = [ref_col] + [c for _, c in preds]
    bad = df[df[need].isna().any(axis=1)]
    print(f"Loaded {len(df)} rows from {args.rows_csv}")
    print(f"reference = {ref_col};  models = {[f'{l} <- {c}' for l, c in preds]}")
    if len(bad):
        shown = bad[[args.label_col] + need] if args.label_col in df.columns else bad[need]
        print(f"{len(bad)} rows have a missing value and are EXCLUDED:\n{shown.to_string(index=False)}")
    d = df.drop(bad.index).reset_index(drop=True)
    x = d[ref_col].to_numpy(float)
    labels = d[args.label_col].astype(str).tolist() if args.label_col in d.columns else [str(i) for i in range(len(d))]

    all_stats = {}
    for label, col in preds:
        s = stats(x, d[col].to_numpy(float), args.bootstrap, args.seed)
        all_stats[label] = s
        print(f"\n{label}  (N={s['N']})")
        print(f"  MAE  {s['MAE']:.5f} J/m2  [95% CI {s['MAE_lo']:.5f}, {s['MAE_hi']:.5f}]    RMSE {s['RMSE']:.5f}")
        print(f"  bias {s['bias']:+.5f} J/m2  [{s['bias_lo']:+.5f}, {s['bias_hi']:+.5f}]   over/under/exact = {s['n_over']}/{s['n_under']}/{s['n_exact']}")
        print(f"  Pearson r {s['pearson']:.3f}   Spearman rho {s['spearman']:.3f}  [{s['spearman_lo']:.3f}, {s['spearman_hi']:.3f}]")
        print(f"  OLS y = {s['slope']:.3f} x {s['intercept']:+.4f}")
        print(f"  MAE with the mean bias removed {s['MAE_after_bias_removed']:.5f};  "
              f"MAE of always predicting the mean reference gamma {s['MAE_const_predictor']:.5f}")
    pd.DataFrame(all_stats).T.to_csv(f"{args.out_prefix}_stats.csv", index_label="model")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.05, 1]})

    ymax = max(float(d[c].max()) for _, c in preds)
    lim = 1.08 * max(float(x.max()), ymax)
    ax.plot([0, lim], [0, lim], color="0.35", lw=0.9, ls="--", zorder=1)
    ax.text(lim * 0.985, lim * 0.03, "dashed: y = x", color="0.35", fontsize=7, ha="right", va="bottom")

    for k, (label, col) in enumerate(preds):
        y = d[col].to_numpy(float)
        ax.scatter(x, y, s=22, marker=MARKERS[k], color=COLORS[k], edgecolor="white", linewidth=0.5,
                   alpha=0.95, zorder=3 + k, label=label)
        bx.scatter(x, y - x, s=22, marker=MARKERS[k], color=COLORS[k], edgecolor="white", linewidth=0.5,
                   alpha=0.95, zorder=3 + k, label=label)

    # label the largest-|error| interfaces of the first model in the error panel (panel a stays uncluttered)
    if args.label_worst > 0:
        e0 = d[preds[0][1]].to_numpy(float) - x
        for i in np.argsort(-np.abs(e0))[: args.label_worst]:
            bx.annotate(labels[i], (x[i], e0[i]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color="0.25")

    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{args.ref_name} $\\gamma$ (J m$^{{-2}}$)")
    ax.set_ylabel(f"{preds[0][0]} $\\gamma$ (J m$^{{-2}}$)" if len(preds) == 1 else "Model $\\gamma$ (J m$^{-2}$)")
    ax.set_title(f"(a) Interface energy, N = {len(x)}", loc="left", fontsize=8.5, fontweight="bold")
    ax.grid(True, color="0.9", lw=0.5, zorder=0)

    ax.set_axisbelow(True); bx.set_axisbelow(True)          # grid under the data and the zero line
    bx.axhline(0, color="0.35", lw=0.9, zorder=2)
    bx.set_xlabel(f"{args.ref_name} $\\gamma$ (J m$^{{-2}}$)")
    bx.set_ylabel("Model $-$ reference (J m$^{-2}$)")
    bx.set_title("(b) Signed error", loc="left", fontsize=8.5, fontweight="bold")
    bx.grid(True, color="0.9", lw=0.5, zorder=0)
    bx.margins(y=0.08)

    # Legend and statistics in one strip under the panels: marker + model name + numbers.
    glyph = ["●", "■", "▲", "◆"]          # same shapes as the markers
    row_h = 0.058
    fig.tight_layout(rect=[0, 0.045 + row_h * len(preds), 1, 1])
    for k, (label, _) in enumerate(preds):
        s = all_stats[label]
        yk = 0.02 + row_h * (len(preds) - 1 - k)
        fig.text(0.03, yk, glyph[k], color=COLORS[k], fontsize=9, va="bottom", ha="left")
        fig.text(0.055, yk, label, color=COLORS[k], fontsize=8, va="bottom", ha="left", fontweight="bold")
        fig.text(0.24, yk,
                 f"MAE {s['MAE']*1e3:.1f} mJ m$^{{-2}}$ (95% CI {s['MAE_lo']*1e3:.1f}–{s['MAE_hi']*1e3:.1f})"
                 f"   bias {s['bias']*1e3:+.1f} mJ m$^{{-2}}$"
                 f"   Pearson r {s['pearson']:.2f}"
                 f"   Spearman $\\rho$ {s['spearman']:.2f} ({s['spearman_lo']:.2f}–{s['spearman_hi']:.2f})",
                 color="0.15", fontsize=7.3, va="bottom", ha="left")

    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{args.out_prefix}.{ext}", dpi=300, bbox_inches="tight")
    print(f"\nWrote {args.out_prefix}.pdf/.svg/.png and {args.out_prefix}_stats.csv")


if __name__ == "__main__":
    main()
