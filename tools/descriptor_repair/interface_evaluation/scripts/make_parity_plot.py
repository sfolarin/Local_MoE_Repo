#!/usr/bin/env python3
"""
make_parity_plot.py

Builds the GULP/SLC-vs-Oishi-published interfacial energy parity plot,
from the output of build_gulp_vs_oishi_table.py.

Reads the comparison CSV (pair, status, oishi_gamma_J_m2, gulp_gamma_J_m2,
max_drift_pct), reports exactly which of the requested pairs are
plottable vs excluded and why (never silently drops a row), then
produces a scatter plot of GULP gamma (y) vs Oishi's own published
gamma (x), with a y=x reference line and summary error statistics
(MAE, RMSE, Pearson r) annotated directly on the figure.

Usage:
    python make_parity_plot.py \
        --comparison-csv parity_plot_20_comparison.csv \
        --out parity_plot_20.png
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-csv", required=True)
    ap.add_argument("--out", default="parity_plot.png")
    ap.add_argument("--title", default="GULP/SLC vs. Oishi et al. (2025) Published Interfacial Energy")
    args = ap.parse_args()

    df = pd.read_csv(args.comparison_csv)
    print(f"Loaded {len(df)} pairs from {args.comparison_csv}")

    plottable = df[df["gulp_gamma_J_m2"].notna() & df["oishi_gamma_J_m2"].notna()].copy()
    excluded = df[~df.index.isin(plottable.index)]

    print(f"\n{len(plottable)} pairs plottable (real GULP result + real Oishi value)")
    if len(excluded):
        print(f"{len(excluded)} pairs excluded, by reason:")
        print(excluded[["pair", "status"]].to_string(index=False))

    if len(plottable) == 0:
        print("\nNothing plottable -- stopping without writing a figure.")
        return

    x = plottable["oishi_gamma_J_m2"].to_numpy()
    y = plottable["gulp_gamma_J_m2"].to_numpy()
    labels = plottable["pair"].tolist()

    mae = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))
    r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else float("nan")

    # On a small, near-zero-centered sample, Pearson r can be dominated by a
    # single geometric outlier -- reporting it alone can be misleading. Run
    # a leave-one-out check: find whichever single point's removal changes r
    # the most, and if that swing is large (sign flip or shift > 0.3), report
    # both values explicitly rather than one fragile number. This is a
    # generic, data-driven check -- it never hardcodes which pair to exclude.
    r_excl_label, r_excl_value, max_delta = None, None, 0.0
    if len(x) > 2:
        for i in range(len(x)):
            mask = np.ones(len(x), dtype=bool)
            mask[i] = False
            r_without_i = np.corrcoef(x[mask], y[mask])[0, 1]
            delta = abs(r_without_i - r)
            if delta > max_delta:
                max_delta = delta
                r_excl_label, r_excl_value = labels[i], r_without_i

    print(f"\nN = {len(x)}")
    print(f"MAE  = {mae:.6f} J/m^2")
    print(f"RMSE = {rmse:.6f} J/m^2")
    print(f"Pearson r = {r:.4f}")
    if r_excl_label is not None and max_delta > 0.3:
        print(f"  -> dominated by a single point: excluding {r_excl_label} "
              f"gives r = {r_excl_value:.4f} (N={len(x)-1})")

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.15
    lim = max(lim, 1e-4)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, alpha=0.6, label="y = x (perfect agreement)")

    ax.scatter(x, y, s=60, c="#2171b5", edgecolors="white", linewidths=0.8, zorder=3)

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=7, xytext=(4, 4),
                    textcoords="offset points", alpha=0.75)

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Oishi et al. published $\gamma$ (J/m$^2$)", fontsize=12)
    ax.set_ylabel(r"GULP / Sanders-Leslie-Catlow $\gamma$ (J/m$^2$)", fontsize=12)
    ax.set_title(args.title, fontsize=12)

    pearson_line = f"Pearson r = {r:.3f}"
    if r_excl_label is not None and max_delta > 0.3:
        pearson_line = (f"Pearson r = {r:.3f} (N={len(x)})\n"
                         f"      = {r_excl_value:.3f} excl. {r_excl_label} (N={len(x)-1})")
    stats_text = f"N = {len(x)}\nMAE = {mae:.5f} J/m$^2$\nRMSE = {rmse:.5f} J/m$^2$\n{pearson_line}"
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
