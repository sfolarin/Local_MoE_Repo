#!/usr/bin/env python3
"""
plot_computational_cost_holdout40.py

Three-panel cost figure for the holdout-40 benchmark (benchmark_cost_holdout40_nl.py):
  (a) time of one energy + force evaluation, (b) speedup, (c) peak allocated GPU memory,
all against the number of atoms, for the 105 holdout-40 TEST structures.

Series: MATPes-r2SCAN in float64 (the precision of the earlier benchmark and of the labels) and
the Local-MoE model (--local full, the default, or --local common for the common network alone;
their costs differ by a few percent). --with-float32 adds MATPes in float32 as a second reference.

One marker per interface family and atom count: records of the same family and size cost the
same, so they are averaged into one marker (--keep-records plots every record instead). The
within-group spread is printed so the caption can quote it. All statistics and fits refer to the
plotted markers. Structures that ran out of memory in any plotted series are listed and left out.

Plain style: periodic-image cells are not marked unless --mark-images is given. --drop-images leaves
them out of the figure and of every statistic (cells whose thinnest plane spacing is <= 2 r_cut = 10 A,
where TorchSOAP enumerates periodic images); the excluded cells and their cost are printed so that the
caption can state the exclusion.

Run, for example:
    python plot_computational_cost_holdout40.py --raw $PROJ/eval_cost_v1/computational_cost_raw.csv \
        --stem computational_cost_holdout40_float64

Outputs (next to the raw CSV): <stem>.png / .pdf / _plot_data.csv.
With --paper: <stem>_paper.png (600 dpi), .pdf, and .svg (vector).
The paper layout changes presentation only: the grouping, exclusions, statistics,
and fitting ranges are identical to the original benchmark analysis.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="computational_cost_raw.csv written by benchmark_cost_holdout40_nl.py")
ap.add_argument("--local", default="full", choices=["full", "common"], help="which Local-MoE series to plot")
ap.add_argument("--max-atoms", type=int, default=None, help="optional cap on the atom count (default: none)")
ap.add_argument("--with-float32", action="store_true", help="also show MATPes in float32 (default: float64 only)")
ap.add_argument("--stem", default="computational_cost_holdout40_float64")
ap.add_argument("--drop-images", action="store_true",
                help="leave out cells where Local-MoE enumerates periodic images (the exclusion is printed)")
ap.add_argument("--mark-images", action="store_true",
                help="mark cells where Local-MoE enumerates periodic images (thinnest plane spacing <= 2 r_cut)")
ap.add_argument("--paper", action="store_true",
                help="journal layout: float32 is the headline comparison (--with-float32 is switched on), log atom axis, "
                     "one shared model legend, compact in-panel scaling keys, "
                     "MACE-MatPES-r2SCAN labels; PDF/SVG/600-dpi PNG as <stem>_paper.*. Also prints the numbers for the text")
ap.add_argument("--figsize", type=float, nargs=2, default=[7.2, 3.3], metavar=("W", "H"),
                help="figure size in inches for --paper (default 7.2 3.3: full journal page width)")
ap.add_argument("--keep-records", action="store_true",
                help="plot every record; default merges records of the same family and atom count")
args = ap.parse_args()
assert not (args.drop_images and args.mark_images), "--drop-images and --mark-images cannot be combined"

RAW = Path(args.raw)
OUT = RAW.parent
LOC = args.local
LOC_NAME = "Local-MoE" if LOC == "full" else "Local-MoE (common network)"
if args.paper:
    args.with_float32 = True          # the paper figure compares against MACE in float32 (precision-matched) and float64
F32 = args.with_float32
PAPER = args.paper
N64 = "MATPes f64" if F32 else "MATPes"
C64, C32, CL, CI = "#1f77b4", "#9467bd", "#ff7f0e", "#d62728"

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 13, "font.weight": "bold",
    "axes.labelsize": 16, "axes.labelweight": "bold", "axes.linewidth": 1.4,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 6, "ytick.major.size": 6, "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.fontsize": 9, "legend.framealpha": 0.95,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# load and reshape: one row per record
# ---------------------------------------------------------------------------
raw = pd.read_csv(RAW)
need = {"record_id", "phase_pair", "n_atoms", "model_key", "status", "median_s", "peak_gpu_allocated_MiB",
        "images_enumerated"}
assert not (need - set(raw.columns)), f"missing columns: {sorted(need - set(raw.columns))}"
raw["images_enumerated"] = raw["images_enumerated"].astype(str).str.strip().str.lower().eq("true")

print("rows per model and status:")
print(raw.groupby(["model_key", "status"]).size().unstack(fill_value=0).to_string())
ok = raw[raw.status == "ok"].copy()
keys = ["matpes64", LOC] + (["matpes32"] if F32 else [])
for k in keys:
    assert (ok.model_key == k).any(), f"no finished rows for model_key '{k}'"
idx = ["record_id", "phase_pair", "n_atoms", "images_enumerated"]
t = ok.pivot(index=idx, columns="model_key", values="median_s")
p = ok.pivot(index=idx, columns="model_key", values="peak_gpu_allocated_MiB")
cols = {"t64": t["matpes64"], "tL": t[LOC], "m64": p["matpes64"], "mL": p[LOC]}
if F32:
    cols.update(t32=t["matpes32"], m32=p["matpes32"])
d = pd.DataFrame(cols).reset_index()
COLS = [c for c in ("t64", "t32", "tL", "m64", "m32", "mL") if c in d.columns]
n_all = len(d)
lost = d[d[COLS].isna().any(axis=1)]
if len(lost):
    print(f"left out (a series is missing or ran out of memory): {len(lost)} records, N = {sorted(lost.n_atoms.tolist())}")
d = d.dropna().reset_index(drop=True)
if args.max_atoms:
    dropped = sorted(d.loc[d.n_atoms > args.max_atoms, "n_atoms"].tolist())
    d = d[d.n_atoms <= args.max_atoms].reset_index(drop=True)
    print(f"cap N <= {args.max_atoms}: dropped sizes {dropped}")
if args.drop_images:
    ex = d[d.images_enumerated]
    if len(ex):
        print(f"EXCLUDED (--drop-images): {len(ex)} records with periodic-image enumeration, N = {sorted(ex.n_atoms.tolist())}; "
              f"{LOC_NAME} time / MATPes float64 time over these: median {(ex.tL / ex.t64).median():.2f}, "
              f"min {(ex.tL / ex.t64).min():.2f}, max {(ex.tL / ex.t64).max():.2f}  "
              f"(speedup vs float64: median {(ex.t64 / ex.tL).median():.2f}, min {(ex.t64 / ex.tL).min():.2f}, "
              f"max {(ex.t64 / ex.tL).max():.2f}); memory ratio {LOC_NAME}/MATPes float64: "
              f"median {(ex.mL / ex.m64).median():.2f}")
    else:
        print("EXCLUDED (--drop-images): no record enumerates periodic images")
    d = d[~d.images_enumerated]
d = d.sort_values(["n_atoms", "record_id"]).reset_index(drop=True)
print(f"{len(d)} of {n_all} records complete in all plotted series (N from {d.n_atoms.min()} to {d.n_atoms.max()})")

# ---------------------------------------------------------------------------
# one marker per (family, atom count)
# ---------------------------------------------------------------------------
d["n_records"] = 1
if not args.keep_records:
    rows, spreads = [], []
    for (fam, n), x in d.groupby(["phase_pair", "n_atoms"], sort=False):
        assert x.images_enumerated.nunique() == 1, f"records of {fam} (N={n}) disagree on the periodic-image flag"
        r = {"phase_pair": fam, "n_atoms": int(n), "n_records": len(x), "images_enumerated": bool(x.images_enumerated.iloc[0])}
        for c in COLS:
            r[c] = float(x[c].mean())
        if len(x) > 1:
            spreads.append({"family": fam, "N": int(n), "records": len(x),
                            **{c: 100.0 * (x[c].max() - x[c].min()) / x[c].mean() for c in COLS}})
        rows.append(r)
    n_rec = len(d)
    d = pd.DataFrame(rows).sort_values(["n_atoms", "phase_pair"]).reset_index(drop=True)
    print(f"merged {n_rec} records into {len(d)} markers (one per family and atom count; {d.phase_pair.nunique()} families)")
    if spreads:
        sp = pd.DataFrame(spreads)
        print("largest within-group spread (max-min)/mean in %: " + ", ".join(f"{c} {sp[c].max():.2f}" for c in COLS))

d["sp64"] = d.t64 / d.tL
d["red64"] = 100.0 * (1.0 - d.mL / d.m64)
if F32:
    d["sp32"] = d.t32 / d.tL
    d["red32"] = 100.0 * (1.0 - d.mL / d.m32)
d.to_csv(OUT / f"{args.stem}_plot_data.csv", index=False)


# ---------------------------------------------------------------------------
# numbers for the text (plotted markers)
# ---------------------------------------------------------------------------
def powerfit(n, y, nmin):
    s = (n >= nmin).to_numpy()
    k, c = np.polyfit(np.log(n[s].to_numpy(float)), np.log(y[s].to_numpy(float)), 1)
    return k, c, int(s.sum())


SER = [("64", "t64", "m64"), ("L", "tL", "mL")] + ([("32", "t32", "m32")] if F32 else [])
fits_t = {k: powerfit(d.n_atoms, d[tc] * 1e3, 1000) for k, tc, mc in SER}
fits_m = {k: powerfit(d.n_atoms, d[mc] / 1024.0, 800) for k, tc, mc in SER}
REFS = [("MATPes float64", "64")] + ([("MATPes float32", "32")] if F32 else [])

print(f"\nplotted markers: {len(d)}")
for ref, k in REFS:
    s_, r_ = d[f"sp{k}"], d[f"red{k}"]
    print(f"speedup vs {ref}: mean {s_.mean():.2f}  median {s_.median():.2f}  min {s_.min():.2f}  max {s_.max():.2f}"
          f" | memory reduction median {r_.median():.1f}%  min {r_.min():.1f}%  max {r_.max():.1f}%")
print(f"time exponents (N>=1000, {fits_t['L'][2]} markers): " + ", ".join(
    f"{nm} {fits_t[k][0]:.2f}" for nm, k in [("MATPes f64", "64")] + ([("MATPes f32", "32")] if F32 else []) + [(LOC_NAME, "L")]))
print(f"memory exponents (N>=800, {fits_m['L'][2]} markers): " + ", ".join(
    f"{nm} {fits_m[k][0]:.2f}" for nm, k in [("MATPes f64", "64")] + ([("MATPes f32", "32")] if F32 else []) + [(LOC_NAME, "L")]))

bins = [0, 1000, 2000, 3000, np.inf]
names = ["N < 1000", "1000 <= N < 2000", "2000 <= N < 3000", "N >= 3000"]
d["bin"] = pd.cut(d.n_atoms, bins=bins, labels=names, right=False)
print("\nby size (median over markers):")
hdr = f"{'bin':18s} {'n':>3s} {'speedup f64':>12s} {'mem red f64 %':>14s}" + (f" {'speedup f32':>12s} {'mem red f32 %':>14s}" if F32 else "")
print(hdr)
for nm in names:
    x = d[d.bin == nm]
    if len(x):
        line = f"{nm:18s} {len(x):3d} {x.sp64.median():12.2f} {x.red64.median():14.1f}"
        if F32:
            line += f" {x.sp32.median():12.2f} {x.red32.median():14.1f}"
        print(line)
print("\nspeedup and cost at fixed sizes (nearest plotted marker):")
for tgt in (100, 500, 1000, 2000, int(d.n_atoms.max())):
    r = d.loc[(np.log(d.n_atoms) - np.log(tgt)).abs().idxmin()]
    line = f"  N ~ {tgt:5d} (marker N = {int(r.n_atoms):5d}): {LOC_NAME} {r.tL * 1e3:6.1f} ms | MATPes f64 {r.t64 * 1e3:7.1f} ms ({r.sp64:5.1f}x)"
    if F32:
        line += f" | MATPes f32 {r.t32 * 1e3:7.1f} ms ({r.sp32:5.1f}x)"
    line += f" | memory {LOC_NAME} {r.mL / 1024.0:6.3f} GiB, f64 {r.m64 / 1024.0:6.2f} GiB" + (f", f32 {r.m32 / 1024.0:6.2f} GiB" if F32 else "")
    print(line)
print("\nfacts for the caption:")
if "gpu" in raw.columns:
    print("  GPU:", ", ".join(sorted(map(str, raw.gpu.dropna().unique()))))
if "precision" in raw.columns:
    print("  precision per series:", "; ".join(f"{k}: {v}" for k, v in raw.groupby("model_key").precision.first().items()))
if "n_repeat" in raw.columns:
    print(f"  repeats per structure after warm-up: {int(ok.n_repeat.min())} to {int(ok.n_repeat.max())} (median reported)")
print(f"  {len(d)} plotted markers, N from {d.n_atoms.min()} to {d.n_atoms.max()}, {d.phase_pair.nunique()} interface families; "
      f"periodic-image cells: {int(d.images_enumerated.sum())} markers {'(marked)' if args.mark_images else '(included, unmarked)' if not args.drop_images else '(excluded)'}")
print("\nper-structure checks (plotted markers):")
for ref, k in REFS:
    sl = d[d[f"sp{k}"] < 1]
    mo = d[d.mL > d[f"m{k}"]]
    print(f"  {LOC_NAME} slower than {ref}: {len(sl)}" + (f" (N = {sorted(sl.n_atoms.tolist())})" if len(sl) else ""))
    print(f"  {LOC_NAME} uses more memory than {ref}: {len(mo)}" + (f" (N = {sorted(mo.n_atoms.tolist())})" if len(mo) else ""))
img = d[d.images_enumerated]
noimg = d[~d.images_enumerated]
print(f"markers with periodic-image enumeration: {len(img)} (N = {sorted(img.n_atoms.tolist())})")
print(f"  median speedup vs float64: with images {img.sp64.median():.2f}, without images {noimg.sp64.median():.2f}" if len(img) else "")
print(f"  markers without periodic images: speedup vs float64 mean {noimg.sp64.mean():.2f}, median {noimg.sp64.median():.2f}, "
      f"min {noimg.sp64.min():.2f} (N = {int(noimg.loc[noimg.sp64.idxmin(), 'n_atoms'])})")
fits_ni = powerfit(noimg.n_atoms, noimg.tL * 1e3, 1000), powerfit(noimg.n_atoms, noimg.mL / 1024.0, 800)
print(f"  markers without periodic images: {LOC_NAME} exponents time (N>=1000) {fits_ni[0][0]:.2f}, memory (N>=800) {fits_ni[1][0]:.2f}")

# ---------------------------------------------------------------------------
# paper figure (--paper)
# ---------------------------------------------------------------------------
def paper_figure():
    """Full-width journal figure; retain every plotted marker and the existing fits."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MaxNLocator

    # Color and shape both distinguish the models. The f64 reference remains
    # lighter than the primary f32 comparison, but is visible at printed size.
    colors = {"64": "#0072B2", "32": "#7B4EA3", "L": "#D55E00", "I": "#C23B42"}
    rc = {
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 8, "font.weight": "normal",
        "axes.labelsize": 9, "axes.labelweight": "bold", "axes.linewidth": 0.9,
        "axes.labelpad": 5, "axes.edgecolor": "0.12",
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "xtick.major.pad": 4, "ytick.major.pad": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    }
    with plt.rc_context(rc):
        ref_label = r"MACE-MatPES-r$^{2}$SCAN"
        local_label = "Local-MoE (full)" if LOC == "full" else "Local-MoE (common)"
        nmin_, nmax_ = float(d.n_atoms.min()), float(d.n_atoms.max())
        xlo, xhi = nmin_ / 1.18, nmax_ * 1.22
        base_ = d if not args.mark_images else d[~d.images_enumerated]
        imgs_ = d[d.images_enumerated] if args.mark_images else d.iloc[0:0]
        styles = {
            "64": dict(marker="o", s=18, c=colors["64"], alpha=0.58, zorder=3),
            "32": dict(marker="^", s=23, c=colors["32"], alpha=0.88, zorder=4),
            "L": dict(marker="s", s=18, c=colors["L"], alpha=0.92, zorder=5),
            "I": dict(marker="D", s=23, c=colors["I"], alpha=0.95, zorder=6),
        }
        for style in styles.values():
            style.update(edgecolors="white", linewidths=0.25)

        fig, axes = plt.subplots(1, 3, figsize=tuple(args.figsize))
        # Fixed margins preserve the requested physical width in every export.
        # A second legend row is reserved only when periodic images are marked.
        fig.subplots_adjust(left=0.082, right=0.972, bottom=0.19,
                            top=0.80 if len(imgs_) else 0.84, wspace=0.48)

        def cost_points(ax, prefix, factor):
            for key, suffix in (("64", "64"), ("32", "32")):
                ax.scatter(d.n_atoms, d[prefix + suffix] * factor, **styles[key])
            ax.scatter(base_.n_atoms, base_[prefix + "L"] * factor, **styles["L"])
            if len(imgs_):
                ax.scatter(imgs_.n_atoms, imgs_[prefix + "L"] * factor, **styles["I"])

        def log_y(ax, values):
            lo, hi = float(np.min(values)), float(np.max(values))
            ax.set_yscale("log")
            ax.set_ylim(lo / 1.22, hi * 1.38)
            # Label 1, 2, 5 per decade for time; memory uses decades plus 5
            # when its dynamic range is large, to avoid an overfull tick axis.
            multipliers = (1, 2, 5) if hi / lo < 100 else (1, 5)
            decades = range(int(np.floor(np.log10(lo))) - 1,
                            int(np.ceil(np.log10(hi))) + 2)
            ticks = [m * 10.0**e for e in decades for m in multipliers
                     if lo / 1.22 <= m * 10.0**e <= hi * 1.38]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(NullFormatter())

        def scaling_key(ax, fits, threshold):
            handles = []
            for key in ("32", "64", "L"):
                k, c, count = fits[key]
                if count < 2 or not np.isfinite(k) or not np.isfinite(c) or nmax_ < threshold:
                    continue
                xx = np.geomspace(threshold, nmax_, 100)
                ax.plot(xx, np.exp(c) * xx**k, color=colors[key],
                        lw=1.0, ls=(0, (3, 2)), alpha=0.95, zorder=2)
                handles.append(Line2D([], [], color=colors[key], lw=1.0,
                                      ls=(0, (3, 2)), label=rf"$N^{{{k:.2f}}}$"))
            if handles:
                # The exponents sit in the empty upper-left area, so no extra
                # atom-axis padding is needed and data/labels cannot collide.
                leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.025, 0.985),
                                title=rf"Fits ($N\geq {threshold}$)", frameon=False,
                                fontsize=7.5, title_fontsize=7.1, labelspacing=0.25,
                                handlelength=1.7, handletextpad=0.45, borderaxespad=0)
                leg.set_zorder(8)

        # (a) Evaluation time, logarithmic in both N and time.
        ax = axes[0]
        cost_points(ax, "t", 1e3)
        log_y(ax, np.concatenate([d.t64, d.t32, d.tL]) * 1e3)
        scaling_key(ax, fits_t, 1000)
        ax.set_ylabel("Energy + force time (ms)")

        # (b) Ratios use the same averaged markers as the original script.
        ax = axes[1]
        for key in ("64", "32"):
            ax.scatter(base_.n_atoms, base_["sp" + key], **styles[key])
        if len(imgs_):
            # Hollow circles / filled triangles retain the reference identity
            # while the red color identifies periodic-image records.
            for key in ("64", "32"):
                style = {**styles[key], "alpha": 0.95}
                if key == "64":
                    style.update(c="none", edgecolors=colors["I"], linewidths=0.8)
                else:
                    style.update(c=colors["I"])
                ax.scatter(imgs_.n_atoms, imgs_["sp" + key], **style)
        ax.axhline(1.0, color="0.45", lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.annotate(r"$1\times$", xy=(0.97, 1.0), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=7, color="0.35")
        ax.set_ylim(0, float(pd.concat([d.sp64, d.sp32]).max()) * 1.12)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=3))
        ax.set_ylabel(r"Local-MoE speedup ($\times$)")
        ax.text(0.055, 0.96, r"$t_{\mathrm{MACE}}\,/\,t_{\mathrm{MoE}}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="0.3")

        # (c) Allocated memory (MiB -> GiB, unchanged).
        ax = axes[2]
        cost_points(ax, "m", 1.0 / 1024.0)
        log_y(ax, np.concatenate([d.m64, d.m32, d.mL]) / 1024.0)
        scaling_key(ax, fits_m, 800)
        ax.set_ylabel("Peak GPU allocation (GiB)")

        # Fewer major x labels keep all three panels readable at final size.
        # This is still a log axis; unlabelled minor ticks show its spacing.
        exponents = range(int(np.floor(np.log10(xlo))) - 1,
                          int(np.ceil(np.log10(xhi))) + 1)
        xticks = [10.0**e for e in exponents if xlo <= 10.0**e <= xhi]
        # Add a high-end tick when the data extend well beyond the last decade.
        # Adjacent labels always differ by at least a factor of three.
        upper_tick = 5 * 10.0**np.floor(np.log10(xhi))
        if xlo <= upper_tick <= xhi and (not xticks or upper_tick / xticks[-1] >= 3):
            xticks.append(upper_tick)
        for ax, tag_ in zip(axes, ("(a)", "(b)", "(c)")):
            ax.set_xscale("log")
            ax.set_xlim(xlo, xhi)
            ax.xaxis.set_major_locator(FixedLocator(xticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("Number of atoms, $N$")
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight("medium")
            ax.text(-0.015, 1.035, tag_, transform=ax.transAxes, fontsize=10.5,
                    fontweight="bold", va="bottom", ha="left")

        def proxy(key, label):
            style = styles[key]
            return Line2D([], [], marker=style["marker"], ls="", color=style["c"],
                          alpha=1.0, markersize=4.6, markeredgewidth=0.3,
                          markeredgecolor="white", label=label)

        handles = [proxy("32", ref_label + " (float32)"),
                   proxy("64", ref_label + " (float64)"), proxy("L", local_label)]
        if len(imgs_):
            handles.append(proxy("I", "Periodic-image cells"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.99),
                   ncol=2 if len(imgs_) else 3, frameon=False, fontsize=7.5,
                   handlelength=0.9, handletextpad=0.45, columnspacing=1.6,
                   labelspacing=0.5, borderaxespad=0)

        out = args.stem + "_paper"
        for extension in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"{out}.{extension}", dpi=600, facecolor="white")
        plt.close(fig)
        print("\nsaved (paper layout):", " | ".join(str(OUT / f"{out}.{ext}")
                                                    for ext in ("pdf", "svg", "png")))


if PAPER:
    paper_figure()
    sys.exit(0)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
nmax = d.n_atoms.max()
base = d if not args.mark_images else d[~d.images_enumerated]
imgs = d[d.images_enumerated] if args.mark_images else d.iloc[0:0]


def tag(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom", ha="left")


def guide(ax, fit, nmin, color, label):
    k, c, _ = fit
    x = np.linspace(nmin, nmax, 50)
    ax.plot(x, np.exp(c) * x ** k, "--", color=color, lw=1.2, alpha=0.8, zorder=2,
            label=rf"{label} fit $\propto N^{{{k:.2f}}}$")


def logticks(ax, lo, hi, cands):
    ax.set_yscale("log")
    ax.set_ylim(lo / 1.25, hi * 1.35)
    ticks = [v for v in cands if lo / 1.25 <= v <= hi * 1.35]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


# (a) time
ax = axes[0]
ax.scatter(d.n_atoms, d.t64 * 1e3, marker="o", s=44, c=C64, alpha=0.85, label="MATPes-r2SCAN (float64)", zorder=3)
if F32:
    ax.scatter(d.n_atoms, d.t32 * 1e3, marker="^", s=50, c=C32, alpha=0.85, label="MATPes-r2SCAN (float32)", zorder=3)
ax.scatter(base.n_atoms, base.tL * 1e3, marker="s", s=44, c=CL, alpha=0.85, label=LOC_NAME, zorder=3)
if len(imgs):
    ax.scatter(imgs.n_atoms, imgs.tL * 1e3, marker="D", s=46, c=CI, alpha=0.85, label=f"{LOC_NAME}, periodic images", zorder=3)
guide(ax, fits_t["64"], 1000, C64, N64)
if F32:
    guide(ax, fits_t["32"], 1000, C32, "MATPes f32")
guide(ax, fits_t["L"], 1000, CL, LOC_NAME)
ax.set_ylabel("Energy + force time (ms)")
allt = np.concatenate([d[c] for c in ("t64", "tL") + (("t32",) if F32 else ())]) * 1e3
logticks(ax, allt.min(), allt.max(), (5, 10, 20, 50, 100, 200, 500, 1000, 2000))
ax.legend(loc="center right")
tag(ax, "(a)")

# (b) speedup
ax = axes[1]
sb = d if not args.mark_images else d[~d.images_enumerated]
ax.scatter(sb.n_atoms, sb.sp64, marker="o", s=44, c=C64, alpha=0.85, zorder=3,
           label="vs MATPes-r2SCAN (float64)" if F32 else "no periodic images")
if F32:
    ax.scatter(sb.n_atoms, sb.sp32, marker="^", s=50, c=C32, alpha=0.85, zorder=3, label="vs MATPes-r2SCAN (float32)")
if len(imgs):
    ax.scatter(imgs.n_atoms, imgs.sp64, marker="D", s=46, c=CI, alpha=0.85, zorder=3, label="periodic images enumerated")
    if F32:
        ax.scatter(imgs.n_atoms, imgs.sp32, marker="D", s=46, c=CI, alpha=0.85, zorder=3)
ax.axhline(1.0, ls="--", lw=1.3, color="0.3", zorder=1)
sp_all = pd.concat([d.sp64] + ([d.sp32] if F32 else []))
ax.set_ylim(min(0.8, sp_all.min() * 0.75), sp_all.max() * 1.30)
ax.set_ylabel(r"Local-MoE speedup ($\times$)")
if F32:
    txt = (f"float64: mean {d.sp64.mean():.2f}" + r"$\times$" + f", median {d.sp64.median():.2f}" + r"$\times$" + "\n"
           f"float32: mean {d.sp32.mean():.2f}" + r"$\times$" + f", median {d.sp32.median():.2f}" + r"$\times$")
    ax.text(0.97, 0.96, txt, transform=ax.transAxes, ha="right", va="top", fontsize=11, fontweight="bold")
else:
    ax.text(0.97, 0.96, f"Mean = {d.sp64.mean():.2f}" + r"$\times$" + f"\nMedian = {d.sp64.median():.2f}" + r"$\times$",
            transform=ax.transAxes, ha="right", va="top", fontsize=12, fontweight="bold")
if F32 or len(imgs):
    ax.legend(loc="lower right")
tag(ax, "(b)")

# (c) memory
ax = axes[2]
ax.scatter(d.n_atoms, d.m64 / 1024.0, marker="o", s=44, c=C64, alpha=0.85, label="MATPes-r2SCAN (float64)", zorder=3)
if F32:
    ax.scatter(d.n_atoms, d.m32 / 1024.0, marker="^", s=50, c=C32, alpha=0.85, label="MATPes-r2SCAN (float32)", zorder=3)
ax.scatter(base.n_atoms, base.mL / 1024.0, marker="s", s=44, c=CL, alpha=0.85, label=LOC_NAME, zorder=3)
if len(imgs):
    ax.scatter(imgs.n_atoms, imgs.mL / 1024.0, marker="D", s=46, c=CI, alpha=0.85, label=f"{LOC_NAME}, periodic images", zorder=3)
guide(ax, fits_m["64"], 800, C64, N64)
if F32:
    guide(ax, fits_m["32"], 800, C32, "MATPes f32")
guide(ax, fits_m["L"], 800, CL, LOC_NAME)
ax.set_ylabel("Peak allocated GPU memory (GiB)")
allm = np.concatenate([d[c] for c in ("m64", "mL") + (("m32",) if F32 else ())]) / 1024.0
logticks(ax, allm.min(), allm.max(), (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50))
ax.legend(loc="lower right")
tag(ax, "(c)")

for ax in axes:
    ax.set_xlabel("Number of atoms")
    ax.set_xlim(0, nmax * 1.06)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / f"{args.stem}.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT / f"{args.stem}.pdf", bbox_inches="tight")
plt.close(fig)
print("\nsaved:", OUT / f"{args.stem}.png", "|", OUT / f"{args.stem}.pdf", "|", OUT / f"{args.stem}_plot_data.csv")
