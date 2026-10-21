#!/usr/bin/env python3
"""Energy and force parity from the evaluator's per-structure NPZ files.

Input: <raw>/<split>/<record>.npz, containing record_id, n_atoms,
E_reference_eV, E_predicted_eV, reference_forces and predicted_forces.

Energy RMSE/MAE weight each structure equally. Force SB-RMSE is
sqrt(mean_structure(mean_atom,component((F_pred - F_ref)**2))). Pooled force
RMSE weights every Cartesian component equally. Axis limits and zooms only
change the view; every selected structure/component contributes to the full
split metrics. No positive-only filter is appropriate for these quantities.

--paper writes a 7.2-inch-wide figure with manuscript-size fonts, shared
comparison axes, a common logarithmic count scale, and PDF/SVG/600-dpi PNG.
Force panels show SB-RMSE only; pooled RMSE and component counts remain in
the statistics CSV.
The stats CSV and caption document any points outside the displayed axes.
Requires numpy, pandas and matplotlib. The existing CLI remains supported.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FormatStrFormatter, MaxNLocator

TITLES = {"train": "Train", "val": "Validation", "test": "Test"}
REFERENCE = r"MACE-MatPES-r$^2$SCAN"
SOURCE_STYLE = {
    "Oishi_all179": ("#0072B2", "o", "Oishi"),
    "Oishi_relaxed52": ("#7B4EA3", "D", "Oishi (relaxed)"),
    "Own_augmented": ("#D55E00", "s", "Own"),
    "Siddharth": ("#009E73", "^", "Siddharth"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--metrics", type=Path, help="optional evaluator metrics.csv consistency check")
    p.add_argument("--model-label", default="Local-MoE")
    p.add_argument("--source-csv", type=Path, help="CSV with record_id,source")
    p.add_argument("--color-by-source", action="store_true")
    p.add_argument("--dedup-csv", type=Path,
                   help="retain representatives using the original n_atoms / rounded E_total / sum_abs_force fingerprint")
    p.add_argument("--energy-range", nargs=2, type=float, metavar=("LO", "HI"),
                   help="shared eV/atom display limits; full metrics are retained")
    p.add_argument("--zoom-row", nargs=2, type=float, metavar=("LO", "HI"),
                   help="separate reference-energy zoom row; replaces --energy-inset")
    p.add_argument("--energy-inset", nargs=2, type=float, metavar=("LO", "HI"),
                   help="shared eV/atom inset limits on both axes")
    p.add_argument("--force-limit", type=float,
                   help="shared symmetric force-axis limit in eV/Angstrom; default includes all components")
    p.add_argument("--paper", action="store_true", help="manuscript layout matching the other Local-MoE figures")
    p.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"),
                   help="inches; paper defaults: 7.2 x 5.6, or 7.2 x 8.0 with --zoom-row")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--out-dir", type=Path, default=Path("."))
    p.add_argument("--stem", default="energy_force_parity")
    a = p.parse_args(argv)
    for name in ("energy_range", "energy_inset", "zoom_row"):
        v = getattr(a, name)
        if v is not None and (not np.isfinite(v).all() or v[0] >= v[1]):
            p.error(f"--{name.replace('_', '-')} requires finite LO < HI")
    if a.force_limit is not None and (not np.isfinite(a.force_limit) or a.force_limit <= 0):
        p.error("--force-limit must be finite and positive")
    if a.figsize is not None and (not np.isfinite(a.figsize).all() or min(a.figsize) <= 0):
        p.error("--figsize requires finite positive dimensions")
    if a.dpi < 1 or len(a.splits) > 3 or len(set(a.splits)) != len(a.splits):
        p.error("use positive dpi and one to three distinct splits")
    if a.color_by_source and a.source_csv is None:
        p.error("--color-by-source requires --source-csv")
    if a.zoom_row is not None:
        a.energy_inset = None
    return a


def scalar(d, name):
    v = np.asarray(d[name])
    if v.size != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return v.reshape(-1)[0].item()


def load_split(root, split):
    rows, forces = [], []
    for path in sorted((root / split).glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as d:
                nn = float(scalar(d, "n_atoms"))
                if not np.isfinite(nn) or nn < 1 or nn != int(nn):
                    raise ValueError("n_atoms must be a positive integer")
                n = int(nn)
                record = scalar(d, "record_id")
                record = record.decode("utf-8") if isinstance(record, bytes) else str(record)
                er, ep = float(scalar(d, "E_reference_eV")), float(scalar(d, "E_predicted_eV"))
                fr, fp = np.asarray(d["reference_forces"], float), np.asarray(d["predicted_forces"], float)
                if fr.shape != (n, 3) or fp.shape != (n, 3):
                    raise ValueError(f"force arrays must both have shape ({n}, 3)")
                if not (np.isfinite([er, ep]).all() and np.isfinite(fr).all() and np.isfinite(fp).all()):
                    raise ValueError("non-finite energy or force value")
                rows.append(dict(record_id=record, n_atoms=n, E_ref=er, E_pred=ep))
                forces.append((fr, fp))
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"{path}: {exc}") from exc
    if not rows:
        raise ValueError(f"no raw NPZ files in {root / split}")
    df = pd.DataFrame(rows)
    if df.record_id.duplicated().any():
        raise ValueError(f"duplicate record_id within {split}")
    return df, forces


def read_lookup(path, columns):
    d = pd.read_csv(path, dtype={"record_id": str})
    if set(columns) - set(d.columns):
        raise ValueError(f"{path}: missing columns {sorted(set(columns) - set(d.columns))}")
    if d[columns].isna().any().any() or d.record_id.duplicated().any():
        raise ValueError(f"{path}: missing values or repeated record_id")
    return d


def load_data(a):
    representatives = None
    dedup_ids = None
    if a.dedup_csv is not None:
        q = read_lookup(a.dedup_csv, ["record_id", "n_atoms", "E_total", "sum_abs_force"])
        key = q.n_atoms.astype(str) + "|" + q.E_total.round(4).astype(str) + "|" + q.sum_abs_force.round(4).astype(str)
        q["rep"] = q.groupby(key).record_id.transform("min")
        representatives = set(q.loc[q.record_id == q.rep, "record_id"])
        dedup_ids = set(q.record_id)
        print("Deduplication uses the supplied rounded energy/force fingerprint, not a geometry comparison.", file=sys.stderr)
    source_map = None
    if a.source_csv is not None:
        q = read_lookup(a.source_csv, ["record_id", "source"])
        source_map = dict(q[["record_id", "source"]].itertuples(index=False, name=None))
    data, stats, seen = {}, [], set()
    for split in a.splits:
        df, fl = load_split(a.raw, split)
        n_records = len(df)
        overlap = seen.intersection(df.record_id)
        if overlap:
            raise ValueError(f"record_id appears in more than one split: {sorted(overlap)[:5]}")
        seen.update(df.record_id)
        if representatives is not None:
            if not set(df.record_id) <= dedup_ids:
                raise ValueError("--dedup-csv does not contain every raw record_id")
            keep = df.record_id.isin(representatives).to_numpy()
            df, fl = df[keep].reset_index(drop=True), [v for v, k in zip(fl, keep) if k]
            if df.empty:
                raise ValueError(f"deduplication leaves no structures in {split}")
        de = ((df.E_pred - df.E_ref) / df.n_atoms).to_numpy()
        fmse = np.array([np.mean((p - r) ** 2) for r, p in fl])
        st = dict(split=split, records_in_files=n_records, structures_used=len(df), atoms=int(df.n_atoms.sum()),
                  E_RMSE_meV_per_atom=1000 * float(np.sqrt(np.mean(de ** 2))),
                  E_MAE_meV_per_atom=1000 * float(np.mean(np.abs(de))),
                  E_bias_meV_per_atom=1000 * float(np.mean(de)),
                  F_SB_RMSE_eV_A=float(np.sqrt(np.mean(fmse))),
                  F_atom_weighted_RMSE_eV_A=float(np.sqrt(np.average(fmse, weights=df.n_atoms))),
                  force_components=int(3 * df.n_atoms.sum()))
        data[split] = (df, fl)
        stats.append(st)
    return data, pd.DataFrame(stats), source_map


def check_metrics(a, stats):
    if a.metrics is None:
        return
    if a.dedup_csv is not None:
        print("Evaluator comparison skipped: deduplication changes the evaluated population.", file=sys.stderr)
        return
    m = pd.read_csv(a.metrics)
    required = {"split", "source", "family", "energy_RMSE_meV_per_atom", "force_RMSE_structure_balanced_eV_A"}
    if required - set(m):
        raise ValueError(f"metrics CSV lacks {sorted(required - set(m))}")
    m = m[(m.source == "ALL") & (m.family == "ALL")].set_index("split")
    if m.index.duplicated().any() or not set(a.splits) <= set(m.index):
        raise ValueError("metrics CSV needs exactly one source=ALL, family=ALL row for each selected split")
    diffs_e, diffs_f = [], []
    for r in stats.itertuples():
        diffs_e.append(abs(float(m.loc[r.split, "energy_RMSE_meV_per_atom"]) - r.E_RMSE_meV_per_atom))
        diffs_f.append(abs(float(m.loc[r.split, "force_RMSE_structure_balanced_eV_A"]) - r.F_SB_RMSE_eV_A))
    if not np.isfinite(diffs_e + diffs_f).all() or max(diffs_e) > 1e-3 or max(diffs_f) > 1e-5:
        raise ValueError("MISMATCH between raw files and evaluator metrics; figure was not generated")
    print(f"Evaluator check passed: max |dE_RMSE|={max(diffs_e):.2e} meV/atom; max |dF_SB|={max(diffs_f):.2e} eV/Angstrom")


def display_limits(a, data, stats):
    all_e = np.concatenate([np.r_[d.E_ref / d.n_atoms, d.E_pred / d.n_atoms] for d, _ in data.values()])
    lo, hi = float(all_e.min()), float(all_e.max())
    pad = 0.04 * max(hi - lo, 0.01)
    erange = tuple(a.energy_range) if a.energy_range is not None else (lo - pad, hi + pad)
    max_force = max(float(np.abs(v).max()) for _, fl in data.values() for pair in fl for v in pair)
    limit = a.force_limit if a.force_limit is not None else (1.05 * max_force if max_force else 1.0)
    for i, split in enumerate(a.splits):
        d, fl = data[split]
        x, y = (d.E_ref / d.n_atoms).to_numpy(), (d.E_pred / d.n_atoms).to_numpy()
        stats.loc[i, "energy_axis_min_eV_per_atom"] = erange[0]
        stats.loc[i, "energy_axis_max_eV_per_atom"] = erange[1]
        stats.loc[i, "force_axis_limit_eV_A"] = limit
        stats.loc[i, "E_structures_outside_axes"] = int(((x < erange[0]) | (x > erange[1]) | (y < erange[0]) | (y > erange[1])).sum())
        stats.loc[i, "F_components_outside_axes"] = sum(int(((np.abs(r) > limit) | (np.abs(p) > limit)).sum()) for r, p in fl)
        for kind, window in (("inset", a.energy_inset), ("zoom", a.zoom_row)):
            if window is not None:
                low, high = window
                selected = (x >= low) & (x <= high)
                stats.loc[i, f"E_{kind}_reference_structures"] = int(selected.sum())
                stats.loc[i, f"E_{kind}_predictions_outside_window"] = int((selected & ((y < low) | (y > high))).sum())
                if kind == "zoom":
                    stats.loc[i, "E_zoom_RMSE_meV_per_atom"] = 1000 * float(np.sqrt(np.mean((y[selected] - x[selected]) ** 2))) if selected.any() else np.nan
    for col in stats:
        if "structures" in col or "components" in col or "outside_window" in col:
            stats[col] = stats[col].astype(int)
    return erange, limit


def configure_style(paper):
    plt.rcParams.update({
        "font.family": "DejaVu Sans" if paper else "STIXGeneral",
        "mathtext.fontset": "dejavusans" if paper else "stix",
        "font.size": 8 if paper else 13, "font.weight": "normal",
        "axes.labelweight": "bold", "axes.linewidth": 0.9 if paper else 1.4,
        "xtick.labelsize": 7.5 if paper else 12, "ytick.labelsize": 7.5 if paper else 12,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3 if paper else 6, "ytick.major.size": 3 if paper else 6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True,
    })


def ticks_for_energy(limits, small=False):
    lo, hi = limits
    if small:
        return np.linspace(lo, hi, 3)
    step = 10 ** np.floor(np.log10((hi - lo) / 4))
    ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
    ticks = ticks[(ticks >= lo - 1e-10 * step) & (ticks <= hi + 1e-10 * step)]
    return ticks if len(ticks) >= 2 else np.linspace(lo, hi, 3)


def energy_axes(ax, limits, inset=False):
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ticks = ticks_for_energy(limits, small=inset)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    spacing = float(np.min(np.diff(ticks)))
    decimals = max(2, min(6, int(np.ceil(-np.log10(spacing))) + (1 if np.isclose(spacing / 10 ** np.floor(np.log10(spacing)), 2.5) else 0)))
    ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
    if not inset:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


def draw_points(ax, x, y, sources, paper, handles=None, inset=False):
    size = (4 if inset else 9) if paper else (14 if inset else 26)
    if sources is None:
        ax.scatter(x, y, s=size, color="#0072B2", alpha=0.8, linewidths=0, zorder=3)
        return
    for source in sorted(sources.unique()):
        keep = (sources == source).to_numpy()
        color, marker, label = SOURCE_STYLE.get(source, ("0.45", "o", source))
        handle = ax.scatter(x[keep], y[keep], s=size, color=color, marker=marker, alpha=0.85,
                            linewidths=0, label=label, zorder=3)
        if handles is not None:
            handles.setdefault(label, handle)


def add_identity(ax, limits, paper):
    ax.plot(limits, limits, "--", color="0.45", lw=0.8 if paper else 1.4, zorder=2)


def make_figure(a, data, stats, sources, erange, force_limit):
    configure_style(a.paper)
    ncols, nrows = len(a.splits), 3 if a.zoom_row is not None else 2
    force_row = nrows - 1
    figsize = a.figsize or ((7.2, 8.0 if nrows == 3 else 5.6) if a.paper else (16.5, 13.4 if nrows == 3 else 9.4))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols, left=0.115 if a.paper else 0.085, right=0.90,
                          top=0.925 if a.color_by_source else 0.94, bottom=0.115,
                          wspace=0.20, hspace=0.53 if a.paper else 0.46)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    hbs, handles = [], {}
    ann_size = 7.2 if a.paper else 12.5
    panel_size = 10 if a.paper else 17
    for col, split in enumerate(a.splits):
        d, fl = data[split]
        st = stats.loc[stats.split == split].iloc[0]
        x, y = (d.E_ref / d.n_atoms).to_numpy(), (d.E_pred / d.n_atoms).to_numpy()
        srcs = d.record_id.map(sources).fillna("Other") if a.color_by_source else None
        ax = axes[0][col]
        energy_axes(ax, erange)
        add_identity(ax, erange, a.paper)
        draw_points(ax, x, y, srcs, a.paper, handles)
        text = f"RMSE = {st.E_RMSE_meV_per_atom:.2f} meV/atom"
        if a.paper:
            text += f"\nMAE = {st.E_MAE_meV_per_atom:.2f} meV/atom"
        text += f"\n$N_\\mathrm{{str}}$ = {int(st.structures_used)}"
        ax.text(0.045, 0.96, text, transform=ax.transAxes, va="top", fontsize=ann_size, linespacing=1.25,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6), zorder=5)
        if a.energy_inset is not None:
            limits = tuple(a.energy_inset)
            # The lower-right corner leaves the low-energy cluster and metrics visible.
            ins = ax.inset_axes([0.55, 0.12, 0.40, 0.40])
            energy_axes(ins, limits, inset=True)
            add_identity(ins, limits, a.paper)
            draw_points(ins, x, y, srcs, a.paper, inset=True)
            ins.tick_params(labelsize=5.5 if a.paper else 9, length=2, pad=1)
            ins.patch.set_alpha(0.97)
            ax.add_patch(Rectangle((limits[0], limits[0]), limits[1] - limits[0], limits[1] - limits[0],
                                   fill=False, ec="0.5", lw=0.65, zorder=4))
        if col == 0:
            ax.set_ylabel(f"{a.model_label} energy\n(eV/atom)", fontsize=9 if a.paper else 14, labelpad=4)
        if a.zoom_row is not None:
            limits = tuple(a.zoom_row)
            zax = axes[1][col]
            energy_axes(zax, limits)
            add_identity(zax, limits, a.paper)
            selected = (x >= limits[0]) & (x <= limits[1])
            draw_points(zax, x[selected], y[selected], srcs[selected] if srcs is not None else None, a.paper)
            rmse = st.E_zoom_RMSE_meV_per_atom
            ztext = f"RMSE = {rmse:.2f} meV/atom" if np.isfinite(rmse) else "No reference structures"
            zax.text(0.045, 0.96, ztext + f"\n$N_\\mathrm{{str}}$ = {int(selected.sum())}",
                      transform=zax.transAxes, va="top", fontsize=ann_size,
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))
            ax.add_patch(Rectangle((limits[0], limits[0]), limits[1] - limits[0], limits[1] - limits[0],
                                   fill=False, ec="0.5", lw=0.65, zorder=4))
            if col == 0:
                zax.set_ylabel(f"{a.model_label} energy\n(eV/atom)", fontsize=9 if a.paper else 14, labelpad=4)
        fax = axes[force_row][col]
        fr = np.concatenate([r.reshape(-1) for r, _ in fl])
        fp = np.concatenate([p.reshape(-1) for _, p in fl])
        visible = (np.abs(fr) <= force_limit) & (np.abs(fp) <= force_limit)
        hb = fax.hexbin(fr[visible], fp[visible], gridsize=100 if a.paper else 130,
                        extent=(-force_limit, force_limit, -force_limit, force_limit), mincnt=1,
                        norm=LogNorm(vmin=1, vmax=10), cmap="viridis", linewidths=0)
        hbs.append(hb)
        add_identity(fax, (-force_limit, force_limit), a.paper)
        fax.set_xlim(-force_limit, force_limit)
        fax.set_ylim(-force_limit, force_limit)
        fax.set_aspect("equal", adjustable="box")
        locator = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], symmetric=True)
        fticks = locator.tick_values(-force_limit, force_limit)
        fticks = fticks[(fticks >= -force_limit) & (fticks <= force_limit)]
        fax.set_xticks(fticks)
        fax.set_yticks(fticks)
        fax.xaxis.set_minor_locator(AutoMinorLocator(2))
        fax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ftext = f"SB-RMSE = {st.F_SB_RMSE_eV_A:.3f} eV/Å"
        fax.text(0.045, 0.96, ftext, transform=fax.transAxes, va="top", fontsize=ann_size, linespacing=1.25,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6), zorder=5)
        if col == 0:
            fax.set_ylabel(f"{a.model_label} force component\n(eV/Å)", fontsize=9 if a.paper else 14, labelpad=4)
        for row in range(nrows):
            axr = axes[row][col]
            axr.text(-0.015, 1.035, f"({chr(97 + row * ncols + col)})", transform=axr.transAxes,
                      fontsize=panel_size, fontweight="bold", va="bottom", ha="left")
            if row == 0:
                axr.set_title(TITLES.get(split, split), fontsize=10 if a.paper else 18, fontweight="bold", pad=8)
    max_count = max([float(h.get_array().max()) for h in hbs if h.get_array().size] + [10.0])
    for hb in hbs:
        hb.set_clim(1, max_count)
    fig.canvas.draw()
    right_pos = axes[force_row][-1].get_position()
    cax = fig.add_axes([0.925, right_pos.y0, 0.014, right_pos.height])
    cb = fig.colorbar(hbs[-1], cax=cax)
    if cb.solids is not None:
        cb.solids.set_rasterized(False)
    cb.set_label("Count (log scale)", fontsize=8 if a.paper else 14, labelpad=3)
    cb.ax.tick_params(labelsize=7 if a.paper else 11, length=2)
    for row in range(nrows):
        positions = [ax.get_position() for ax in axes[row]]
        ybottom = min(v.y0 for v in positions)
        text = (f"{REFERENCE} force component (eV/Å)" if row == force_row
                else f"{REFERENCE} energy (eV/atom)")
        fig.text((positions[0].x0 + positions[-1].x1) / 2, ybottom - (0.36 if a.paper else 0.64) / figsize[1],
                  text, ha="center", va="top", fontsize=9 if a.paper else 16, fontweight="bold")
    if handles:
        fig.legend(list(handles.values()), list(handles), loc="upper center", bbox_to_anchor=(0.53, 1.0),
                   ncol=min(4, len(handles)), frameon=False, fontsize=7 if a.paper else 11,
                   handletextpad=0.4, columnspacing=1)
    return fig, axes


def make_caption(a, stats, erange, force_limit):
    splits = "; ".join(f"{TITLES.get(r.split, r.split).lower()}: {r.structures_used} structures" for r in stats.itertuples())
    text = (f"Figure X. Energy and force parity of {a.model_label} against MACE-MatPES-r²SCAN ({splits}). "
            "The top row shows energy per atom, with one point per selected structure. "
            "Energy RMSE and MAE weight structures equally. ")
    if a.energy_inset is not None:
        text += f"Insets show the range {a.energy_inset[0]:g} to {a.energy_inset[1]:g} eV/atom on both axes. "
    if a.zoom_row is not None:
        text += (f"The middle row selects structures with reference energies from {a.zoom_row[0]:g} to {a.zoom_row[1]:g} eV/atom; "
                 "its RMSE uses all predictions for those structures, including predictions outside the displayed window. ")
    text += ("The bottom row shows Cartesian force components as hexagonal-bin counts on a shared logarithmic colour scale "
             "with identical bin geometry across splits. Force SB-RMSE is the square root of the mean of the per-structure "
             "mean squared component errors, giving each structure equal weight. Dashed lines denote exact agreement. "
             "All full-split metrics include points outside the displayed axes. ")
    clipped = []
    for r in stats.itertuples():
        if r.E_structures_outside_axes or r.F_components_outside_axes:
            clipped.append(f"{r.split}: {r.E_structures_outside_axes} energy points and {r.F_components_outside_axes} force components")
    if clipped:
        text += (f"The displayed energy range is {erange[0]:g} to {erange[1]:g} eV/atom and the force range is ±{force_limit:g} eV/Å; "
                 "points outside these axes are " + "; ".join(clipped) + ". ")
    for kind, window in (("inset", a.energy_inset), ("zoom", a.zoom_row)):
        if window is not None:
            counts = [f"{r.split}: {int(getattr(r, f'E_{kind}_predictions_outside_window'))}" for r in stats.itertuples()
                      if getattr(r, f"E_{kind}_predictions_outside_window")]
            if counts:
                text += "Predictions outside the zoom window for structures whose reference energies lie within it: " + "; ".join(counts) + ". "
    if a.dedup_csv is not None:
        text += "Representatives were selected using the supplied rounded atom-count, total-energy and absolute-force fingerprint. "
    return text.strip() + "\n"


def main(argv=None):
    a = parse_args(argv)
    data, stats, sources = load_data(a)
    check_metrics(a, stats)
    erange, force_limit = display_limits(a, data, stats)
    fig, _ = make_figure(a, data, stats, sources, erange, force_limit)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    base = a.out_dir / a.stem
    stats.to_csv(base.with_name(base.name + "_stats.csv"), index=False)
    base.with_name(base.name + "_caption.txt").write_text(make_caption(a, stats, erange, force_limit), encoding="utf-8")
    config = dict(raw=str(a.raw), splits=a.splits, model_label=a.model_label,
                  energy_range=list(erange), force_limit=force_limit, energy_inset=a.energy_inset,
                  zoom_row=a.zoom_row, figsize=list(fig.get_size_inches()), dpi=a.dpi,
                  dedup_csv=str(a.dedup_csv) if a.dedup_csv else None,
                  metrics_csv=str(a.metrics) if a.metrics else None,
                  metrics_checked=bool(a.metrics is not None and a.dedup_csv is None),
                  all_selected_points_in_full_metrics=True)
    base.with_name(base.name + "_settings.json").write_text(json.dumps(config, indent=2) + "\n")
    extensions = ("pdf", "svg", "png") if a.paper else ("pdf", "png")
    for ext in extensions:
        # Fixed canvas keeps the requested physical size when inserted into a manuscript.
        fig.savefig(base.with_name(base.name + f".{ext}"), dpi=a.dpi if ext == "png" else None, facecolor="white")
    plt.close(fig)
    columns = ["split", "structures_used", "atoms", "E_RMSE_meV_per_atom", "E_MAE_meV_per_atom", "F_SB_RMSE_eV_A", "F_atom_weighted_RMSE_eV_A"]
    print(stats[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    clipped = stats[["split", "E_structures_outside_axes", "F_components_outside_axes"]]
    if clipped.iloc[:, 1:].to_numpy().any():
        print("Points outside displayed axes (retained in metrics; also recorded in stats and caption):\n" + clipped.to_string(index=False), file=sys.stderr)
    print("saved", base.with_name(base.name + ".pdf"))
    print("saved statistics and caption in", a.out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
