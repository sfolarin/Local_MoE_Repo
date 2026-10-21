#!/usr/bin/env python3
"""Audit and plot Local-MoE NVE energy conservation on a common time window.

Energy change: 1000 * (E_total(t)/N - E_total(0)/N), in meV/atom.
Every displayed maximum, fitted slope, and envelope uses the same interval.
Completion is checked from the logged duration/steps and protocol metadata;
final atomic-coordinate files are not needed for this analysis.

Example:
  python3 plot_local_moe_nve_drift.py --md-root md_holdout40_v1 --paper

Paper output: <prefix>_paper.pdf / .svg / .png (600 dpi by default).
Also writes the per-structure summary, validation JSON, binned envelope CSV,
and a caption derived from the verified data and available protocol metadata.
Without --paper, retains the three diagnostic figure types of the earlier script.
Requires numpy, pandas, and matplotlib.
"""
# --- added when the repository was assembled: no machine-specific paths ---------------------------------------
# MOE_HET (environment variable) = the data workspace folder that holds Local_MoE_MATPES_R2SCAN_Expanded_v1/ and
# External_Benchmark_Oishi2025/; REPO_ROOT = this repository.
import os as _os
import sys as _sys
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[1]
_REPO_CODE_DIRS = [str(REPO_ROOT / _d) for _d in ['src/local_moe', 'training', 'data_splits', 'evaluation', 'interface_energy', 'cost', 'md', 'deploy', 'tests', 'figures']]
for _p in _REPO_CODE_DIRS:
    if _p not in _sys.path:
        _sys.path.append(_p)
# -------------------------------------------------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter

HET = (_os.environ.get("MOE_HET", "."))
SOURCE_ORDER = {"Oishi_relaxed52": 0, "Oishi_all179": 0, "Siddharth": 1, "Own_augmented": 2}
SOURCE_GROUP = {"Oishi_relaxed52": "Oishi", "Oishi_all179": "Oishi",
                "Siddharth": "Siddharth", "Own_augmented": "Own"}
SOURCE_STYLE = {"Oishi": ("#0072B2", "o"), "Own": ("#D55E00", "s"),
                "Siddharth": ("#7B4EA3", "D")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md-root", type=Path,
                   default=Path(HET) / "Local_MoE_MATPES_R2SCAN_Expanded_v1/md_holdout40_v1")
    p.add_argument("--manifest", type=Path, help="default: <md-root>/md_manifest.csv")
    p.add_argument("--out-dir", type=Path, help="default: <md-root>/figures")
    p.add_argument("--prefix", default="LocalMoE_holdout40_nve")
    p.add_argument("--limit", type=float, default=0.1,
                   help="chosen maximum-excursion criterion, in meV/atom (default: 0.1)")
    p.add_argument("--duration-ps", type=float, default=10.0,
                   help="common analysis interval from the first NVE sample (default: 10 ps)")
    p.add_argument("--bins", type=int, default=250,
                   help="time bins for the envelope and structure-weighted mean (default: 250)")
    p.add_argument("--allow-missing", action="store_true",
                   help="explicitly omit missing or incomplete cases, listing every omission")
    p.add_argument("--reference-summary", type=Path,
                   help="optional earlier summary CSV to compare by record_id; never overwritten")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--paper", action="store_true",
                   help="three-panel journal figure: binned range/mean, maximum excursion, and fitted slope")
    p.add_argument("--figsize", type=float, nargs=2, default=(7.2, 3.3), metavar=("W", "H"),
                   help="paper figure size in inches (default: 7.2 3.3)")
    a = p.parse_args()
    if a.duration_ps <= 0 or a.limit <= 0 or a.bins < 1 or a.dpi < 1:
        p.error("duration, limit, number of bins, and dpi must be positive")
    return a


class IncompleteRun(ValueError):
    """Missing files or a trajectory that does not reach its expected endpoint."""


def load_case(root, r, index, duration):
    path = root / f"case_{r.record_id}"
    logfile = path / "nve_10ps_log.csv"
    if not logfile.is_file():
        raise IncompleteRun("NVE log missing")
    metadata_file = path / "final_protocol_summary.json"
    meta = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
    protocol = meta.get("protocol", {})
    if meta.get("record_id", r.record_id) != r.record_id:
        raise ValueError("protocol record_id disagrees with the manifest")
    n_atoms = int(r.n_atoms)
    if n_atoms <= 0 or n_atoms != float(r.n_atoms):
        raise ValueError("manifest n_atoms is not a positive integer")
    if int(meta.get("n_atoms", n_atoms)) != n_atoms:
        raise ValueError("protocol atom count disagrees with the manifest")
    raw = pd.read_csv(logfile)
    required = ["step", "time_ps", "total_energy_per_atom_eV", "temperature_K"]
    if set(required) - set(raw):
        raise ValueError(f"missing log columns: {sorted(set(required) - set(raw))}")
    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
        if not np.isfinite(raw[col].to_numpy(float)).all():
            raise ValueError(f"non-finite values in {col}")
    if "stage" in raw and not raw.stage.astype(str).str.upper().eq("NVE").all():
        raise ValueError("log contains a stage other than NVE")
    duplicates = int(raw.step.duplicated().sum())
    d = raw.sort_values("step", kind="stable").drop_duplicates("step", keep="last").reset_index(drop=True)
    if len(d) < 2:
        raise IncompleteRun("fewer than two distinct NVE samples")
    t = d.time_ps.to_numpy(float) - float(d.time_ps.iloc[0])
    steps = d.step.to_numpy(float)
    if steps[0] != 0:
        raise IncompleteRun("initial NVE sample at step 0 is missing")
    if not (np.diff(t) > 0).all():
        raise ValueError("logged time is not strictly increasing")
    dt_fs = protocol.get("dt_fs")
    if dt_fs is not None:
        dt_fs = float(dt_fs)
        if dt_fs <= 0 or not np.allclose(t, (steps - steps[0]) * dt_fs / 1000.0, rtol=0, atol=1e-8):
            raise ValueError("time and step columns disagree with the recorded timestep")
    expected = float(protocol.get("nve_ps", duration))
    if t[-1] < max(expected, duration) - 1e-8:
        raise IncompleteRun(f"ends at {t[-1]:g} ps; needs {max(expected, duration):g} ps")
    if dt_fs is not None and steps[-1] < round(expected * 1000.0 / dt_fs):
        raise IncompleteRun("last logged step is before the protocol endpoint")

    # Independent consistency checks on the full log, before window selection.
    energy_sum_error = None
    energy_atom_error = None
    if {"potential_energy_eV", "kinetic_energy_eV", "total_energy_eV"} <= set(d):
        error = d.total_energy_eV - d.potential_energy_eV - d.kinetic_energy_eV
        energy_sum_error = float(np.max(np.abs(error)))
        if not np.isfinite(energy_sum_error) or energy_sum_error > 1e-8:
            raise ValueError("total energy is inconsistent with potential plus kinetic energy")
    if "total_energy_eV" in d:
        error = d.total_energy_eV / n_atoms - d.total_energy_per_atom_eV
        energy_atom_error = float(np.max(np.abs(error)))
        if not np.isfinite(energy_atom_error) or energy_atom_error > 1e-10:
            raise ValueError("per-atom energy is inconsistent with total energy / n_atoms")
    volume_range = None
    if "volume_A3" in d:
        volume = d.volume_A3.to_numpy(float)
        if not np.isfinite(volume).all() or (volume <= 0).any():
            raise ValueError("invalid NVE cell volume")
        volume_range = float(np.ptp(volume) / volume[0])
        if volume_range > 1e-10:
            raise ValueError("cell volume changes during the NVE trajectory")

    mask = t <= duration + 1e-8
    tt = t[mask]
    if len(tt) < 2 or not np.isclose(tt[-1], duration, rtol=0, atol=1e-8):
        raise ValueError("analysis endpoint does not coincide with a logged sample")
    energy = d.total_energy_per_atom_eV.to_numpy(float)[mask]
    drift = (energy - energy[0]) * 1000.0
    label = f"{r.label} #{index}"
    row = {
        "production_index": index, "record_id": r.record_id, "interface": label, "source": r.source,
        "n_atoms": n_atoms, "start_nve_ps": float(tt[0]), "end_nve_ps": float(tt[-1]), "n_samples": len(tt),
        "final_drift_meV_per_atom": float(drift[-1]),
        "max_abs_drift_meV_per_atom": float(np.abs(drift).max()),
        "peak_to_peak_meV_per_atom": float(np.ptp(drift)),
        "linear_drift_meV_per_atom_per_ps": float(np.polyfit(tt, drift, 1)[0]),
        "nve_mean_temperature_K": float(d.temperature_K.to_numpy(float)[mask].mean()),
        "npt_volume_change_percent": meta.get("npt_volume_change_percent", np.nan),
        "relax_fmax_final_eV_A": meta.get("relaxation", {}).get("Fmax_final_eV_A", np.nan),
        "nve_steps_per_second": meta.get("nve_steps_per_second", np.nan),
        "split_group": r.split_group, "recorded_split": meta.get("split", "unknown"),
        "analysis_window_ps": duration, "available_end_nve_ps": float(t[-1]),
        "available_samples": len(d), "expected_nve_ps": expected, "recorded_dt_fs": dt_fs,
        "duplicates_resolved": duplicates, "volume_relative_range": volume_range,
        "energy_sum_max_error_eV": energy_sum_error,
        "energy_per_atom_max_error_eV": energy_atom_error,
    }
    return row, (label, tt, drift), meta


def binned_envelope(curves, duration, bins):
    edges = np.linspace(0.0, duration, bins + 1)
    low, high = np.full(bins, np.inf), np.full(bins, -np.inf)
    total, count = np.zeros(bins), np.zeros(bins)
    for _, t, drift in curves:
        k = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, bins - 1)
        np.minimum.at(low, k, drift)
        np.maximum.at(high, k, drift)
        sums, samples = np.bincount(k, weights=drift, minlength=bins), np.bincount(k, minlength=bins)
        present = samples > 0
        total[present] += sums[present] / samples[present]
        count[present] += 1
    present = count > 0
    mean = np.full(bins, np.nan)
    mean[present] = total[present] / count[present]
    low[~present], high[~present] = np.nan, np.nan
    return pd.DataFrame({"time_left_ps": edges[:-1], "time_right_ps": edges[1:],
                         "time_centre_ps": (edges[:-1] + edges[1:]) / 2,
                         "min_drift_meV_per_atom": low, "max_drift_meV_per_atom": high,
                         "structure_mean_drift_meV_per_atom": mean,
                         "n_contributing_structures": count.astype(int)})


def source_series(summary):
    group = summary.source.map(lambda x: SOURCE_GROUP.get(x, str(x)))
    names = [x for x in ("Oishi", "Siddharth", "Own") if (group == x).any()]
    names += sorted(set(group) - set(names))
    for name in names:
        color, marker = SOURCE_STYLE.get(name, ("#666666", "^"))
        yield name, group.eq(name).to_numpy(), color, marker


def paper_figure(a, out, summary, envelope):
    rc = {
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "font.size": 8,
        "font.weight": "normal", "axes.labelsize": 9, "axes.labelweight": "bold",
        "axes.linewidth": 0.9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 3.5,
        "ytick.major.size": 3.5, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    n = summary.n_atoms.to_numpy()
    maximum = summary.max_abs_drift_meV_per_atom.to_numpy()
    slopes = summary.linear_drift_meV_per_atom_per_ps.to_numpy() * 1000.0
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=tuple(a.figsize))
        fig.subplots_adjust(left=0.085, right=0.972, bottom=0.19, top=0.835, wspace=0.50)
        ax = axes[0]
        edges = np.r_[envelope.time_left_ps.to_numpy(), envelope.time_right_ps.iloc[-1]]
        low, high = envelope.min_drift_meV_per_atom.to_numpy(), envelope.max_drift_meV_per_atom.to_numpy()
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]],
                        step="post", color="#0072B2", alpha=0.22, linewidth=0, zorder=2)
        ax.plot(envelope.time_centre_ps, envelope.structure_mean_drift_meV_per_atom,
                color="#17406E", lw=0.9, zorder=4)
        ax.axhline(0, color="0.4", ls=(0, (3, 2)), lw=0.7, zorder=1)
        ax.set_xlim(0, a.duration_ps)
        amplitude = max(float(maximum.max()), 1e-6)
        ax.set_ylim(-1.2 * amplitude, 1.2 * amplitude)
        ax.set_xlabel("NVE time (ps)")
        ax.set_ylabel(r"$\Delta E$ (meV/atom)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 5, 10]))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        for name, mask, color, marker in source_series(summary):
            style = dict(s=21, c=color, marker=marker, alpha=0.87,
                         edgecolors="white", linewidths=0.3, zorder=3)
            axes[1].scatter(n[mask], maximum[mask], **style)
            axes[2].scatter(n[mask], slopes[mask], **style)
        axes[1].set_ylim(0, 1.16 * amplitude)
        axes[1].set_ylabel(r"Max. $|\Delta E|$ (meV/atom)")
        slope_limit = max(float(np.abs(slopes).max()) * 1.25, 1e-6)
        axes[2].set_ylim(-slope_limit, slope_limit)
        axes[2].axhline(0, color="0.4", ls=(0, (3, 2)), lw=0.7, zorder=1)
        axes[2].set_ylabel("Linear drift (µeV/atom/ps)")

        xlow, xhigh = float(n.min()) / 1.18, float(n.max()) * 1.22
        ticks = []
        for exponent in range(int(np.floor(np.log10(xlow))) - 1,
                              int(np.ceil(np.log10(xhigh))) + 1):
            for multiplier in (1, 2, 5):
                value = multiplier * 10.0**exponent
                if xlow <= value <= xhigh and (not ticks or value / ticks[-1] >= 3):
                    ticks.append(value)
        for ax in axes[1:]:
            ax.set_xscale("log")
            ax.set_xlim(xlow, xhigh)
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("Number of atoms, $N$")
        for ax, tag in zip(axes, ("(a)", "(b)", "(c)")):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 4, 5, 10]))
            scale = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 5
            places = max(0, min(8, int(np.ceil(-np.log10(scale)))))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _, digits=places: f"{value:.{digits}f}"))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="both", top=True, right=True)
            ax.text(-0.015, 1.035, tag, transform=ax.transAxes, fontsize=10.5,
                    fontweight="bold", va="bottom", ha="left")
        handles = [Patch(facecolor="#0072B2", alpha=0.22, linewidth=0,
                         label=f"Range ({len(summary)} structures)"),
                   Line2D([], [], color="#17406E", lw=0.9, label="Structure mean")]
        for name, mask, color, marker in source_series(summary):
            handles.append(Line2D([], [], marker=marker, ls="", color=color,
                                  markersize=4.4, markeredgecolor="white", markeredgewidth=0.3,
                                  label=f"{name} ({int(mask.sum())})"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.99),
                   ncol=min(4, len(handles)), frameon=False, fontsize=7.6,
                   handlelength=1.5, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
        for extension in ("pdf", "svg", "png"):
            fig.savefig(out / f"{a.prefix}_paper.{extension}", dpi=a.dpi)
        plt.close(fig)


def diagnostic_figures(a, out, summary, curves):
    """Retain full-trajectory, per-case bar, and size-dependence diagnostics."""
    maximum = summary.max_abs_drift_meV_per_atom.to_numpy()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelsize": 12, "axes.labelweight": "bold", "pdf.fonttype": 42})
    def save(fig, suffix):
        for extension in ("pdf", "png"):
            fig.savefig(out / f"{a.prefix}_{suffix}.{extension}", dpi=a.dpi, bbox_inches="tight")
        plt.close(fig)
    legend_rows = int(np.ceil(len(curves) / 5))
    fig, ax = plt.subplots(figsize=(12, 4.8 + legend_rows * 0.2))
    for i, (label, t, drift) in enumerate(curves):
        ax.plot(t, drift, color=plt.get_cmap("tab20")(i % 20),
                ls=["-", "--", ":"][(i // 20) % 3], lw=0.7, label=label)
    ax.axhline(0, color="0.4", ls="--", lw=0.7)
    ax.set(xlim=(0, a.duration_ps), xlabel="NVE time (ps)", ylabel="Energy change (meV/atom)")
    ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=7, frameon=False)
    save(fig, "energy_drift")
    fig, ax = plt.subplots(figsize=(max(12, len(summary) * 0.24 + 4), 6))
    ax.bar(np.arange(len(summary)), maximum, width=0.72, color="#0072B2")
    ax.set_xticks(np.arange(len(summary)), summary.interface, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Maximum energy excursion (meV/atom)")
    fig.subplots_adjust(bottom=0.37)
    save(fig, "max_energy_excursion")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, mask, color, marker in source_series(summary):
        axes[0].scatter(summary.n_atoms[mask], maximum[mask], c=color, marker=marker, label=name)
        axes[1].scatter(summary.n_atoms[mask], summary.linear_drift_meV_per_atom_per_ps[mask],
                        c=color, marker=marker, label=name)
    axes[0].axhline(a.limit, color="0.35", ls=":", lw=1)
    axes[1].axhline(0, color="0.35", ls="--", lw=0.8)
    axes[0].set_ylabel("Maximum energy excursion (meV/atom)")
    axes[1].set_ylabel("Linear drift (meV/atom/ps)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Number of atoms")
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "drift_vs_natoms")


def compare_summary(path, summary):
    old = pd.read_csv(path).set_index("record_id")
    new = summary.set_index("record_id")
    if not old.index.is_unique or set(old.index) != set(new.index):
        raise ValueError("reference-summary record IDs differ from the selected trajectories")
    fields = ["n_atoms", "start_nve_ps", "end_nve_ps", "n_samples", "final_drift_meV_per_atom",
              "max_abs_drift_meV_per_atom", "peak_to_peak_meV_per_atom",
              "linear_drift_meV_per_atom_per_ps", "nve_mean_temperature_K"]
    difference = {}
    for field in fields:
        x, y = old.loc[new.index, field].to_numpy(float), new[field].to_numpy(float)
        difference[field] = float(np.max(np.abs(x - y)))
        if not np.allclose(x, y, rtol=1e-9, atol=1e-9):
            raise ValueError(f"reference-summary mismatch in {field}: {difference[field]:g}")
    return {"records": len(new), "max_absolute_differences": difference, "matches": True}


def main():
    a = parse_args()
    out = a.out_dir or a.md_root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(a.manifest or a.md_root / "md_manifest.csv")
    needed = {"record_id", "label", "source", "split_group", "n_atoms"}
    if needed - set(manifest) or not manifest.record_id.is_unique:
        raise ValueError("manifest lacks required columns or has duplicate record IDs")
    manifest["_order"] = manifest.source.map(SOURCE_ORDER).fillna(99)
    manifest = manifest.sort_values(["_order", "split_group", "record_id"]).reset_index(drop=True)
    rows, curves, metadata, omitted = [], [], [], []
    for index, r in manifest.iterrows():
        try:
            row, curve, meta = load_case(a.md_root, r, index, a.duration_ps)
        except (IncompleteRun, FileNotFoundError) as error:
            omitted.append({"record_id": r.record_id, "reason": str(error)})
            continue
        except ValueError as error:
            raise ValueError(f"{r.record_id}: {error}") from error
        rows.append(row)
        curves.append(curve)
        metadata.append(meta)
    if omitted and not a.allow_missing:
        raise SystemExit(json.dumps(omitted, indent=2) + "\nUse --allow-missing only to explicitly omit these cases.")
    if not rows:
        raise SystemExit("no complete NVE trajectories")
    summary = pd.DataFrame(rows)
    envelope = binned_envelope(curves, a.duration_ps, a.bins)
    maximum = summary.max_abs_drift_meV_per_atom.to_numpy()
    slopes = summary.linear_drift_meV_per_atom_per_ps.to_numpy() * 1000
    worst = summary.iloc[int(np.argmax(maximum))]
    stable = int((maximum <= a.limit).sum())
    protocols = [m.get("protocol", {}) for m in metadata]
    protocol_keys = set().union(*(set(p) for p in protocols))
    shared_protocol = {key: protocols[0].get(key) for key in sorted(protocol_keys)
                       if all(p.get(key) == protocols[0].get(key) for p in protocols)}
    families = int(summary.split_group.nunique())
    group_counts = {name: int(mask.sum()) for name, mask, _, _ in source_series(summary)}
    report = {
        "manifest_records": len(manifest), "complete_records": len(summary), "omitted": omitted,
        "analysis_start_ps": 0.0, "analysis_end_ps": a.duration_ps,
        "n_atoms_min": int(summary.n_atoms.min()), "n_atoms_max": int(summary.n_atoms.max()),
        "raw_source_counts": {k: int(v) for k, v in summary.source.value_counts().items()},
        "display_source_counts": group_counts, "distinct_manifest_family_labels": families,
        "recorded_split_counts": {k: int(v) for k, v in summary.recorded_split.value_counts().items()},
        "recorded_dtype_counts": {k: int(v) for k, v in pd.Series([m.get("dtype", "unknown") for m in metadata]).value_counts().items()},
        "shared_protocol": shared_protocol, "criterion_meV_per_atom": a.limit, "passing_records": stable,
        "largest_excursion_meV_per_atom": float(maximum.max()), "worst_record_id": worst.record_id,
        "median_max_excursion_meV_per_atom": float(np.median(maximum)),
        "largest_absolute_slope_microeV_per_atom_per_ps": float(np.abs(slopes).max()),
        "median_slope_microeV_per_atom_per_ps": float(np.median(slopes)),
        "binned_envelope": {"bins": a.bins, "width_ps": a.duration_ps / a.bins,
                            "range": "extrema over structures and all logged samples in each time bin",
                            "mean": "equal-weight mean of per-structure bin means"},
        "max_energy_sum_error_eV": float(summary.energy_sum_max_error_eV.max()) if summary.energy_sum_max_error_eV.notna().any() else None,
        "max_energy_per_atom_error_eV": float(summary.energy_per_atom_max_error_eV.max()) if summary.energy_per_atom_max_error_eV.notna().any() else None,
        "max_volume_relative_range": float(summary.volume_relative_range.max()) if summary.volume_relative_range.notna().any() else None,
        "notes": ["Run names such as holdout40 do not establish the number of families in the supplied manifest.",
                  "Training-family disjointness cannot be checked without the training split assignments.",
                  "All metrics use the same analysis window and the first NVE sample as the energy reference."],
    }
    if a.reference_summary:
        report["reference_summary_comparison"] = compare_summary(a.reference_summary, summary)
    summary.to_csv(out / f"{a.prefix}_energy_drift_summary.csv", index=False)
    envelope.to_csv(out / f"{a.prefix}_binned_envelope.csv", index=False)
    (out / f"{a.prefix}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    source_description = " and ".join(f"{count} {'constructed' if name == 'Own' else name}"
                                      for name, count in group_counts.items())
    caption = (f"Figure X. NVE energy stability of Local-MoE for {len(summary)} SiO2 interface structures "
               f"({source_description} interfaces), "
               f"containing {int(summary.n_atoms.min())}-{int(summary.n_atoms.max())} atoms. "
               "(a) Change in total energy per atom relative to the first NVE sample. The shaded band shows "
               f"the minimum and maximum across all structures and samples in {a.duration_ps / a.bins:g} ps time bins; "
               "the line is the equal-weight mean of the per-structure bin means. "
               f"(b) Maximum absolute energy excursion and (c) ordinary-least-squares linear drift rate, "
               f"both calculated over the same 0-{a.duration_ps:g} ps interval. "
               f"{stable} of {len(summary)} structures satisfy the chosen maximum-excursion criterion of "
               f"{a.limit:g} meV/atom; the largest excursion is {maximum.max():.6f} meV/atom. ")
    if all(key in shared_protocol for key in ("dt_fs", "npt_ps", "temperature_K", "pressure_bar", "nve_integrator")):
        caption += (f"Following structural relaxation and {shared_protocol['npt_ps']:g} ps NPT equilibration at "
                    f"{shared_protocol['temperature_K']:g} K and {shared_protocol['pressure_bar']:g} bar, velocities "
                    f"were rescaled to an initial NVE temperature of {shared_protocol['temperature_K']:g} K. "
                    f"The reported NVE protocol used {shared_protocol['nve_integrator']} with a "
                    f"{shared_protocol['dt_fs']:g} fs timestep and fixed cell volume.")
    (out / f"{a.prefix}_caption.txt").write_text(caption.strip() + "\n")
    if a.paper:
        paper_figure(a, out, summary, envelope)
    else:
        diagnostic_figures(a, out, summary, curves)
    print(f"Complete trajectories: {len(summary)}/{len(manifest)}; common window 0-{a.duration_ps:g} ps")
    print(f"Source groups: {group_counts}; distinct manifest family labels: {families}")
    print(f"Maximum excursion: {maximum.max():.9f} meV/atom ({worst.record_id})")
    print(f"Criterion {a.limit:g} meV/atom: {stable}/{len(summary)} pass")
    print(f"Largest |slope|: {np.abs(slopes).max():.9f} microeV/atom/ps")
    if omitted:
        print("Explicit omissions:", json.dumps(omitted))
    print("Outputs:", out)


if __name__ == "__main__":
    main()
