#!/usr/bin/env python3
"""
routing_from_md.py

Routing weights of the holdout-40 Local-MoE along the NVE trajectories, and the routing figures.

For every finished structure in the manifest the script evaluates the model on each saved NVE frame
(nve_10ps_frames.extxyz, one frame every 200 steps = 0.1 ps, 101 frames incl. the first) and averages the 7 routing
weights w_ik over the atoms of each region, then over the frames:

  all            every atom (every structure)
  interface      region 'interface'    )  only structures with region labels (the Own_augmented ones);
  phase1_bulk    region 'phase1_bulk'  )  the masks are the ones the trainer uses for the test metrics, written
  phase2_bulk    region 'phase2_bulk'  )  by run_local_moe_md.py into case_*/regions.npz

Outputs in --out-dir
  routing_nve_region_means.csv       structure x region: mean weight per expert (+ std over frames)
  routing_nve_per_frame.csv          structure x region x frame (optional, --per-frame)
  <prefix>_all_atoms.{png,pdf}       stacked bars, all-atom mean routing weight, every structure
  <prefix>_three_regions.{png,pdf}   three panels (phase 1 bulk | interface centre | phase 2 bulk), structures with labels
  and a printed table of the region means.

Needs the three checkpoints and the descriptor (same defaults as run_local_moe_md.py) and a GPU (or --device cpu).
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
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HET = (_os.environ.get("MOE_HET", "."))
SOURCE_ORDER = {"Oishi_relaxed52": 0, "Oishi_all179": 0, "Siddharth": 1, "Own_augmented": 2}
REGIONS = ("all", "phase1_bulk", "interface", "phase2_bulk")

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12, "font.weight": "bold",
    "axes.labelsize": 16, "axes.labelweight": "bold", "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.linewidth": 1.4, "xtick.labelsize": 12, "ytick.labelsize": 9, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "xtick.major.size": 6, "xtick.major.width": 1.2,
    "xtick.minor.visible": True, "xtick.minor.size": 3, "legend.fontsize": 11, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md-root", type=Path, default=Path(HET) / "Local_MoE_MATPES_R2SCAN_Expanded_v1/md_holdout40_v1")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None, help="default: <md-root>/figures")
    p.add_argument("--prefix", default="LocalMoE_holdout40_routing")
    p.add_argument("--common-checkpoint", type=Path, default=None)
    p.add_argument("--routing-checkpoint", type=Path, default=None)
    p.add_argument("--c1b-checkpoint", type=Path, default=None)
    p.add_argument("--torchsoap-module", type=Path, default=None)
    p.add_argument("--test-csv", type=Path, default=None,
                   help="test.csv of the dataset, for readable interface names (default: <proj>/data/expanded_holdout40_v1/test.csv)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    p.add_argument("--per-frame", action="store_true", help="also write routing_nve_per_frame.csv")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--recompute", action="store_true", help="ignore a cached routing_nve_region_means.csv")
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def region_label(r):
    return {"all": "All atoms", "phase1_bulk": "Phase 1 bulk", "interface": "Interface center",
            "phase2_bulk": "Phase 2 bulk"}[r]


def main():
    a = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    out = a.out_dir or a.md_root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    proj = Path(HET) / "Local_MoE_MATPES_R2SCAN_Expanded_v1"
    paths = {
        "common": a.common_checkpoint or proj / "runs/common_holdout40_ew10000_v1/best_checkpoint.pt",
        "routing": a.routing_checkpoint or proj / "runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt",
        "c1b": a.c1b_checkpoint or proj / "runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt",
        "soap": a.torchsoap_module or REPO_ROOT / 'src/local_moe/torchsoap_autograd_module_periodic_regnorm_nl.py',
    }
    man = pd.read_csv(a.manifest or a.md_root / "md_manifest.csv")
    man["_o"] = man["source"].map(SOURCE_ORDER).fillna(1)
    man = man.sort_values(["_o", "split_group", "record_id"]).reset_index(drop=True)
    cache = out / "routing_nve_region_means.csv"

    def finished(rid):
        d = a.md_root / f"case_{rid}"
        return all((d / f).is_file() for f in ("nve_10ps_frames.extxyz", "regions.npz", "nve_final_structure.extxyz"))

    done_ids = [r for r in man["record_id"] if finished(r)]
    missing = [r for r in man["record_id"] if r not in set(done_ids)]
    if missing:
        msg = f"{len(missing)} of {len(man)} structures have no finished NVE run: {missing}"
        if not a.allow_missing:
            raise SystemExit(msg + "\n(use --allow-missing to use the finished ones)")
        print("WARNING:", msg)

    # The cache is per structure: only structures that are finished and not yet in it are evaluated, so a later run
    # (after more MD has finished) adds to it instead of reusing a stale table.
    cached = None
    if cache.is_file() and not a.recompute and not a.per_frame:
        cached = pd.read_csv(cache)
        cached = cached[cached["record_id"].isin(done_ids)]
    have = set(cached["record_id"]) if cached is not None and len(cached) else set()
    need = [r for r in done_ids if r not in have]
    experts = ([c[len("w_mean_"):] for c in cached.columns if c.startswith("w_mean_")] if have else None)
    print(f"finished {len(done_ids)} | already in cache {len(have)} | to evaluate {len(need)}", flush=True)
    new_rows, frame_rows = [], []
    if need:
        from ase.io import read
        from local_moe_ase import LocalMoECalculator
        for k, v in paths.items():
            if not Path(v).expanduser().is_file():
                raise SystemExit(f"MISSING FILE: {v}")
        calc = LocalMoECalculator(paths["common"], paths["routing"], paths["c1b"], paths["soap"], device=a.device,
                                  dtype=a.dtype)
        experts = list(calc.expert_names)
        byid = man.set_index("record_id")
        for rid in need:
            r = byid.loc[rid]
            z = np.load(a.md_root / f"case_{rid}" / "regions.npz")
            has_regions = bool(z["has_regions"])
            masks = {"all": np.ones(int(r["n_atoms"]), dtype=bool)}
            if has_regions:
                masks.update({k: z[f"mask_{k}"].astype(bool) for k in ("phase1_bulk", "interface", "phase2_bulk")})
            traj = read(str(a.md_root / f"case_{rid}" / "nve_10ps_frames.extxyz"), index=":")
            if len(traj[0]) != len(masks["all"]):
                raise SystemExit(f"{rid}: frames have {len(traj[0])} atoms, manifest says {r['n_atoms']}")
            per = {k: [] for k in masks}
            for fi, at in enumerate(traj):
                w = calc.per_atom_terms(at)["w"]                      # (N, 7)
                for k, m in masks.items():
                    per[k].append(w[m].mean(axis=0))
                    if a.per_frame:
                        frame_rows.append({"record_id": rid, "region": k, "frame": fi,
                                           **{f"w_{e}": float(v) for e, v in zip(experts, per[k][-1])}})
            for k, lst in per.items():
                arr = np.asarray(lst)
                new_rows.append({"record_id": rid, "label": r["label"], "source": r["source"],
                                 "split_group": r["split_group"], "n_atoms": int(r["n_atoms"]), "region": k,
                                 "n_region_atoms": int(masks[k].sum()), "n_frames": len(arr),
                                 **{f"w_mean_{e}": float(v) for e, v in zip(experts, arr.mean(axis=0))},
                                 **{f"w_std_frames_{e}": float(v) for e, v in zip(experts, arr.std(axis=0))}})
            print(f"  {rid:14s} N={int(r['n_atoms']):5d}  frames={len(traj):3d}  regions={'yes' if has_regions else 'no '}",
                  flush=True)
    if not (have or need):
        raise SystemExit("no finished runs")
    parts = ([cached] if have else []) + ([pd.DataFrame(new_rows)] if new_rows else [])
    means = pd.concat(parts, ignore_index=True)
    rank = {rid: i for i, rid in enumerate(man["record_id"])}
    means = means.assign(_r=means["record_id"].map(rank)).sort_values(["_r"], kind="stable").drop(columns="_r") \
        .reset_index(drop=True)
    means.to_csv(cache, index=False)
    if a.per_frame and frame_rows:
        pd.DataFrame(frame_rows).to_csv(out / "routing_nve_per_frame.csv", index=False)
    print(f"wrote {cache} ({means['record_id'].nunique()} structures)")

    colors = [plt.get_cmap("tab10")(i) for i in range(len(experts))]
    names = [e.replace("_", "-") for e in experts]
    wcols = [f"w_mean_{e}" for e in experts]

    def stacked(ax, sub, show_y=True, label_col="ylabel"):
        y = np.arange(len(sub))
        left = np.zeros(len(sub))
        for c, col, nm in zip(colors, wcols, names):
            v = sub[col].to_numpy()
            ax.barh(y, v, left=left, height=0.78, color=c, edgecolor="none", label=nm)
            left += v
        ax.set_xlim(0, 1)
        ax.set_ylim(len(sub) - 0.4, -0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(sub[label_col].tolist(), fontweight="bold")
        if not show_y:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("Mean routing weight", labelpad=6)
        for t in ax.get_xticklabels():
            t.set_fontweight("bold")

    # ------------------------------------------------------------------ all-atom figure
    allm = means[means["region"] == "all"].copy()
    allm["ylabel"] = allm["label"] + "  N=" + allm["n_atoms"].astype(str)
    height = max(4.0, 0.2 * len(allm) + 1.6)
    fig, ax = plt.subplots(figsize=(9.0, height))
    stacked(ax, allm)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=11)
    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.5 / height + 0.04, top=1 - 1.0 / height)
    p1, p2 = out / f"{a.prefix}_all_atoms.png", out / f"{a.prefix}_all_atoms.pdf"
    fig.savefig(p1, dpi=a.dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(p2, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    saved = [p1, p2]

    # ------------------------------------------------------------------ three-region figure
    have = means[means["region"] == "phase1_bulk"]["record_id"].tolist()
    if have:
        def rec_no(r):
            return int("".join(ch for ch in str(r) if ch.isdigit()) or 0)

        ids = sorted((r for r in allm["record_id"] if r in set(have)), key=rec_no)     # record order, as in the old figure
        order = {r: i for i, r in enumerate(ids)}
        # readable names: "02_Cristobalite__07_AFI" -> "Cristobalite / AFI" (phase 1 / phase 2), (a)/(b) for the pair
        names_by_id = {r: r for r in ids}
        tcsv = a.test_csv or proj / "data/expanded_holdout40_v1/test.csv"
        if Path(tcsv).is_file():
            tdf = pd.read_csv(tcsv)
            if {"record_id", "family"} <= set(tdf.columns):
                fam = dict(zip(tdf["record_id"].astype(str), tdf["family"].astype(str)))
                seen = {}
                for r in ids:
                    f = fam.get(str(r))
                    if f is None or "__" not in f:
                        continue
                    parts = [re.sub(r"^\d+_", "", x) for x in f.split("__")]
                    base = " / ".join(parts)
                    seen[base] = seen.get(base, 0) + 1
                    names_by_id[r] = f"{base} ({'ab'[min(seen[base] - 1, 1)]})"
        else:
            print(f"(no {tcsv}: three-region figure labelled by record id)")
        panels = [("phase1_bulk", "(a)", "Phase 1 bulk"), ("interface", "(b)", "Interface center"),
                  ("phase2_bulk", "(c)", "Phase 2 bulk")]
        height = max(4.5, 0.27 * len(ids) + 2.0)
        fig, axes = plt.subplots(1, 3, figsize=(15.5, height))
        for k, (ax, (reg, tag, title)) in enumerate(zip(axes, panels)):
            sub = means[(means["region"] == reg) & means["record_id"].isin(ids)].copy()
            sub["o"] = sub["record_id"].map(order)
            sub = sub.sort_values("o")
            sub["ylabel"] = sub["record_id"].map(names_by_id)
            stacked(ax, sub, show_y=(k == 0))
            ax.set_title(title, pad=8)
            ax.text(0.0, 1.015, tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=15, fontweight="bold")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=11)
        fig.subplots_adjust(left=0.165, right=0.985, bottom=0.8 / height + 0.04, top=1 - 1.25 / height, wspace=0.07)
        p3, p4 = out / f"{a.prefix}_three_regions.png", out / f"{a.prefix}_three_regions.pdf"
        fig.savefig(p3, dpi=a.dpi, bbox_inches="tight", pad_inches=0.08)
        fig.savefig(p4, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        saved += [p3, p4]
    else:
        print("no structure with region labels among the finished runs: three-region figure skipped")

    # ------------------------------------------------------------------ console table
    print("\nMean routing weight (mean over structures; the spread is the standard deviation across structures)")
    print(f"{'region':18s} {'structures':>10s}  " + "  ".join(f"{n:>13s}" for n in names))
    for reg in REGIONS:
        sub = means[means["region"] == reg]
        if sub.empty:
            continue
        cells = [f"{sub[c].mean():6.3f}+-{sub[c].std(ddof=0):5.3f}" for c in wcols]
        print(f"{region_label(reg):18s} {len(sub):10d}  " + "  ".join(f"{c:>13s}" for c in cells))
    dom = allm[wcols].to_numpy().argmax(axis=1)
    print("\ndominant channel (all-atom mean), number of structures: "
          + ", ".join(f"{names[i]} {int((dom == i).sum())}" for i in sorted(set(dom))))
    print("\nSaved:")
    for p in saved + [cache]:
        print(" ", p)


if __name__ == "__main__":
    main()
