#!/usr/bin/env python3
"""
make_oishi_gamma_table_and_parity.py

Two-panel table and parity plot of Oishi interface energies, MATPes-r2SCAN vs
the expanded Local-MoE model (trained on Oishi, Own and Siddharth), all
recomputed on identical MATPes-relaxed geometries in J/m^2.

Input: oishi_gamma_candidates.csv, written by build_oishi_gamma_candidates.py
       (columns: case, gamma_MATPES_J_m2, gamma_LocalMoE_J_m2, origin,
        pool_split, drop_reason). Rows with a non-empty drop_reason are excluded.

Each row carries the expanded model's own training status for that AB record:
train, val (used to choose the checkpoint) or test (never used). The table
shades val and test rows so unseen structures are visible.

Run from External_Benchmark_Oishi2025/:
    python make_oishi_gamma_table_and_parity.py

Styling follows make_relaxed37_interface_energy_table_samegeom_corrected_v1.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="oishi_gamma_candidates.csv")
ap.add_argument("--out-dir", default="Oishi_Gamma_Table_Expanded4608_v1")
ap.add_argument("--stem", default="oishi_gamma_expanded4608")
ap.add_argument("--figheight", type=float, default=None)
ap.add_argument("--model-label", default="Local-MoE", help="model name used in the subtitle, axis, title and footers")
ap.add_argument("--header-label", default=None, help="short model name for the table column header (default: --model-label)")
ap.add_argument("--trained-on", default="trained on Oishi, Own and Siddharth", help="phrase after the model name in the subtitle")
ap.add_argument("--table-font", type=float, default=14.5, help="table body font size in points (header, title, footers scale with it)")
ap.add_argument("--keep-set-column", action="store_true", help="show the Set column even when every row has the same status")
ap.add_argument("--extra-footer", default="", help="one extra footnote line (e.g. what was excluded)")
ap.add_argument("--model-col", default="gamma_LocalMoE_J_m2",
                help="column with the model's gamma (e.g. gamma_full_J_m2 in gamma66_fullmoe_*.csv)")
ap.add_argument("--split-col", default="pool_split",
                help="column with train/val/test status (e.g. new_split in gamma66_fullmoe_*.csv)")
ap.add_argument("--ref-label", default="MATPes",
                help="name of the reference model in labels (e.g. MACE-MatPES); '-r2SCAN' is appended")
ap.add_argument("--ref-header", default=None,
                help="short reference name for the table column header (default: --ref-label); use e.g. MatPES so the header fits")
ap.add_argument("--paper", action="store_true",
                help="table for a paper: no title, subtitle or footnotes inside the image (they go in the LaTeX caption); "
                     "written as table_<stem>_paper.*, the titled table is left alone")
ap.add_argument("--only-unseen", action="store_true",
                help="list only rows the model did not train on (validation and test)")
args = ap.parse_args()

OUT = Path(args.out_dir)
LABEL = args.model_label
REF = args.ref_label
HEAD = args.header_label or LABEL
if args.only_unseen:
    args.stem += "_unseen"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 12, "font.weight": "bold",
    "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def pretty_index(x):
    return str(x).replace("-", "\u2212")


def pretty_case(case):
    """AFT_0_0_1_EAB_0_0_1 -> AFT (001) / EAB (001)"""
    t = str(case).split("_")
    if len(t) < 8:
        return str(case).replace("__", " / ")
    o1 = "".join(pretty_index(v) for v in t[1:4])
    o2 = "".join(pretty_index(v) for v in t[5:8])
    return f"{t[0]} ({o1}) / {t[4]} ({o2})"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
cand = pd.read_csv(args.csv, keep_default_na=False)   # keep empty drop_reason as "", never NaN
for src, dst in ((args.model_col, "gamma_LocalMoE_J_m2"), (args.split_col, "pool_split")):
    assert src in cand.columns, f"column '{src}' not found in {args.csv}; columns: {list(cand.columns)}"
    if src != dst:
        cand[dst] = cand[src]
if "drop_reason" not in cand.columns:
    cand["drop_reason"] = ""          # input already contains only the rows to use
need = ["case", "gamma_MATPES_J_m2", "gamma_LocalMoE_J_m2", "pool_split", "drop_reason"]
assert not (set(need) - set(cand.columns)), f"missing columns: {sorted(set(need) - set(cand.columns))}"
for c in ("gamma_MATPES_J_m2", "gamma_LocalMoE_J_m2"):
    cand[c] = pd.to_numeric(cand[c], errors="raise")

df = cand[cand.drop_reason.astype(str) == ""].copy()
assert len(df) > 0, "no rows left after the drop filter"
assert (df.gamma_MATPES_J_m2 > 0).all(), "a row with MATPes gamma <= 0 slipped through"
assert df.case.is_unique, "duplicate cases"
bad = sorted(set(df.pool_split) - {"train", "val", "test"})
assert not bad, f"unexpected pool_split values: {bad}"

n_before = len(df)
if args.only_unseen:
    df = df[df.pool_split != "train"].copy()
    assert len(df) > 0, "no unseen rows"
    print(f"--only-unseen: keeping {len(df)} of {n_before} rows (validation and test)")
figheight = args.figheight if args.figheight is not None else max(
    5.6, 2.55 + (int(np.ceil(len(df) / 2)) + 1.4) * 0.40 * args.table_font / 14.5)
if args.paper and args.figheight is None:
    figheight -= (1.05 - 0.10) + (1.55 - 0.10)          # the title block and footnote block are not drawn
df["err"] = df.gamma_LocalMoE_J_m2 - df.gamma_MATPES_J_m2
df = df.sort_values(["gamma_MATPES_J_m2", "case"]).reset_index(drop=True)
df["Interface"] = df.case.map(pretty_case)
n = len(df)
dropped = cand[cand.drop_reason.astype(str) != ""].drop_reason.value_counts().to_dict()
cnt = df.pool_split.value_counts().to_dict()
print(f"{n} Oishi interfaces kept; dropped: {dropped}")
print(f"training status of the kept rows: {cnt}")


def stats(g):
    x, y = g.gamma_MATPES_J_m2.to_numpy(), g.gamma_LocalMoE_J_m2.to_numpy()
    e = y - x
    r = {"N": len(g), "MAE": np.abs(e).mean(), "bias": e.mean(), "RMSE": np.sqrt((e ** 2).mean())}
    if len(g) >= 8:
        r["Pearson"] = np.corrcoef(x, y)[0, 1]
        r["slope"] = np.polyfit(x, y, 1)[0]
    return r


groups = {"all": df, "train": df[df.pool_split == "train"], "val + test": df[df.pool_split != "train"],
          "val": df[df.pool_split == "val"], "test": df[df.pool_split == "test"]}
st = pd.DataFrame({k: stats(v) for k, v in groups.items() if len(v)}).T
print("\nSTATISTICS (J/m2)")
print(st.to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\nerrors positive on {int((df.err > 0).sum())} of {n} rows; "
      f"unseen (val+test) positive on {int((groups['val + test'].err > 0).sum())} of {len(groups['val + test'])}")
st.to_csv(OUT / f"{args.stem}_stats.csv")
df.to_csv(OUT / f"{args.stem}_rows.csv", index=False)

# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
n_block = int(np.ceil(n / 2))
left, right = df.iloc[:n_block], df.iloc[n_block:]
FS = args.table_font
show_set = df.pool_split.nunique() > 1 or args.keep_set_column      # a Set column / shading only helps when statuses differ
ncol = 5 if show_set else 4
spacer = ncol


def make_rows(block, start):
    out = []
    for k, r in enumerate(block.itertuples()):
        row = [str(start + k + 1), r.Interface, f"{r.gamma_MATPES_J_m2:.4f}", f"{r.gamma_LocalMoE_J_m2:.4f}"]
        out.append(row + ([r.pool_split] if show_set else []))
    return out


lr, rr = make_rows(left, 0), make_rows(right, n_block)
while len(rr) < n_block:
    rr.append([""] * ncol)
combined = [a_ + [""] + b_ for a_, b_ in zip(lr, rr)]
panel_head = ["No.", "Interface", (args.ref_header or REF) + r" $\gamma$", HEAD + r" $\gamma$"] + (["Set"] if show_set else [])
headers = panel_head + [""] + panel_head
status = [list(left.pool_split) + [""] * (n_block - len(left)), list(right.pool_split) + [""] * (n_block - len(right))]

fig, ax = plt.subplots(figsize=(18.0, figheight))
ax.axis("off")
title = ("Interface energies of Oishi SiO$_2$ heterostructures not used for training" if args.only_unseen
         else "Interface energies of Oishi SiO$_2$ heterostructures")
H = figheight   # margins below are absolute inches, so the layout holds at any figure height
TOP, BOT = (0.10, 0.10) if args.paper else (1.05, 1.55)


def put(*a, **k):
    """ax.text for the title, subtitle and footnotes, skipped in --paper mode"""
    if not args.paper:
        ax.text(*a, **k)


put(0.5, 1 - 0.04 / H, title, transform=ax.transAxes, ha="center", va="top", fontsize=FS + 9.5, fontweight="bold")
put(0.5, 1 - 0.50 / H, r"$\gamma$ in J m$^{-2}$; " + REF + r"-r$^2$SCAN and " + LABEL + " (" + args.trained_on + ") "
        "evaluated on identical " + REF + "-relaxed geometries", transform=ax.transAxes, ha="center", va="top", fontsize=FS - 0.5)

cw = [0.050, 0.235, 0.095, 0.115] + ([0.075] if show_set else [])
col_widths = cw + [0.02] + cw
tbl = ax.table(cellText=combined, colLabels=headers, cellLoc="center", colLoc="center",
               colWidths=col_widths, bbox=[0.015, BOT / H, 0.970, 1 - (TOP + BOT) / H])
tbl.auto_set_font_size(False)
tbl.set_fontsize(FS)
SHADE = {"test": "#dcefdc", "val": "#fbefcf"}
iface_cols = (1, ncol + 2)
for (row, col), cell in tbl.get_celld().items():
    cell.set_linewidth(0.6)
    cell.set_edgecolor("0.55")
    if row == 0:
        cell.set_text_props(weight="bold", fontsize=FS + 0.5)
        cell.set_facecolor("0.90")
        cell.set_height(0.040)
    else:
        cell.set_height(0.030)
        base = "0.955" if row % 2 == 0 else "white"
        panel = 0 if col < ncol else 1
        st_here = status[panel][row - 1] if (col != spacer and show_set) else ""
        cell.set_facecolor(SHADE.get(st_here, base))
    if col == spacer:
        cell.set_facecolor("white")
        cell.visible_edges = "open"            # no border lines, so the neighbouring panels keep complete borders
    if col in iface_cols:
        cell.get_text().set_ha("left")
        cell.PAD = 0.04

ex = "; ".join(f"{v} with {REF} $\\gamma\\leq0$" if k == "MATPes gamma <= 0" else f"{v} with {k}" for k, v in dropped.items())
if args.only_unseen:
    l1 = f"{n} Oishi interfaces with {REF} $\\gamma>0$ that the model was not fitted on."
    l2 = f"Test (never used for fitting or selection): {cnt.get('test', 0)}" + (
        f"; validation (used to select the checkpoint): {cnt['val']}." if cnt.get("val", 0) else ".")
    l3 = (("Green = test, yellow = validation." if cnt.get("val", 0) else "Green = test.") if show_set else "")
    if args.extra_footer:
        l3 = (l3 + " " if l3 else "") + args.extra_footer
else:
    l1 = f"{n} Oishi interfaces with {REF} $\\gamma>0$ are listed." + (f" Excluded: {ex}." if ex else "")
    if args.extra_footer:
        l1 += " " + args.extra_footer
    l2 = (f"{LABEL} training status of each AB record: train {cnt.get('train', 0)}, "
          f"validation {cnt.get('val', 0)}, test {cnt.get('test', 0)}.")
    l3 = "Shaded rows were not used for training: green = test, yellow = validation (used to select the checkpoint)."
put(0.5, 1.12 / H, l1, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS - 1.0, fontweight="bold")
put(0.5, 0.70 / H, l2, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS - 1.0, fontweight="bold")
if l3:
    put(0.5, 0.26 / H, l3, transform=ax.transAxes, ha="center", va="bottom", fontsize=FS - 1.5, fontweight="normal")
for ext in ("png", "pdf", "svg"):
    p = OUT / f"table_{args.stem}{'_paper' if args.paper else ''}.{ext}"
    fig.savefig(p, dpi=600, bbox_inches="tight") if ext == "png" else fig.savefig(p, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# parity plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.4, 6.8))
sty = {"train": dict(marker="o", s=52, c="#2171b5", alpha=0.75, edgecolor="black", lw=0.5, zorder=3),
       "val": dict(marker="D", s=70, c="#ff7f0e", alpha=0.95, edgecolor="black", lw=0.6, zorder=4),
       "test": dict(marker="s", s=76, c="#2ca25f", alpha=0.95, edgecolor="black", lw=0.6, zorder=5)}
lab = {"train": "AB record in training set", "val": "validation (checkpoint selection)", "test": "test (never used)"}
for k in ("train", "val", "test"):
    g = df[df.pool_split == k]
    if len(g):
        ax.scatter(g.gamma_MATPES_J_m2, g.gamma_LocalMoE_J_m2, label=f"{lab[k]} (N = {len(g)})", **sty[k])
v = np.concatenate([df.gamma_MATPES_J_m2, df.gamma_LocalMoE_J_m2])
lo, hi = 0.0, v.max() * 1.06
ax.plot([lo, hi], [lo, hi], "--", color="0.25", lw=1.6, label="y = x", zorder=2)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(REF + r"-r$^2$SCAN interface energy $\gamma$ (J m$^{-2}$)")
ax.set_ylabel(LABEL + r" interface energy $\gamma$ (J m$^{-2}$)")
ax.set_title("Oishi interface energies, " + LABEL + " vs " + REF + "-r$^2$SCAN", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
a, u = st.loc["all"], (st.loc["val + test"] if "val + test" in st.index else None)
if args.only_unseen:
    txt = (f"N = {int(a.N)}: MAE = {a.MAE:.4f}, bias = {a.bias:+.4f} J m$^{{-2}}$\n"
           f"  Pearson r = {a.Pearson:.2f}, slope = {a.slope:.2f}\n"
           f"  errors > 0: {int((df.err > 0).sum())} of {len(df)}" + (f"\ntest only N = {int(st.loc['test'].N)}: MAE = {st.loc['test'].MAE:.4f}, "
           f"bias = {st.loc['test'].bias:+.4f}" if ("test" in st.index and "val" in st.index) else ""))
else:
    txt = (f"all N = {int(a.N)}: MAE = {a.MAE:.4f}, bias = {a.bias:+.4f} J m$^{{-2}}$\n"
           f"  Pearson r = {a.Pearson:.2f}, slope = {a.slope:.2f}")
    if u is not None:
        txt += f"\nunseen N = {int(u.N)}: MAE = {u.MAE:.4f}, bias = {u.bias:+.4f}"
ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=10.5, va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.6", alpha=0.92))
fig.tight_layout()
fig.savefig(OUT / f"parity_{args.stem}.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT / f"parity_{args.stem}.pdf", bbox_inches="tight")
plt.close(fig)
print("\nsaved to", OUT)
