# Routing diagnostics on the repaired model

Per-expert routing weights along the 52 ten-picosecond NVE trajectories of the
protocol cohort, evaluated with the selected Local-MoE.

## Why this was regenerated

The routing figures in `<proj>/md_holdout40_v1/figures/` are dated 18 October
and were produced from pre-repair trajectories with the pre-repair checkpoints,
which are what `routing_from_md.py` resolves to when its artifact arguments are
left unset. They describe a model the paper does not report. This package
replaces them.

`routing_from_md.py` also caches `routing_nve_region_means.csv` per structure
and reuses rows for structures already present, so the regeneration used a fresh
output directory and `--recompute`.

## Model

| role | path | sha256 (first 24) |
|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |

## Method

Every saved NVE frame is evaluated: one frame per 200 steps = 0.1 ps, 101 frames
per structure including the first. The seven routing weights are averaged **over
the atoms of each region first, then over the frames** — not pooled over
atom-frames, which would weight large regions differently.

Regions come from `case_*/regions.npz` as written by the MD runner: `all` for
every structure, and `interface`, `phase1_bulk`, `phase2_bulk` for the 22
structures carrying region labels. All 22 are `Own_augmented`; the Oishi records
have no region annotation. Phase 1 and phase 2 follow the order of the two
halves of the family string, so `02_Cristobalite__07_AFI` means phase 1 =
Cristobalite, phase 2 = AFI.

Expert order, which fixes the colour assignment in every figure, is
`ML-BKS, Original_BKS, Chik_BKS, PSO1_BKS, PSO2_BKS, Tersoff, Vashishta`.

## Result

Mean weight, with the spread across structures:

| region | n | ML-BKS | Original_BKS | Chik_BKS | PSO1_BKS | PSO2_BKS | Tersoff | Vashishta |
|---|---|---|---|---|---|---|---|---|
| all atoms | 52 | 0.016+-0.031 | 0.001+-0.001 | 0.079+-0.030 | 0.018+-0.011 | 0.000+-0.000 | 0.112+-0.035 | 0.774+-0.059 |
| phase 1 bulk | 22 | 0.003+-0.004 | 0.001+-0.000 | 0.056+-0.018 | 0.020+-0.011 | 0.000+-0.000 | 0.168+-0.094 | 0.751+-0.093 |
| interface centre | 22 | 0.036+-0.022 | 0.003+-0.001 | 0.077+-0.011 | 0.013+-0.006 | 0.000+-0.000 | 0.116+-0.024 | 0.755+-0.029 |
| phase 2 bulk | 22 | 0.007+-0.008 | 0.001+-0.000 | 0.057+-0.018 | 0.018+-0.014 | 0.000+-0.000 | 0.101+-0.027 | 0.815+-0.035 |

The interface signature is ML-BKS enrichment. Per structure, the
interface/phase-1-bulk ratio of ML-BKS weight has **median 15.5x, range 5.2x to
124x, with 0 of 22 structures showing no enrichment**. Note that this median of
per-structure ratios is not the same quantity as the ratio of the means above
(0.036/0.003, about 12x); the per-structure median with its range is the
defensible statement.

Chik-BKS also rises modestly at the interface (0.077 against 0.056 and 0.057 in
the two bulks) while Tersoff falls relative to phase 1 bulk (0.116 against
0.168). Vashishta dominates everywhere at 0.75-0.82 and is the largest channel
for all 52 structures.

## Effect of the descriptor repair on routing

Comparing against the pre-repair table structure by structure, the largest
change in any region-averaged weight is Vashishta at the interface, -0.0138, and
ML-BKS at the interface, +0.0063. Every qualitative conclusion is unchanged.
Unlike the cost measurements, the routing story did not depend on the descriptor
defect.

## Two different spreads

`routing_nve_region_means.csv` carries both `w_mean_*` and `w_std_frames_*`. The
latter is the standard deviation **over the 101 frames** of a single structure.
The `+-` values in the table above are the standard deviation **across
structures**. They are not interchangeable, and a figure caption should say
which one its error bars show.

## Figure ordering

The all-atom figure orders structures by source group (Oishi, then Siddharth,
then Own_augmented), then `split_group`, then `record_id`. The three-region
figure orders by the numeric part of the record id. **Neither figure is ordered
by any descriptor or complexity quantity.** Any caption claiming an ordering by
descriptor complexity is incorrect.

The three-region figure shows **22 bars**, being the 22 region-labelled
structures. These are eleven interface systems in two configurations each,
labelled `(a)` and `(b)` from the `family` column of the test set, so a reader
counting distinct interface systems sees eleven and a reader counting bars sees
twenty-two.

## Limitations

- Region labels exist only for the 22 `Own_augmented` structures, so the
  interface/bulk contrast is drawn entirely from this work's constructed
  interfaces and not from the Oishi set.
- Weights are means over frames of a 10 ps trajectory at 300 K; no time
  dependence is reported here.
- `PSO2_BKS` is 0.000 to three decimals everywhere. It is not exercised by this
  cohort.

## Files

`figures/` the all-atom and three-region figures as PDF and PNG. `metrics/` the
region means with per-frame standard deviations, and the full per-frame table
(11,919 rows) for anyone wanting a different aggregation. `logs/`, `scripts/`
the run log and submit script. Hashes in `metrics/MANIFEST_SHA256.txt`.
