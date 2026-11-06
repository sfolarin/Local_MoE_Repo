# Routing diagnostics on the repaired model

Per-expert routing weights along the 52 ten-picosecond NVE trajectories of the
protocol cohort, evaluated with the selected Local-MoE.

## Why this was regenerated

The routing figures in `<proj>/md_holdout40_v1/figures/` are dated 18 October and
were produced from pre-repair trajectories with the pre-repair checkpoints, which
is what `routing_from_md.py` resolves to when its artifact arguments are left
unset. They describe a model the paper does not report. `routing_from_md.py` also
caches `routing_nve_region_means.csv` per structure, so the regeneration used a
fresh output directory and `--recompute`.

## Model

| role | path | sha256 (first 24) |
|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |

## Method

Every saved NVE frame is evaluated: one per 200 steps = 0.1 ps, 101 frames per
structure including the first. The seven routing weights are averaged **over the
atoms of each region first, then over the frames**.

Regions come from `case_*/regions.npz`: `all` for every structure, and
`interface`, `phase1_bulk`, `phase2_bulk` for the 22 structures carrying region
labels. All 22 are `Own_augmented`. Phase 1 and phase 2 follow the order of the
two halves of the family string, so `02_Cristobalite__07_AFI` means phase 1 =
Cristobalite, phase 2 = AFI.

Expert order, which fixes the colour assignment in every figure:
`ML-BKS, Original_BKS, Chik_BKS, PSO1_BKS, PSO2_BKS, Tersoff, Vashishta`.

## Result

Mean weight, with the standard deviation **across structures**:

| region | n | ML-BKS | Original_BKS | Chik_BKS | PSO1_BKS | PSO2_BKS | Tersoff | Vashishta |
|---|---|---|---|---|---|---|---|---|
| all atoms | 52 | 0.016+-0.031 | 0.001+-0.001 | 0.079+-0.030 | 0.018+-0.011 | 2.06e-04 | 0.112+-0.035 | 0.774+-0.059 |
| phase 1 bulk | 22 | 0.003+-0.004 | 0.001+-0.000 | 0.056+-0.018 | 0.020+-0.011 | 2.08e-04 | 0.168+-0.094 | 0.751+-0.093 |
| interface centre | 22 | 0.036+-0.022 | 0.003+-0.001 | 0.077+-0.011 | 0.013+-0.006 | 2.22e-04 | 0.116+-0.024 | 0.755+-0.029 |
| phase 2 bulk | 22 | 0.007+-0.008 | 0.001+-0.000 | 0.057+-0.018 | 0.018+-0.014 | 1.79e-04 | 0.101+-0.027 | 0.815+-0.035 |

### PSO2-BKS is small, not unused

An earlier version of this report rounded PSO2-BKS to 0.000 and then concluded
the channel was not exercised. That conclusion came from the rounding. The
measured values:

| region | mean | min | max | structures with exactly zero |
|---|---|---|---|---|
| all atoms | 2.058e-04 | 1.216e-04 | 4.755e-04 | 0 of 52 |
| phase 1 bulk | 2.081e-04 | 1.023e-04 | 4.908e-04 | 0 of 22 |
| interface centre | 2.218e-04 | 1.531e-04 | 3.041e-04 | 0 of 22 |
| phase 2 bulk | 1.791e-04 | 1.050e-04 | 3.478e-04 | 0 of 22 |

The channel carries a small nonzero weight in every structure and every region.

### The interface signature

Per structure, the interface/phase-1-bulk ratio of ML-BKS weight has **median
15.5x, range 5.2x to 124x, with 0 of 22 structures showing no enrichment**. This
median of per-structure ratios is a different quantity from the ratio of the
means above (0.036/0.003, about 12x); the per-structure median with its range is
the defensible statement.

Chik-BKS rises modestly at the interface (0.077 against 0.056 and 0.057) while
Tersoff falls relative to phase 1 bulk (0.116 against 0.168). Vashishta dominates
everywhere at 0.75-0.82 and is the largest channel for all 52 structures.

## Effect of the descriptor repair on routing

The largest change in any region-averaged weight is Vashishta at the interface,
-0.0138, and ML-BKS at the interface, +0.0063. Every qualitative conclusion is
unchanged; unlike the cost measurements, the routing story did not depend on the
descriptor defect.

## Figure mapping

| figure | displayed subset | ordering | panels |
|---|---|---|---|
| `LocalMoE_ep13_routing_all_atoms.{pdf,png}` | all **52** structures, region `all` | source group (Oishi, Siddharth, Own_augmented), then `split_group`, then `record_id` | one, stacked horizontal bars |
| `LocalMoE_ep13_routing_three_regions.{pdf,png}` | the **22** region-labelled structures | numeric part of `record_id` | three: (a) phase 1 bulk, (b) interface centre, (c) phase 2 bulk |

Both are generated from `metrics/routing_nve_region_means.csv`, which is the
regenerated selected-model data; no retained routing figure draws on pre-repair
data. **Neither figure is ordered by any descriptor or complexity quantity**, so
a caption claiming an ordering by descriptor complexity is incorrect.

The three-region figure shows 22 bars: eleven interface systems in two
configurations each, labelled `(a)` and `(b)` from the `family` column. A reader
counting interface systems sees eleven; a reader counting bars sees twenty-two.
No code path in `routing_from_md.py` produces a subset of 16.

## Two different spreads

`routing_nve_region_means.csv` carries both `w_mean_*` and `w_std_frames_*`. The
latter is the standard deviation **over the 101 frames** of a single structure;
the `+-` values in the table above are the standard deviation **across
structures**. A caption must say which one its error bars show.

## Limitations

- Region labels exist only for the 22 `Own_augmented` structures, so the
  interface/bulk contrast comes entirely from this work's constructed interfaces
  and not from the Oishi set.
- Weights are means over frames of a 10 ps trajectory at 300 K; no time
  dependence is reported.

## Files

`figures/` both figures as PDF and PNG. `metrics/` region means with per-frame
standard deviations, and the full per-frame table (11,919 rows). `logs/`,
`scripts/` the run log and submit script. Hashes in
`metrics/MANIFEST_SHA256.txt`.
