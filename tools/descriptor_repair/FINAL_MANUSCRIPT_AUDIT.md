# Final manuscript audit

Evidence baseline: this commit.

Quantitative results are reproduced in `audit/master_metrics.csv` at full
precision, each row naming its statistic, population, n, baseline and source
field. That table covers items 2 and 4 through 8. Items 1, 3 and 9 are
provenance, executed settings and deliverables rather than measurements: the
artifact hashes are in each package's `metrics/MANIFEST_SHA256.txt` and in the
execution logs, the executed settings are enumerated in section 3 with the file
each comes from, and section 9 lists what this commit contains.

Every claim is mapped in `audit/claim_to_evidence.csv`. Commands are in
`REPRODUCTION.md`.

This audit covers the **code, artifacts and results**. The manuscript, SI and
bibliography were not available while it was prepared, so every item that
requires reading them is marked **REQUIRES AUTHOR INPUT** and no statement is
made about what the manuscript currently says.

## Classification

| item | subject | status |
|---|---|---|
| 1 | model identity and provenance | **VERIFIED** |
| 2 | dataset, split and group structure | **VERIFIED** |
| 3 | methods: architecture, training, precision | **VERIFIED (executed settings)** / comparison REQUIRES AUTHOR INPUT |
| 4 | static energetics and interface energies | **VERIFIED** |
| 5 | routing diagnostics | **CORRECTED** / captions REQUIRES AUTHOR INPUT |
| 6 | dynamics, 10 ps and 500 ps | **CORRECTED** |
| 7 | computational cost | **CORRECTED** |
| 8 | invariance, force conservativity, stress | **VERIFIED, with coverage stated** |
| 9 | consolidated deliverables | **this commit** |

## 1. Model identity and provenance — VERIFIED

Four artifacts define the reported model, and all four are verified at full
length before and after every run in this package:

| role | sha256 (first 24) | selection |
|---|---|---|
| common network | `5b17849752a535739b16cfc8` | epoch 116 on val |
| routing gate | `9e8e0ca6af4976ef8c74c34e` | epoch 110 on val structure-balanced KL |
| C1b residual | `c7ad2a89c570549309ad8906` | epoch 13 on val loss |
| descriptor | `0afd9a1ba7b9a24f8bf8151a` | repaired module |

`pilot/common/training_contract.json` records the features as
`expanded_holdout40_v1_features_rotinv_v1` with
`source_sha256["torchsoap_autograd_module_periodic_regnorm.py"]` equal to the
repaired descriptor's hash, `initialization: fresh_common_network` and
`weights_discarded: true`. The common network was therefore trained from scratch
on features regenerated with the repaired descriptor, not warm-started from
pre-repair weights. The gate distilled from
`expanded_holdout40_v1_distillation_rotinv_v1`, and C1b chains to both.

**Contemporaneous versus post hoc.** Execution logs for the cost, routing,
invariance, 10 ps and 500 ps runs all print the four artifact hashes before work
begins; the 500 ps cohort has 19 such logs, all four hashes in each. The
`provenance_hashes.json` sidecars were written afterwards and say so, and alone
would establish only that the artifacts as they exist now match. The two agree.

**A documentation hazard.** The repaired descriptor is snapshotted under the
original filename `torchsoap_autograd_module_periodic_regnorm.py`. Only the
sha256 distinguishes it from the pre-repair module, and the SI should say so.

## 2. Dataset, split and group structure — VERIFIED

512 configurations in 261 groups, split 369 train / 38 val / 105 test, verified
against `data_splits/holdout40_records.csv` and reproduced independently. The
split summary records `features_generated: false` and the empty source-split
cells `Siddharth:val` and `Siddharth:test`, which is consistent: the 41
Siddharth configurations are training-only. `holdout40_split_summary.json`
carries the counts.

## 3. Methods — executed settings VERIFIED; comparison REQUIRES AUTHOR INPUT

Recorded in `pilot/common/training_contract.json` and the two
`training_info.json` files, not inferred:

energy reference `-9.668443844547909` eV/atom under `preserve_baseline_constant`;
force metric `structure_balanced`; `geometry_dtype: float64` with
`network_dtype: float32`, recorded separately; normalization
`original_compute_feature_normalization_on_train_only`; AdamW, lr 1e-3, weight
decay 1e-7, grad clip 1.0; ReduceLROnPlateau factor 0.5, scheduler patience 10;
early stopping patience 30 at min_delta 1e-5, selection on val; hidden dims
[256, 128]; energy weight 1e4, force weight 1e3; seed 20260811; residual
architecture [392, 256, 128, 7] with centering "per atom across expert channels";
descriptor `species=(Si,O), r_cut=5.0, n_max=6, l_max=4`, 392 input features
including the [isSi, isO] one-hot; torch 2.8.0+cu128, CUDA 12.8, Python 3.9.19,
numpy 1.26.4.

Forces differentiate the complete energy including the gating weights'
coordinate dependence: `pilot/c1b/training_info.json` records
`routing_coordinate_gradient_enabled: true` with
`routing_parameters_frozen: true`, and the finite-difference check in item 8
would fail were that gradient omitted.

**Pending:** whether these match the manuscript's stated settings, and which
settings the manuscript asserts that are supported only by script defaults.

## 4. Static energetics — VERIFIED

The gamma66 interface evaluation, 66 interface cases. All values are
interface energies in J/m^2 against MATPES r2SCAN references, generated
from the committed metrics rather than transcribed.

| split | model | status | N | MAE | RMSE | bias | Pearson r | slope |
|---|---|---|---|---|---|---|---|---|
| all | full Local-MoE | current | 66 | 0.005521 | 0.007640 | 0.005207 | 0.925122 | 0.946167 |
| all | common part only | current | 66 | 0.005139 | 0.007641 | 0.004831 | 0.918228 | 0.957295 |
| all | common, pre-repair descriptor | superseded | 66 | 0.010101 | 0.013150 | 0.008516 | 0.782802 | 0.868337 |
| train | full Local-MoE | current | 26 | 0.004771 | 0.006738 | 0.004530 | 0.938164 | 0.978011 |
| train | common part only | current | 26 | 0.004386 | 0.006149 | 0.004197 | 0.948497 | 0.971825 |
| train | common, pre-repair descriptor | superseded | 26 | 0.009107 | 0.013042 | 0.006435 | 0.685493 | 0.731904 |
| test | full Local-MoE | current | 40 | 0.006008 | 0.008173 | 0.005648 | 0.917513 | 0.922879 |
| test | common part only | current | 40 | 0.005628 | 0.008470 | 0.005243 | 0.900774 | 0.944542 |
| test | common, pre-repair descriptor | superseded | 40 | 0.010747 | 0.013221 | 0.009868 | 0.840271 | 0.934847 |

Effect of the residual, as percentage change from common-only to full:

| split | metric | common only | full | change % | verdict |
|---|---|---|---|---|---|
| all | MAE | 0.005139 | 0.005521 | +7.4313 | worse |
| all | RMSE | 0.007641 | 0.007640 | -0.0083 | better |
| all | bias | 0.004831 | 0.005207 | +7.7941 | worse |
| all | pearson_r | 0.918228 | 0.925122 | +0.7508 | better |
| train | MAE | 0.004386 | 0.004771 | +8.7799 | worse |
| train | RMSE | 0.006149 | 0.006738 | +9.5713 | worse |
| train | bias | 0.004197 | 0.004530 | +7.9348 | worse |
| train | pearson_r | 0.948497 | 0.938164 | -1.0895 | worse |
| test | MAE | 0.005628 | 0.006008 | +6.7482 | worse |
| test | RMSE | 0.008470 | 0.008173 | -3.5084 | better |
| test | bias | 0.005243 | 0.005648 | +7.7209 | worse |
| test | pearson_r | 0.900774 | 0.917513 | +1.8583 | better |

The pre-repair common-descriptor row is retained and marked
`superseded` in the source table, so the improvement from the repair is
visible rather than quietly dropped.

The residual was trained on per-configuration energies and forces, not
on interface energies, and the per-split verdicts above are the evidence
for how little interface-specific signal it carries.

## 5. Routing diagnostics — CORRECTED

Regenerated on the selected model; the 18 October figures came from pre-repair
trajectories and checkpoints. See `ROUTING_DIAGNOSTICS.md`.

Three corrections matter for the text. The three-region figure displays **22**
structures — eleven interface systems in two configurations each — and no code
path in `routing_from_md.py` produces a subset of 16. Both figures order by
source group, split group and record id, **not** by any descriptor or complexity
quantity. And PSO2-BKS is nonzero in all 52 structures; an earlier version of
this audit rounded it to 0.000 and wrongly called the channel unused.

The interface signature is ML-BKS enrichment, median **15.5x** over phase-1 bulk
per structure, range 5.2x to 124x, 0 of 22 exceptions. That is a median of
per-structure ratios, not the ratio of means (about 12x).

**Pending:** correcting the actual captions, which requires the manuscript.

## 6. Dynamics — CORRECTED

52 configurations at 10 ps: all attempted, all completed, all criterion-
satisfying, median maximum excursion 0.003126 meV/atom, cost scaling N^0.663 over
162-4608 atoms.

Ten configurations at 500 ps, **50 times** the protocol duration: all reach
500.0 ps, nine of ten satisfy the 0.1 meV/atom criterion, median maximum
excursion 0.005680 meV/atom.

`own_0146` (CHA__EMT, 2592 atoms) does not, at 0.519446 meV/atom. Its behaviour,
without attributed mechanism: the criterion is first exceeded at **127.75 ps**,
**1490 of 2001** samples lie above it, and **61%** of the final drift had
accumulated before the largest sampled increment of +0.209973 between 273.50 and
273.75 ps. The next four increments are +0.110351, +0.055894, +0.037599 and
+0.037402. It is therefore progressive growth containing one larger increment,
not a single event against a conserved background. Temperature rises 300 to about
355 K; arithmetically that implies potential energy fell, but **no structural
analysis was performed, no rearrangement is established, and no timestep
mechanism is demonstrated**. Separating timestep truncation from physical
enthalpy release would require a smaller-timestep rerun, which has not been done.

Drift slopes are tabulated descriptively with OLS standard errors.
**No significance claim is made:** the residuals are strongly autocorrelated, so
OLS errors understate uncertainty, and six of ten trajectories have
abs(slope)/SE above 4 — an earlier version of this audit wrongly claimed none of
the other nine had a resolvable slope.

## 7. Computational cost — CORRECTED

Remeasured on an A100-80GB with the selected model; the archived figure used
pre-repair artifacts and a first replacement run landed on a 40 GB card and was
discarded. Both internal gates passed at about 1e-16.

Three statistic families, reported separately because they were being conflated:
median-of-medians timings (112.993577, 90.032201, 12.099392, 13.046559 ms), peak
memory as a **maximum** (30.803709, 15.415611, 0.554940, 0.549447 GiB), and
paired per-configuration speedups over n = 105 — median **8.676736x** against
MACE float64 and **6.929799x** against float32, with median paired memory
reductions **98.081212%** and **96.175662%**.

Any quoted speedup must name baseline and model. The image-enumerating subset is
**14 of 105** configurations at median 2.912119x against 9.173889x for the other
91; the family-merged grouping in the figure gives ten of 63 markers at about
3.07x against 9.39x, a different population.

The repaired descriptor is about **14% slower** than the pre-repair one, so
speedups quoted from the archived figure are optimistic. The measured time
exponent of 0.08 for N >= 1000 is not an asymptotic complexity.

## 8. Invariance, conservativity, stress — VERIFIED with coverage stated

The October audit condemned both pre-repair descriptors at l >= 2 (per-centre
relative L2 1.250; power spectrum 1.933, 33.19, 33.47 at l = 2, 3, 4), and the
corrected module was written after those verdicts and had never been tested. It
now passes on all three routes: energy invariant to 2.4e-12 meV/atom, forces
covariant to 4.0e-13 eV/A, translation leaving abs(sum F) at 1e-15, and accuracy
identical to four decimal places across identity, six named rotations and 48
random orientations. Invariance is a **bound of about 4e-13 eV/A**, roughly
eighty times the two-evaluation noise floor, not bitwise identity.

Coverage, precisely: 9 structures and 14 rotations per route; stress covariance
on the **calculator routes only**, since the trainer route returns none; the
conservativity check samples **six random atom-component pairs** for forces and
**three reciprocal pairs between two atoms** for the Hessian, on the 9-atom
built-in quartz, and the h to h/2 agreement is **consistent with second-order
truncation over the two steps tested** rather than establishing an order. The
10 ps preflight finite-difference check covers two configurations, nine force
components each, plus all six stress components.

`tests/test_torchsoap_neighbour_list.py` still targets the pre-repair module and
does not cover the descriptor the paper uses; it accepts `--module`, so the
rerun is available but has not been done.

## 9. Deliverables — this commit

`audit/master_metrics.csv`, `audit/claim_to_evidence.csv`, `REPRODUCTION.md`,
the four package reports, `CORRECTIONS_ROUND2.md`, and per-package
`metrics/MANIFEST_SHA256.txt`. The release archive is pinned to this commit.

## Outstanding — REQUIRES AUTHOR INPUT

The manuscript, SI and bibliography have not been examined. Until they are:

1. No comparison of executed settings against the manuscript's stated settings.
2. No caption corrections, including the three identified above (the displayed
   subset of 22, the ordering, and the optimistic speedups).
3. No reference, label or placeholder audit.
4. The origin of the "16 structures" figure count is unexplained. The code and
   data give 22, and no code path produces 16.
