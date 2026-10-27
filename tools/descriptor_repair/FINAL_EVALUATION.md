# Local-MoE static evaluation: frozen checkpoints, all 512 configurations

Branch `descriptor-repair-pilot`. Paths, user name and allocation scrubbed.

**This is the first test-partition evaluation in this work.** Every prior run excluded
test by construction. The checkpoints were selected on validation only, which is what
makes a one-shot test evaluation legitimate — and it is one-shot: these 105
configurations are no longer a clean holdout for any later decision about this model.

All four artefacts verified against the authorised hashes before the run **and again
after it**, unchanged. No fitting, no tuning, no checkpoint selection. Normalisation
came from the checkpoints and the training contract, never recomputed from evaluated
data. Precision is the training evaluation precision by construction: float64
descriptor, float32 networks, `N·e_ref` in float64.

---

## 1. Headline finding

**On the held-out test partition the full Local-MoE is statistically indistinguishable
from the common network alone, in both energy and forces.**

| test partition, 105 configs | common | Local-MoE | ratio | sign test | bootstrap 95 % CI |
|---|---|---|---|---|---|
| energy RMSE (meV/atom) | 2.4603 | 2.5735 | 1.0460 | 53 / 52, **p = 1** | [0.9059, 1.0928] spans 1 |
| force SB-RMSE (eV/Å) | 0.05765 | 0.05765 | 1.0001 | 51 / 54, p = 0.85 | [0.9964, 1.0043] spans 1 |

The +4.6 % energy figure is **not** a demonstrated regression: 53 better versus 52
worse is a coin flip on 105 samples. No improvement is detected either.

### What "indistinguishable" does and does not mean

A non-significant result is a failure to detect a difference, **not** evidence that the
two models are equivalent. The interval widths say how much that failure constrains:

| test quantity | ratio CI | what it excludes |
|---|---|---|
| energy | [0.9059, 1.0928] | nothing useful — the data are consistent with the MoE being 9.4 % better *or* 9.3 % worse |
| forces | [0.9964, 1.0043] | a true effect larger than ≈0.4 % in either direction |

So the two cases are not alike. The **force** comparison is tight enough to bound any
real effect below about half a percent — still not equivalence, but a quantitative
ceiling. The **energy** comparison is underpowered: at n = 105 it cannot rule out an
effect of either sign up to roughly 9 %, so "no detected difference" carries little
information there. Any claim of equivalence, in either quantity, would require a
pre-specified equivalence margin and a test designed against it; neither exists here.

Training tells the opposite story, and the contrast is the point:

| partition | energy ratio | CI | force ratio | CI |
|---|---|---|---|---|
| train | 0.9110 | [0.8773, 0.9397] **excludes 1** | 0.9795 | [0.9765, 0.9821] **excludes 1** |
| val | 0.9544 | [0.9268, 0.9920] excludes 1 | 0.9972 | [0.9930, 1.0015] spans 1 |
| **test** | 1.0460 | [0.9059, 1.0928] **spans 1** | 1.0001 | [0.9964, 1.0043] **spans 1** |

Train force improvement is overwhelming — 291 of 369 structures, p = 5.2e-30 — and
transfers to nothing measurable. That ordering is the signature of a residual head
fitting the training set without generalising.

Validation needs a caveat: its energy CI excludes 1.0 while its sign test gives
p = 0.26 (23 of 38). At n = 38 the sign test has little power, and an aggregate gain
larger than the per-structure median gain means the improvement is concentrated in a
few high-error structures. Validation also selected the checkpoints, which is exactly
why it cannot carry this claim.

---

## 2. Where a signal does survive, and where it does not

Per-structure direction and aggregate magnitude disagree, consistently. Over the 22
test structures carrying region labels (all `Own_augmented`):

| region | per-structure direction | SB-RMSE change | SB ratio CI |
|---|---|---|---|
| interface | **19 of 22 lower, p = 0.00086** | −0.26 % | [0.9938, 1.0013] spans 1 |
| phase1_bulk | 14 / 8, p = 0.29 | **−0.98 %** | [0.9851, 0.9992] **excludes 1** |
| phase2_bulk | 14 / 8, p = 0.29 | −0.40 % | [0.9894, 1.0035] spans 1 |

Two findings here cut against each other and both should be reported:

1. **At the interface the direction is robust but the magnitude is not.** 19 of 22 at
   p = 0.00086 is not chance. The mean paired change is −0.000538 eV/Å with 95 % CI
   [−0.001007, +0.000035], which includes zero, on a 0.168 eV/Å base. The residual
   almost always helps at the interface, by an amount that rounds away.
2. **The only aggregate force improvement anywhere on held-out data is in
   phase1_bulk**, −0.98 % with a CI excluding 1.0 — *not* at the interface. That is
   the opposite of what the architecture's motivation predicts, and it is the single
   statistically significant aggregate effect in the entire held-out set.

Energy on those same 22 structures shows the same dissociation, more sharply:

| | common | Local-MoE |
|---|---|---|
| RMSE (meV/atom) | 4.47316 | **4.80742** |
| per-structure | — | **better on 18 of 22, p = 0.0043** |
| ratio CI | — | [0.7566, 1.0987] spans 1 |

The MoE improves energy on 18 of 22 structures while the aggregate RMSE rises 7.5 %.
A small number of structures get substantially worse and dominate the squared error.
For MD stability, occasional large residual errors matter more than RMSE does, so this
is worth following up in the dynamics work rather than filing as a wash.

---

## 3. Deliverable 1 — raw per-configuration predictions

`final/moe/per_configuration.csv`, 512 rows:
`record_id, split, n_atoms, E_reference_eV, E_predicted_eV, dE_eV_per_atom`

Per-structure arrays, `final/moe/raw_predictions/<split>/<record_id>.npz`, with
`record_id, n_atoms, E_reference_eV, E_predicted_eV, reference_forces,
predicted_forces`. Counts 369 / 38 / 105.

Common-only predictions in the identical schema at
`pilot/common/raw_predictions/<split>/`, same counts.

Produced by the project's own `export_c1b_parity_predictions_holdout40.py`
(`40f7c795dc46423d`) driven against the **original, unpatched**
`train_c1b_holdout40.py` (`333dd835069596`), whose `load_records()` returns all three
splits. The pilot adapters were deliberately not used: their X1/X3 patches remove test.

---

## 4. Deliverable 2 — per-partition metrics

`final/figs_moe/energy_force_parity_stats.csv`. Energy in meV/atom, forces in eV/Å.
`F_SB` is `sqrt(mean(per-structure MSE))`; `F_pooled` is
`sqrt(average(per-structure MSE, weights=n_atoms))`.

**Local-MoE**

| split | N | atoms | E RMSE | E MAE | E bias | F SB | F pooled |
|---|---|---|---|---|---|---|---|
| train | 369 | 515322 | 1.8312 | 1.0426 | +0.1678 | 0.05630 | 0.06148 |
| val | 38 | 58638 | 4.5035 | 2.5176 | −0.8971 | 0.11323 | 0.13288 |
| test | 105 | 117840 | 2.5735 | 1.2255 | +0.3989 | 0.05765 | 0.07622 |

**Common network only**

| split | N | atoms | E RMSE | E MAE | E bias | F SB | F pooled |
|---|---|---|---|---|---|---|---|
| train | 369 | 515322 | 2.0101 | 1.2284 | −0.6604 | 0.05748 | 0.06260 |
| val | 38 | 58638 | 4.7188 | 2.7677 | −1.6452 | 0.11355 | 0.13311 |
| test | 105 | 117840 | 2.4603 | 1.1954 | −0.3052 | 0.05765 | 0.07630 |

Note the bias sign flip on test: −0.3052 for common, +0.3989 for the MoE. Both small.

---

## 5. Deliverable 3 — common-only vs full on identical configurations

"Identical" is verified, not assumed. For every partition the record-id sets are
equal, atom counts match, and the **reference** energies and forces are bit-identical
between the two export paths. The comparison script additionally gates itself on
`max |moe_residual_total|` over the test split being exactly `0.00e+00 eV` for the
common-only model — confirming the residual head was genuinely removed rather than
merely small.

Metrics-level comparison is §1; region breakdown and paired statistics are §2. Files:
`final/compare/local_moe_vs_common_per_structure.csv` (210 rows = 105 × 2 models) and
`local_moe_vs_common_regions.csv`. Paired statistics in `final/paired_test.json` and
`final/iface_test.json`.

### Statistical method, stated explicitly

**The bootstrap resamples individual configurations, not split groups.** Within each
partition separately, structures are drawn with replacement, n out of n, where n is
that partition's configuration count (369 / 38 / 105; 22 for the region-labelled
subset). Partitions are never pooled and never resampled as blocks, so no interval
here mixes configurations across train, val and test.

The resample is **paired**: one index vector per replicate is applied to both models,
so a structure enters or leaves a replicate for the common network and the Local-MoE
together. This preserves the pairing that makes the comparison sensitive; resampling
the two models independently would discard it and widen the intervals.

- 20 000 replicates, seed 20260811, `numpy.random.default_rng`.
- Percentile intervals (2.5 / 97.5), not bias-corrected or accelerated.
- RMSE-ratio replicates are computed as `sqrt(mean(MSE_moe)) / sqrt(mean(MSE_common))`
  over the resampled structures; atom-weighted variants weight by `3·n_atoms`.
- The sign test is an exact two-sided binomial test on the count of structures each
  model fits better, ties excluded. It is distribution-free and assumes nothing about
  the error distribution; it tests direction only, not magnitude.
- Per-structure quantities use the comparison script's own definitions:
  `dE/atom = (E_pred − E_ref)/n_atoms`, and region force RMSE
  `= sqrt(sumsq_region / (3·n_region))`.

At n = 38 (validation) and n = 22 (region-labelled subset) both the sign test and the
bootstrap have limited power; intervals there are wide and should not be read as
precise.

**Region labels cover only 22 of 105 test structures** (47904 of 117840 atoms), all
from `Own_augmented`. Every region-resolved statement is restricted to that subset.

---

## 6. Deliverable 4 — six-panel parity figure

`final/figs_moe/energy_force_parity.{pdf,png}` and `_stats.csv`, from the project's
`plot_energy_force_parity_holdout40.py` (`cbca3f795b3358a4`) with
`--model-label "Local-MoE"`. No "corrected", no "pilot", as requested. Top row energy
per atom per split, bottom row force components as 2-D density on a shared log colour
scale, annotations regenerated from the new data. Common-only equivalent at
`final/figs_common/`.

**The published layout was reproduced before being reused.** Regenerating the Oct-18
figure from the original predictions with default options produced a stats CSV that
diffs to zero against the archived one, and its built-in consistency check against the
evaluator's `metrics.csv` agreed to 4.44e-16 meV/atom and 5.55e-17 eV/Å. That fixes
the invocation as default options — no `--zoom-row` (so `nrows=2`, six panels), no
`--dedup-csv` (archived stats show `structures_used == records_in_files`). The script
and its `.bak` are byte-identical with the same mtime as the published PDF, so the
current file is what produced it.

One documentation discrepancy, unresolved: `README_v1.md:140` describes
`plot_energy_force_parity.py --paper`, and the holdout40 script has no `--paper` flag.
The README appears to refer to an earlier script.

---

## 7. Deliverable 5 — logs, hashes, budget

### Verification performed

| check | result |
|---|---|
| frozen hashes, before and after the run | all four unchanged |
| metrics recomputed from npz vs the figure's stats CSV | max difference **0.000e+00** |
| export (`predict_energy_forces`) vs pilot (`evaluate_split`), train and val | **0.00e+00** — bit-identical |
| common-side reproduction vs `metrics.csv` | 4 of 4, max relative 1.09e-15 |
| region AW/SB recomputed vs published `regions.csv` | max difference 5.55e-17 |
| common-only residual on test | exactly 0.00e+00 eV |
| configuration counts | 369 / 38 / 105 on both models |

The cross-path check deserves emphasis: the export reaches the energy through
`predict_energy_forces` and the pilot reached it through `evaluate_split`. Two code
paths over identical frozen weights, agreeing to zero. Test has no prior value because
it was never evaluated before — by design.

### Hashes

| artefact | sha256 |
|---|---|
| common checkpoint | `5b17849752a535739b16cfc8e390e9dd53bbe27c8e50b975235142cf16e346e1` |
| router checkpoint | `9e8e0ca6af4976ef8c74c34e1ded97b2aebeabb51c278037ab3ecab1145e1c66` |
| residual checkpoint | `c7ad2a89c570549309ad8906e477a42adad298b51251da5aa33a09c878ade451` |
| corrected descriptor | `0afd9a1ba7b9a24f8bf8151a7a3bdfc409544ed0920355de0598ffae79d51d21` |
| `export_c1b_parity_predictions_holdout40.py` | `40f7c795dc46423d` |
| `plot_energy_force_parity_holdout40.py` | `cbca3f795b3358a4` |
| `eval_test_regions_common_vs_full.py` (original) | `42845f3c68fdc511` |
| `eval_compare_patched.py` | `05926b2949ddd6ce` |
| `final_eval.sh` | `803d9389bf8a6364` |
| `finish_eval.sh` | `96cd49cbaa815450` |
| `patch_compare.py` | `090dc85d6008f8e8` |
| `paired_test.py` | `53af1cfa29bc73b2` |
| `iface_test.py` | `a414068f5cec8d26` |

Descriptor config is identical across the export and the trainer:
`species=("Si","O"), r_cut=5.0, n_max=6, l_max=4`, verified against trainer lines
1283–1287 and the corrected module's dataclass defaults. The export uses
`residual_mean, residual_std = common_mean, common_std`, matching trainer lines
1277–1278.

### Allocated GPU-hours

See `final_evaluation/GPU_ACCOUNTING.md` for the `sacct` transcript. Total
**4.5364 of 8.000 GPU-h; 3.4636 remain.** Both aborted runs are included.

**Correction to the previous report.** The pilot job was reported at 0.6314 GPU-h from
its elapsed time while still running; `sacct` final `ElapsedRaw` is 3606 s, so the
correct figure is 1.0017. The earlier total of 3.6619 was understated.

Three further four-GPU jobs ran on the same day — 60298211 (14391 s), 60298314
(8019 s), 60305203 (14399 s), together about 40.9 GPU-h. They are excluded as
unrelated work; if any belong to this project the cap picture changes entirely.

The statistical analyses ran on a login node at no GPU cost.

---

## 8. A defect found and patched

`eval_test_regions_common_vs_full.py` crashed with
`pandas.errors.EmptyDataError: No columns to parse from file`.

Root cause is in the original, not in our inputs. It guards its two reproduction files
with `Path.is_file()`, which is **True for a zero-byte file**; pandas then raises on
one. It is likewise True for a file lacking a `test` row, where `.iloc[0]` on the empty
selection raises `IndexError`. The pilot produced both states deliberately: the
trainer's `write_csv` writes an empty file when given no rows, and the pilot excluded
test, so `test_region_force_metrics.csv` is 0 bytes and `final_split_metrics.csv` has
train and val rows only.

The original already encodes the right behaviour for absent reproduction data — it
prints a skip message and continues. Four anchored patches extend that to the empty and
test-less cases, each required to match exactly once or nothing is written. The
original is read-only and unchanged at `42845f3c68fdc511`.

Nothing computed changed. `_per_structure.csv` and `_regions.csv` come from the fresh
evaluation, and the common-side reproduction stayed active and passed 4 of 4.

**The absence of prior test numbers is itself the evidence that the test partition
never influenced fitting.** The patched run prints exactly that, for both files.

---

## 9. Caveats

1. **The holdout is spent.** These 105 configurations cannot serve as a clean holdout
   for any further decision about this model.
2. **Region labels cover 22 of 105 test structures**, all `Own_augmented`. Every
   region-resolved claim is restricted to that subset, including the interface result.
3. **Direction and magnitude disagree** at the interface (19/22 at p = 0.00086, but a
   magnitude CI including zero) and for energy on the labelled subset (18/22 at
   p = 0.0043, but RMSE up 7.5 %). Reporting only one of the two would misrepresent the
   result in either direction.
4. **The one significant aggregate force improvement is in phase1_bulk, not at the
   interface** — contrary to the architecture's motivation.
5. **Validation selected the checkpoints**, so its improvements cannot be cited as
   generalisation evidence.
6. The train/val figures match the pilot's own training-time evaluation exactly; they
   are not an independent confirmation of generalisation, only of the export path.
7. **Finite-difference gradient coverage remains 1 of 3** from the pilot
   (`PASS_WITH_UNVERIFIED_GRADIENTS`). Unchanged by this work.
8. **Static evaluation and parity only.** No MD, no interface-energy benchmarks, no
   cost suite.
9. **No result here establishes equivalence.** Every non-significant comparison is a
   failure to detect a difference at the available sample size, not a demonstration
   that none exists. The force intervals bound any real test-set effect below about
   0.4 %; the energy interval bounds nothing useful, admitting effects up to roughly
   9 % in either direction. Establishing equivalence would need a pre-specified margin
   and a test designed against it.

---

## 10. What this implies for the manuscript

Reporting the test partition honestly means the central claim cannot be that the
Mixture-of-Experts residual improves accuracy over the common network: no such
improvement is detectable on held-out data. Three framings survive the data:

- The architecture is **conservative by construction** (forces and stress from autograd
  of one scalar; net force zero to 4e-14 eV/Å) and shows no detectable held-out
  accuracy cost relative to the common network — force differences bounded below about
  0.4 %, energy differences unconstrained at this sample size — while adding per-centre
  routing that is interpretable by region.
- The residual **improves interface forces in 19 of 22 held-out structures**
  (p = 0.00086), a robust direction with a small magnitude whose CI includes zero.
- The train-set gains are large and real but **do not generalise**, which is a result
  about residual-head capacity and dataset size worth stating rather than hiding.

The 20–27 % pilot improvement reported earlier was in training and validation loss. It
does not appear on held-out data. That belongs in the advisor conversation before the
next draft.

The MD and interface-energy evaluations now matter more than they did, because
conservativeness and dynamical stability are the claims the static numbers still
support.
