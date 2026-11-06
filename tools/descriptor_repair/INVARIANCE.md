# Rotational invariance, translation and conservativity

The measurement the descriptor repair rests on, and which had never been made.

## Why it was missing

`C0_descriptor_verdicts.txt` (2026-10-25 18:37 Pacific, in `stats/`) condemned
**both** pre-repair descriptors — the original dense module (`186453838f89...`)
and the neighbour-list module (`30acc0838fb6...`) — at a 1e-9 relative tolerance
under random SO(3) rotations:

| quantity | verdict |
|---|---|
| per-centre 390-vector relative L2 | 1.250 FAIL |
| global 780-vector relative L2 | 0.977 FAIL |
| power spectrum l=0 | 3.5e-12 PASS |
| power spectrum l=1 | 6.1e-11 PASS |
| power spectrum l=2 | 1.933 FAIL |
| power spectrum l=3 | 33.19 FAIL |
| power spectrum l=4 | 33.47 FAIL |

A relative L2 of order unity means the descriptor is essentially uncorrelated
with itself under rotation. float32 and float64 verdicts are identical, so the
defect was in the formulation at l >= 2, not precision.
`A2_compare_module_copies.txt` shows the two modules share identical `config`,
`cutoff`, `radial`, `angular` and `contract` elements, so both computed the same
values. That module appears in the `src_snapshot` of every pre-repair feature
dataset and training run.

`torchsoap_corrected_rotinv_v1.py` was written 2026-10-26 01:05 Pacific, six and
a half hours **after** those verdicts. It is the audit's output and had never
itself been audited.

## What was run

The same harness, unchanged, against the selected model:

| role | path | sha256 (first 24) |
|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |

Nine structures: six holdout-40 test records of 60, 378, 648, 810, 972 and 1404
atoms, plus built-in alpha-quartz (9 atoms, hexagonal), a rattled and sheared
beta-cristobalite (24 atoms, triclinic) and a random SiO2-like cell (72 atoms,
triclinic). Three evaluation routes: the trainer's `predict_energy_forces` with
float32 networks, and the MD calculator in float32 and float64. Fourteen
rotations per structure (six named, eight random SO(3)), a rigid translation, and
a same-input noise floor. A100-80GB.

Three of the six test records have gamma = 79.1 degrees and two built-ins are
triclinic, so the test is not confined to the canonical frame in which the
dataset is stored. `D_frame_census.txt` shows all 512 records have
lower-triangular cells with mean abs(a_x)/abs(a) = 1.0000, i.e. **a** along x; a
non-invariant descriptor can look consistent in that frame alone, so invariance
had to be shown on frames the convention does not cover.

## Result

| route | noise dE | noise dF | max abs dE rot | max abs dF rot | max rel dF | max dSigma GPa | trans dF | verdict |
|---|---|---|---|---|---|---|---|---|
| trainer32 | 0 | 7.11e-15 | 0.0000e+00 | 3.5016e-13 | 4.514e-13 | **not evaluated** | 2.18e-13 | PASS |
| calc32 | 0 | 5.33e-15 | 0.0000e+00 | 3.5022e-13 | 4.513e-13 | 7.782e-14 | 2.19e-13 | PASS |
| calc64 | 0 | 5.33e-15 | 2.3685e-12 | 4.0212e-13 | 5.389e-13 | 6.893e-14 | 3.03e-13 | PASS |

**Stress covariance is evaluated on the calculator routes only.** The trainer
route returns no stress: its per-rotation records contain
`dE_meV_atom, dF_max, dF_rms, dF_rel, F_rms, sumF_rot` and no stress field,
while the calculator routes add `dS_GPa`. An earlier version of this table
printed `0.000e+00` for the trainer route, which rendered a missing measurement
as a perfect score.

Energies are invariant to within 2.4e-12 meV/atom worst case and exactly zero in
most cases; forces are covariant to within 4.0e-13 eV/A against force magnitudes
of 0.1-7 eV/A; translation leaves energy unchanged and abs(sum F) at 1e-15.

Stated as a bound: invariance holds **to about 4e-13 eV/A**, roughly eighty times
the two-evaluation noise floor of 5e-15. It is float64 accumulation over the
structure, not bitwise identity.

## Accuracy is orientation-independent

On the six real test records, against stored MatPES r2SCAN labels with reference
forces rotated consistently:

| orientation | E RMSE meV/atom | E MAE meV/atom | F SB-RMSE eV/A | F pooled RMSE |
|---|---|---|---|---|
| identity | 1.7539 | 1.1373 | 0.01291 | 0.00721 |
| each of Rz(30), Rz(90), Rx(90), Ry(90), Rz(180), Rx(180) | 1.7539 | 1.1373 | 0.01291 | 0.00721 |
| 48 pooled random SO(3) | 1.7539 | 1.1373 | 0.01291 | 0.00721 |

Identical to four decimal places, in all three routes.

## Conservativity, with its sampling stated

`fd_and_hessian(..., h=1e-4, n_fd=6)` on the float64 route, built-in
alpha-quartz:

| quantity | unrotated | random rotation |
|---|---|---|
| max abs(F_autograd - F_fd), h = 1e-4 A | 8.661e-08 eV/A | 1.628e-07 eV/A |
| the same at h/2 | 2.159e-08 | 4.064e-08 |
| ratio | 4.01 | 4.005 |
| Hessian asymmetry max abs(H_ab - H_ba) | 1.610e-07 | 4.641e-08 |
| Hessian scale | 2.324 eV/A^2 | 2.771 eV/A^2 |

**What was sampled.** The force check draws **six random (atom, component)
pairs** — not all components of selected atoms. The Hessian check evaluates
**three reciprocal derivative pairs between atom 0 and its nearest neighbour**,
chosen for non-zero coupling; it is not the full Hessian, and the asymmetry
figure is a bound over those three pairs only.

The factor-of-four reduction between h and h/2 is **consistent with second-order
truncation over the two step sizes tested**. Two points do not establish a
convergence order.

Independently, the 10 ps protocol preflight finite-differenced the model's own
energy on `external_0219` (N=162, float64) and found forces agreeing to
3.038e-07 eV/A (limit 1e-5) and all six stress components to 2.562e-08 eV/A^3 =
4.105e-06 GPa (limit 2e-6 eV/A^3). That check would fail if the gradient omitted
the gating weights' dependence on coordinates, and
`pilot/c1b/training_info.json` records
`"routing_coordinate_gradient_enabled": true` with
`"routing_parameters_frozen": true`.

## Positive control

With the harness's diagnostic subclass substituted, predictions shift by
**+1552 meV/atom** against the stored export and invariance breaks: dE = -54.2
meV/atom under Rz(30), max abs(dF) = 3.45 eV/A, rms dF/rms F = **0.767**, FAIL on
all 27 structure-route combinations. The harness therefore separates invariant
from non-invariant descriptors by about thirteen orders of magnitude.

One point is **not** established: the control's help text describes it as a
"correctly weighted" subclass, yet it fails invariance when layered over a module
that already carries the correction. A double application of the angular
weighting would explain it, but that has not been verified in
`test_torchsoap_rotation.py` and is not asserted here.

## Coverage, stated plainly

- Invariance and translation: 9 structures, 3 routes, 14 rotations each.
- Stress covariance: calculator routes only; the trainer route provides none.
- Conservativity: float64 route, six random atom-component FD pairs and three
  reciprocal Hessian pairs between two atoms, on the 9-atom built-in quartz,
  unrotated and under one random rotation. This is the narrowest claim here.
- The 10 ps preflight FD check covers two configurations (`external_0219`,
  `external_0200`), three atoms and nine force components each, plus all six
  stress components.
- `tests/test_torchsoap_neighbour_list.py` is a regression test for the
  **pre-repair** neighbour-list module and does not cover the descriptor the
  paper uses. It accepts `--module`; that rerun has not been done.

## A documentation hazard

The repaired descriptor is snapshotted inside the pilot run directories under the
**original** filename `torchsoap_autograd_module_periodic_regnorm.py`, carrying
the repaired content. Anyone comparing snapshots by filename will conclude the
pilot used the same descriptor as the pre-repair runs. Only the sha256
distinguishes them: `0afd9a1b...` repaired, `186453838f89...` original.

## Files

`stats/` the two run JSONs (shipped and positive control) and the three
pre-repair audit records establishing the before-state. `scripts/` the harness
and submit script. `logs/` the run log. Hashes in
`metrics/MANIFEST_SHA256.txt`.
