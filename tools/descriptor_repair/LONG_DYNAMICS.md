# Long NVE dynamics: ten interfaces to 500 ps

Ten holdout-40 test interfaces integrated for 500 ps in the NVE ensemble with
the selected Local-MoE, to test energy conservation two orders of magnitude
beyond the 10 ps protocol cohort.

## Model

| role | path | sha256 (first 24) |
|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |

Every case carries a `provenance_hashes.json` recording these four hashes plus
the executed runner and calculator. The runner itself records artifact *paths*
but not hashes, which is why the sidecars exist: without them a trajectory
cannot be tied to a particular model after the fact.

## Protocol

Each run starts from its own NPT endpoint in the 10 ps cohort
(`<proj>/md_ep13_v1/case_<id>/npt_final_structure.extxyz`) and rescales exactly
to 300 K, then integrates 1,000,000 steps at dt = 0.5 fs in float64, logging
every 500 steps (0.25 ps) and writing frames every 10,000 steps (5 ps). All ten
reached 500.0 ps with 2001 log samples.

These are **not continuations** of the 10 ps NVE trajectories. Both start from
the same NPT endpoint and both rescale to 300 K, but the long runner reads that
state back from extxyz at finite precision, so the trajectories diverge
chaotically. Per-case 10 ps and 500 ps numbers are therefore not expected to
agree, and do not.

Five of the ten resumed from a checkpoint partway through. Restart continuity
was checked independently of the runner: logged time is strictly increasing in
all ten, no timestamp is duplicated, and no sampling gap exceeds the nominal
0.25 ps. The resumed runs preserve their original energy baseline.

## Result

Every statistic below was recomputed from the raw 2001-row logs and agrees with
the values the runner recorded to a relative 1e-6.

**Nine of ten satisfy the 0.1 meV/atom maximum-excursion criterion**, with a
median maximum excursion of 0.005680 meV/atom. The runner's own
`energy_conserving` flag independently reports 9/10.

The exception is `own_0146` (CHA__EMT, 2592 atoms): maximum excursion
**0.519446 meV/atom**, final drift +0.492553, peak-to-peak 0.522377, fitted
slope +1.348764e-03 meV/atom/ps against a standard error of 1.0555e-05 — about
128 sigma, a genuinely resolved secular drift. For comparison the other nine
have slopes of 1.9e-07 to 1.6e-06 against standard errors of 5.7e-08 to
2.9e-07, so none of them has a statistically resolvable drift at all.

The drift is **event-driven, not accumulating error**. It rises gradually to
0.295 meV/atom by 250 ps, then jumps to 0.512 within one 25 ps window — the
steepest 0.25 ps segment is +0.83989 meV/atom/ps at t = 273.50 ps — and then
plateaus flat at about 0.515 for the remaining 225 ps, moving only -0.0007 over
the last 10 ps. Its RMS fluctuation about the fitted trend is 0.0681 meV/atom
against 0.0004-0.0019 for the other nine, which is that single step showing up
as residual. Temperature rises monotonically 300 -> 355 K across the run; that
is roughly 7 meV/atom of kinetic energy against a total-energy change of 0.49,
so potential energy fell about 6.5 meV/atom. The reading consistent with those
numbers is an exothermic structural rearrangement that the 0.5 fs timestep could
not integrate tightly through, after which conservation is restored.

On this same trajectory the first 10 ps are clean: its
`first_10ps_max_abs_drift_meV_per_atom` is **0.002931**. The failure is visible
only on the long run, which is what a 500 ps test is for. It crosses the
criterion at roughly 150 ps.

`own_0146` is reported, not excluded. It was one of ten structures selected
before any of them ran (`metrics/long_manifest.csv`), and the criterion
detecting a real event is a more informative result than ten quiet trajectories.

## Independent consistency checks

Across all ten: maximum |E_total - E_pot - E_kin| = 1.75e-11 eV, maximum
per-atom energy reconstruction error 5.33e-15 eV, and cell volume constant to
machine precision (NVE, fixed cell).

## Cost

336.5 h wall in total, dominated by `external_0110` at 80.9 h for 4608 atoms.
At 16 threads that is 5,384 core-hours, or 42.1 node-hours on 128 cores.

## Limitations

- Ten structures, 216-4608 atoms. Not a survey.
- The `own_0146` interpretation rests on the energy and temperature traces. No
  structural analysis of the rearrangement was performed; the frames are in the
  run tree, not in this package.
- A smaller timestep was not tried. Rerunning `own_0146` at dt = 0.25 fs would
  separate timestep truncation during the event from genuine enthalpy release:
  a roughly fourfold reduction would indicate the former.
- `recorded_split_counts` in the figure's validation JSON reads
  `{"unknown": 10}` because the long-run summaries carry no `split` field. All
  ten are test-split records, established from the cohort manifest, not from the
  long-run metadata.
- `shared_protocol` in that JSON is empty for a mechanical reason: the long
  runner writes run settings at the top level rather than under `protocol`, so
  per-case fields such as walltime prevent any key being common. The protocol
  facts are verified per case in `metrics/long500ps_verification_ep13.csv`.

## Files

`figures/` the three-panel 500 ps figure, its binned envelope, per-case summary,
caption and validation JSON. `metrics/` the recomputed verification table, the
pre-registered manifest of ten, and the derived plotting manifest that adds
`split_group` from the cohort manifest (the original is left untouched).
`case_*/` the raw NVE log, the runner's summary and the provenance sidecar for
each structure. `scripts/` the submit script. Hashes in
`metrics/MANIFEST_SHA256.txt`.

## Reproduction

    python3 <repo>/md/plot_local_moe_nve_drift.py \
        --md-root <proj>/md_ep13_long500ps_v1 \
        --manifest <proj>/md_ep13_long500ps_v1/long_manifest_plot.csv \
        --out-dir <out> --prefix LocalMoE_ep13_long500ps \
        --log-name nve_500ps_log.csv --summary-name long_nve_summary.json \
        --duration-ps 500 --limit 0.1 --paper --dpi 600
