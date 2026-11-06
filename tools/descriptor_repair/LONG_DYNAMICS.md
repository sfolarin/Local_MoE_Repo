# Long NVE dynamics: ten interfaces to 500 ps

Ten holdout-40 test interfaces integrated for 500 ps in the NVE ensemble with
the selected Local-MoE. The duration is **50 times** that of the 10 ps protocol
cohort.

## Model

| role | path | sha256 (first 24) |
|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |

## Protocol

Each run starts from its own NPT endpoint in the 10 ps cohort
(`<proj>/md_ep13_v1/case_<id>/npt_final_structure.extxyz`), rescales exactly to
300 K, then integrates 1,000,000 steps at dt = 0.5 fs in float64, logging every
500 steps (0.25 ps) and writing frames every 10,000 steps (5 ps). All ten
reached 500.0 ps with 2001 log samples.

These are not continuations of the 10 ps NVE trajectories. Both begin from the
same NPT endpoint and rescale to 300 K, but the long runner reads that state
back from extxyz at finite precision, so the two trajectories diverge. Per-case
10 ps and 500 ps statistics are therefore not expected to agree.

Restart continuity was checked independently of the runner: logged time is
strictly increasing in all ten, no timestamp is duplicated, and no sampling gap
exceeds the nominal 0.25 ps.

## Result

Every statistic below was recomputed from the raw 2001-row logs and agrees with
the values the runner recorded to a relative 1e-6.

**Nine of ten satisfy the 0.1 meV/atom maximum-excursion criterion**, median
maximum excursion 0.005680 meV/atom. The runner's own `energy_conserving` flag
independently reports 9/10.

### The exception: own_0146 (CHA__EMT, 2592 atoms)

Maximum excursion **0.519446 meV/atom**, final drift +0.492553, peak-to-peak
0.522377. The observed behaviour, stated without mechanism:

| observation | value |
|---|---|
| first logged sample with abs(drift) > 0.1 meV/atom | t = **127.75 ps** (drift +0.125334) |
| samples above the criterion | **1490 of 2001** |
| drift at t = 250.00 ps | +0.295193 |
| drift at t = 273.50 ps | +0.302297 |
| drift at t = 273.75 ps | +0.512270 |
| largest positive sampled increment | **+0.209973** between 273.50 and 273.75 ps |
| next four largest increments | +0.110351, +0.055894, +0.037599, +0.037402 |
| accumulated before the largest increment | +0.302297, i.e. **61%** of the final value |
| mean temperature / maximum temperature | 336.73 K / 360.83 K |

So the criterion is exceeded from about 128 ps onward, drift grows progressively
across most of the trajectory, and the largest single sampled increment accounts
for roughly two fifths of the final value against the 61% that preceded it. This
is **not** a single discrete event against an otherwise conserved background, and
the earlier description of it as event-driven was wrong.

Temperature rises from 300 K to about 355 K over the run. Arithmetically a 55 K
rise corresponds to roughly 7 meV/atom of kinetic energy against a total-energy
change of 0.49 meV/atom, which implies potential energy fell. That is an
observation about the logged quantities. **No structural analysis was performed,
no rearrangement has been established, and no timestep mechanism has been
demonstrated.** Distinguishing timestep truncation from physical enthalpy release
would require rerunning at a smaller timestep, which has not been done.

On this same trajectory the first 10 ps are clean:
`first_10ps_max_abs_drift_meV_per_atom` is **0.002931**. The failure appears only
on the longer run.

`own_0146` is reported, not excluded. It was one of ten structures fixed before
any of them ran (`metrics/long_manifest.csv`).

### Drift slopes, descriptively

Ordinary least-squares slopes over the full 500 ps with their OLS standard
errors:

| record | N | slope (meV/atom/ps) | OLS SE | abs(slope)/SE | max excursion |
|---|---|---|---|---|---|
| own_0146 | 2592 | +1.348764e-03 | 1.055502e-05 | 127.8 | 0.519446 |
| own_0098 | 2976 | -1.561183e-06 | 7.878409e-08 | 19.8 | 0.002977 |
| external_0110 | 4608 | -3.724574e-07 | 5.668511e-08 | 6.6 | 0.001957 |
| own_0010 | 912 | -9.102865e-07 | 1.602879e-07 | 5.7 | 0.007062 |
| external_0230 | 216 | +1.205596e-06 | 2.211761e-07 | 5.5 | 0.004615 |
| own_0110 | 1872 | +4.936582e-07 | 1.030825e-07 | 4.8 | 0.002675 |
| external_0022 | 234 | -2.864197e-07 | 2.338517e-07 | 1.2 | 0.006562 |
| own_0136 | 384 | +2.360840e-07 | 2.048541e-07 | 1.2 | 0.005900 |
| external_0148 | 216 | +3.171723e-07 | 2.879353e-07 | 1.1 | 0.008194 |
| external_0172 | 336 | -1.856107e-07 | 2.294867e-07 | 0.8 | 0.005461 |

The ratios are reported for completeness and are **not** significance statements.
These residuals are strongly autocorrelated in time, so OLS standard errors
understate the uncertainty and no appropriate correlated-error analysis has been
performed. An earlier version of this report claimed that none of the other nine
trajectories had a statistically resolvable slope; that was false on its own
numbers, since six of ten exceed a ratio of 4.

## Independent consistency checks

Across all ten: maximum abs(E_total - E_pot - E_kin) = 1.75e-11 eV, maximum
per-atom energy reconstruction error 5.33e-15 eV, cell volume constant to machine
precision.

## Cost

336.5 h wall in total, dominated by `external_0110` at 80.9 h for 4608 atoms. At
16 threads that is 5,384 core-hours, or 42.1 node-hours on 128 cores. These are
sums over the 19 executions.

## Provenance: contemporaneous versus post hoc

Two distinct kinds of evidence, which should not be merged.

**Contemporaneous.** Each execution's Slurm log prints the sha256 of all four
artifacts, the runner (`2548a66b02fbcd5c4c6ba3fb`) and the trainer script
(`333dd835069596d81d43b264`) before integration starts. All 19 logs in
`logs/` carry all four artifact hashes. These record what the running job
actually loaded, at the time it loaded it.

**Post hoc.** Each `case_*/provenance_hashes.json` was written after the fact by
a separate script and says so in its own `note` field. It verifies that the
artifacts **as they exist now** match the pinned hashes. On its own it would not
establish what any historical execution loaded.

Taken together the two agree, so the trajectories are tied to the selected model
by contemporaneous record and not by later assertion. Where only the sidecar
existed, the narrower statement would have been the only available one.

Ten records correspond to **19 executions**: four records were resubmitted twice
and one once, the rest ran straight through. Each summary's `slurm_job` field
names only the last execution, so the job history is in the logs, not the
summaries. Nine logs contain `RESUMING`, documenting the restart points.

## Further limitations

- Ten structures, 216-4608 atoms. Not a survey.
- `recorded_split_counts` in the figure's validation JSON reads
  `{"unknown": 10}` because the long-run summaries carry no `split` field. All
  ten are test-split records, established from the cohort manifest.
- `shared_protocol` in that JSON is empty because the long runner writes run
  settings at the top level rather than under `protocol`, so per-case fields
  prevent any key being common. The protocol facts are verified per case in
  `metrics/long500ps_verification_ep13.csv`.

## Files

`figures/` the three-panel figure, binned envelope, per-case summary, caption and
validation JSON. `metrics/` the recomputed verification table, the pre-registered
manifest of ten, and the derived plotting manifest adding `split_group` (the
original is untouched). `case_*/` raw NVE log, runner summary, provenance
sidecar. `logs/` all 19 Slurm execution logs and the ten per-record logs.
`scripts/` the submit script. Hashes in `metrics/MANIFEST_SHA256.txt`.

## Reproduction

    python3 <repo>/md/plot_local_moe_nve_drift.py \
        --md-root <proj>/md_ep13_long500ps_v1 \
        --manifest <proj>/md_ep13_long500ps_v1/long_manifest_plot.csv \
        --out-dir <out> --prefix LocalMoE_ep13_long500ps \
        --log-name nve_500ps_log.csv --summary-name long_nve_summary.json \
        --duration-ps 500 --limit 0.1 --paper --dpi 600
