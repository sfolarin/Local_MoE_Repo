# MD for the selected Local-MoE — artifact check and CPU pilot

Common epoch 116, router epoch 110, residual epoch 13, repaired descriptor. Run
2026-11-01 on CPU; the GPU allocation is exhausted and none was used.

Two things are reported here and they are **not** the same thing: existing MD results
that were checked and found not to match these artifacts, and one new pilot that does.

---

## 1. Existing MD results do not match, and are not reused

Three directories of prior MD exist: 52 completed cases in `md_holdout40_v1`, 10 in
`md_long500ps_v1`, 2 in `md_smoke`. All 52 summaries record **one** artifact set, and
not one of its four members is a selected artifact:

| role | path recorded in the run | its hash today | selected artifact |
|---|---|---|---|
| common | `runs/common_holdout40_ew10000_v1/best_checkpoint.pt` | `fde94fa621c1a06d450e7c4a` | `5b17849752a535739b16cfc8` |
| routing | `runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt` | `625915d496ca4127f81e24c1` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt` | `99d0cfaec332de044c5fa96e` | `c7ad2a89c570549309ad8906` |
| descriptor | `scripts_nl/torchsoap_autograd_module_periodic_regnorm_nl.py` | `30acc0838fb689cb7274e526` | `0afd9a1ba7b9a24f8bf8151a` |

The common and routing entries are exactly the two checkpoints that the interface
evaluation had to override, being that script's defaults. Every artifact has mtime
2026-10-18, a week before the descriptor repair. The prior runs are the **pre-repair
model throughout**.

Two further points about their provenance, independent of the mismatch. The summaries
record artifact **paths but no hashes**, so even a path match would not establish what
the file contained at run time. And `trainer_script` points into a different repository
(`local-moe-silica-interfaces`), not the branch checkout the interface evaluation used,
though the two copies of `train_c1b_holdout40.py` happen to be byte-identical
(`333dd835069596d81d43b264`).

Conclusion: no matching trajectories exist. Nothing below reuses them. The one thing
taken from that directory is `md_manifest.csv` — a list of structures and atom counts,
a property of the dataset rather than of any model.

## 2. Artifacts and code

Hashes printed by the jobs themselves, not collected afterwards:

| role | file | sha256 (24) |
|---|---|---|
| common network (epoch 116) | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate (epoch 110) | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual (epoch 13) | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |
| C1b model class | `training/train_c1b_holdout40.py` | `333dd835069596d81d43b264` |

**Calculator**: `src/local_moe/local_moe_ase.py`, class `LocalMoECalculator` — the one
the runner instantiates (line 131). The repository also contains
`moe_persistent_calculator.py`; it is not used here.

**Runner**: `md/run_local_moe_md.py`, unmodified.

**Numerical precision**: `--dtype float64` throughout, confirmed in every log line and
in the summary. The descriptor ran in neighbour-list mode — `backend=torch, r_cut
5.000000 Å, N=162, edges 3944`.

The repaired descriptor is the neighbour-list module plus the angular-weight
correction: 16 changed hunks against
`torchsoap_autograd_module_periodic_regnorm_nl.py`, identical public function surface
except that the repaired file adds `_angular_weight`. It is therefore MD-capable, and
no O(N²) fallback is involved.

Every artifact path was passed explicitly on the command line. This matters for the
same reason it did in the interface evaluation: the runner's defaults point elsewhere,
including `--trainer-script`, whose default resolves into the other repository.

## 3. Preflight (CPU), job 60662850

Both checks run on the selected structure before any dynamics.

**Route parity** — this calculator against `trainer.predict_energy_forces` on the same
record, descriptor `TorchSOAPConfig(species=("Si","O"), r_cut=5.0, n_max=6, l_max=4)`:

| quantity | measured | acceptance |
|---|---|---|
| energy difference | `+0.00000 meV/atom` (below printed resolution) | < 0.1 meV/atom |
| forces, max abs difference | 3.430e-07 eV/Å | < 2.0e-03 eV/Å |
| forces, rms difference | 7.686e-08 eV/Å | — |

PASS. Note what this check is: the trainer's networks are float32 and the calculator
runs float64, so agreement is expected only at float32 rounding, and the thresholds are
set accordingly. It is a weaker statement than the interface evaluation's route check,
which was exactly 0.00e+00 because both sides ran at the same precision.

**Finite differences** — central differences of the model's own energy. Step
**h = 1.0e-4** for both the Cartesian displacement (Å) and the strain (dimensionless).
Forces are checked on three atoms, indices 0, N/2, N−1, all three components each;
stress on all six components:

| quantity | worst \|FD − analytic\| | acceptance | margin |
|---|---|---|---|
| forces | 3.038e-07 eV/Å | 1e-5 eV/Å | 33× |
| stress | 2.562e-08 eV/Å³ = 4.105e-06 GPa | 2e-6 eV/Å³ | 78× |

PASS. The error sits entirely in the three diagonal stress components (1.86e-08,
1.86e-08, 2.56e-08 eV/Å³); the off-diagonals agree to 1e-14, which is float64 noise.
Stress is therefore trustworthy, which is what makes the NPT stage meaningful.

## 4. Structure selection

Chosen by atom count from `md_manifest.csv` **before** simulating, as the smallest
test-split configuration available:

| | |
|---|---|
| record | `external_0219` |
| framework pair | LOV__VSV |
| atoms | 162 (the smallest of 52; next are 168, 216, 216) |
| split | test |

All 52 manifest entries are test-split, and manifest and run atom counts agree on every
one of them.

## 5. Pilot, job 60663099

Protocol as documented, every parameter left at its default:

1. fixed-cell relaxation, FIRE to fmax 0.05 then BFGS to fmax 0.01
2. NPT, ASE `NPTBerendsen`, isotropic — 5 ps, dt 0.5 fs, 300 K, 1 bar, taut 100 fs,
   taup 1000 fs, compressibility 4.57e-5 /bar
3. exact velocity rescale to 300 K
4. NVE, ASE `VelocityVerlet` — 10 ps, dt 0.5 fs, logged every 20 steps

No thermostat, barostat or velocity rescaling acts during NVE.

**Relaxation converged**: FIRE 6 steps then BFGS 20 steps, Fmax 0.269 → 0.0098 eV/Å
against the 0.01 target. (The optimiser logs hold 7 and 21 rows; ASE writes an initial
row before the first step.)

**NPT**: initial pressure +2.1905 GPa, mean +0.0132, final −0.0123 GPa; mean temperature
292.71 K; volume +2.449%, 3156.06 → 3233.36 Å³. Velocity rescale
292.275751 K → 300.000000 K, factor 1.013127806.

**No failures.** Status `done`; all expected outputs written.

## 6. Energy conservation

Baseline is the **initial NVE sample**, which is the only zero that makes an excursion
meaningful — the NPT stage ends elsewhere:

```
E_total(t=0) = -1587.383158872     eV
             =    -9.798661474518  eV/atom
```

1001 samples over 10.00 ps:

| quantity | value | units |
|---|---|---|
| maximum absolute excursion | **0.005262** | meV/atom |
| final change, E(10 ps) − E₀ | **+0.002972** | meV/atom |
| peak-to-peak | 0.009384 | meV/atom |
| fitted linear drift (least squares vs t) | **+1.5298e-05** | meV/atom/ps |
| temperature, mean / min / max | 303.23 / 262.25 / 341.08 | K |

**Energy conserving**: the maximum excursion is a factor of 19 inside the 0.1 meV/atom
criterion.

Every one of these was recomputed from the raw `nve_10ps_log.csv` by
`analyze_md_pilot_ep13.py` and compared against the runner's own
`final_protocol_summary.json`; worst relative difference **9.2e-10** across all seven,
and the conservation verdict agrees. The analysis aborts without writing anything if
they disagree, so these numbers are reproducible from the trajectory rather than
asserted by the code that produced it.

## 7. Compute

| | |
|---|---|
| job | `md_pilot_ep13`, premium QOS, CPU, 1 node, 64 threads |
| elapsed | **18 m 21 s** (`sacct`, COMPLETED) |
| NPT | 538.10 s, 18.58 steps/s |
| NVE | 548.74 s, 36.45 steps/s |
| GPU-hours | **0.000** |

NVE runs at twice the NPT rate because NPT evaluates the stress tensor every step.

### Extending the cohort

Measured: **6.709 s per atom per 15 ps** at N = 162 on 64 threads.

| | |
|---|---|
| full 52-case cohort (75,606 atoms), scaled linearly in N, run serially | **≈ 141 h** |
| largest single case, N = 4608 | ≈ 8.6 h |
| 500 ps NVE at N = 162 (10⁶ steps at the measured rate) | ≈ 7.6 h |

**These are linear extrapolations from one measurement, not established bounds.** The
only timing evidence is a single 162-atom case; everything above is that rate multiplied
by atom count. Linear scaling is plausible for a neighbour-list descriptor, but the true
scaling has not been measured and could fall on either side of it. A 162-atom cell with
3,944 neighbour pairs underuses 64 threads, which would make larger cases cheaper per
atom; against that, larger cells bring memory-bandwidth pressure, cache misses and
growing neighbour-list overhead, which push the other way. Which effect dominates at
N ≈ 4600 is unknown here.

One timed case at N ≈ 1000 would replace the extrapolation with a measured scaling, and
is the cheapest thing that would make a cohort figure defensible. Note also that cases
parallelise, so wall time is the serial figure divided by the number of concurrent jobs.

## 8. Outputs

In `MD_Pilot_EP13_Deliverable_v1/`:

| file | contents |
|---|---|
| `md_pilot_ep13_v1_nve_stats.csv` | every quantity in §6 plus baseline, atom count, rates and wall times |
| `md_pilot_ep13_v1_cost_projection.csv` | §7 cohort figures as data |
| `case_external_0219/` | the full case: input, relaxed and final structures, NPT and NVE logs and frames, relaxation logs, `final_protocol_summary.json` |
| `scripts/` | runner, calculator, both submit scripts, the analysis script |
| `logs/` | preflight, pilot and analysis logs, path-scrubbed |
| `MANIFEST_SHA256.txt` | sha256 of every packaged file and of the five artifacts |

The trajectory frames are `.extxyz`, which the repository's `.gitignore` excludes, so
they live in the archive rather than in git. `regions.npz` is excluded for the same
reason.

In the branch the same content is committed under
`tools/descriptor_repair/dynamics_pilot/` — `metrics/`, `logs/`, `scripts/` and
`case_external_0219/`, with the report at `tools/descriptor_repair/MD_PILOT.md`. The
directory is **not** called `md_pilot`: `.gitignore` excludes `md_*/`, which would have
dropped every file under it. The raw per-sample data needed to re-verify §6 and to draw
energy against time is `dynamics_pilot/case_external_0219/nve_10ps_log.csv`, 1001 rows
carrying `time_ps`, `potential_energy_eV`, `kinetic_energy_eV`, `total_energy_eV` and
`total_energy_per_atom_eV`.

## 9. Reproduction

Preflight and pilot, as submitted, with the allocation name replaced:

```bash
A=( --het "<het>" --repo "<het>/moe-interface-potential"
    --trainer-script     "<het>/moe-interface-potential/training/train_c1b_holdout40.py"
    --stage-root         "<proj>/data/expanded_holdout40_v1_c1b_stage"
    --common-checkpoint  "<repair>/pilot/common/best_checkpoint.pt"
    --routing-checkpoint "<repair>/pilot/router/best_torchsoap_routing_gate.pt"
    --c1b-checkpoint     "<repair>/pilot/c1b/best_c1b_residual_model.pt"
    --torchsoap-module   "<repair>/torchsoap_corrected_rotinv_v1.py"
    --device cpu --dtype float64 )

# preflight
python3 <het>/moe-interface-potential/md/run_local_moe_md.py "${A[@]}" --parity external_0219
python3 <het>/moe-interface-potential/md/run_local_moe_md.py "${A[@]}" --check  external_0219

# pilot
python3 <het>/moe-interface-potential/md/run_local_moe_md.py "${A[@]}" \
  --records external_0219 --out-root "<proj>/md_ep13_pilot_v1"

# verification and statistics
python3 analyze_md_pilot_ep13.py \
  --case     "<proj>/md_ep13_pilot_v1/case_external_0219" \
  --out-dir  MD_Pilot_EP13_Deliverable_v1 \
  --manifest "<proj>/md_holdout40_v1/md_manifest.csv"
```

SLURM: `--qos=premium --account=<alloc> --constraint=cpu --nodes=1 --ntasks=1
--cpus-per-task=64`, with `OMP_NUM_THREADS=MKL_NUM_THREADS=64`.

`--out-root` is load-bearing. The runner defaults to `md_holdout40_v1`, which holds the
52 pre-repair cases, and it **skips** any case whose `nve_final_structure.extxyz`
already exists — so the default would either mix two models in one directory or
silently do nothing.

## 10. Limitations

- **n = 1.** One 162-atom cell for 10 ps. Every check passes with large margins, but
  this does not establish that the epoch-13 model is stable in MD. The superseded
  cohort used 52 cases to support that claim; this model has one.
- The pilot is the **smallest** structure in the set. Stability at 162 atoms says
  little about 4,608.
- The cohort figures are linear extrapolations from one timing measurement at the
  smallest size. The scaling itself is unmeasured, so they are neither an upper nor a
  lower bound.
- Route parity is agreement at float32 rounding, not exactness, because the trainer's
  networks are float32.
- Finite differences sample three atoms of 162 for forces. All six stress components
  are checked.
- The 500 ps suite was not launched.
