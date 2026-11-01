# Interface-energy evaluation — 66 families, epoch-13 residual

Interface energies γ = (E_AB − E_A − E_B) / (2·area) for the full Local-MoE and its
common-network part, on all 66 manifest families, evaluated on the frozen
MATPes-relaxed AB / A / B geometries. No retraining, no change of normalization, no
relaxation.

Run 2026-10-31. Output directory `Gamma66_EP13_Deliverable_v1/`.

---

## 1. Artifacts

Every artifact path was passed explicitly on the command line. This matters: the
evaluation script's defaults point at **different** checkpoints, and left to itself it
would have loaded them.

| role | file | sha256 (24) | best epoch |
|---|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` | **116** |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` | **110** |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` | **13** |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` | — |

The hashes match the previous handoff and the epochs match the specification, so the
artifacts are confirmed on two independent identifiers.

Shared hyperparameters recorded in the checkpoints: energy weight 10000.0, force weight
1000.0, seed 20260811, e_ref −9.668443844547909 eV/atom.

**The code that ran**, which is as much a provenance question as which checkpoint:

| role | file (relative to the branch checkout) | sha256 (24) |
|---|---|---|
| evaluation script | `interface_energy/eval_gamma66_fullmoe.py` | `37bf101af94479353bcc8816` |
| C1b model class | `training/train_c1b_holdout40.py` | see `MANIFEST_SHA256.txt` |
| submit script | `run_gamma66_ep13_cpu.sh` (packaged) | see `MANIFEST_SHA256.txt` |

The C1b checkpoint stores weights, not the model class, so the evaluation rebuilds the
class by importing the training script. `--c1b-script` was not passed, so it resolved to
`<repo>/training/train_c1b_holdout40.py` from `--repo`. That file is therefore part of
what produced these numbers, and the packaging script hashes it.

**A second copy of the evaluation script exists** at
`Local_MoE_MATPES_R2SCAN_Expanded_v1/scripts/eval_gamma66_fullmoe.py`. It is **not** the
copy that ran, and the two are not identical: they share the same computation but differ
in path handling, and the other copy's `--c1b-script` help text points at `scripts/`
rather than the correct `training/`. Only the `interface_energy/` copy, hashed above, is
cited by this report.

**Defaults that were overridden.** The script would otherwise have used
`runs/common_holdout40_ew10000_v1/best_checkpoint.pt` (`fde94fa621c1a06d450e7c4a`) and
`runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt`
(`625915d496ca4127f81e24c1`) — neither of which is the specified checkpoint — and
`torchsoap_autograd_module_periodic_regnorm.py` rather than the repaired descriptor. The
checkpoint's own gate would have caught the first two; nothing would have caught the
descriptor, because the C1b checkpoint records no descriptor path.

## 2. Partition

`gamma66_manifest.csv`, 66 families, frozen `new_split`: **26 train, 40 test**. Every
family retained; none dropped for any reason. Geometries are those referenced in the
manifest, read unchanged.

## 3. Gates

All enabled; all passed.

| gate | result |
|---|---|
| route parity (ASE route vs training route, 3 validation structures, N = 180/240/432) | max \|dE\| = **0.00e+00** eV, max \|dF\| = **0.00e+00** eV/Å |
| atom-count closure N_A + N_B = N_AB | passed on all 66, re-verified independently from the geometry files |
| γ_full = γ_common + γ_residual | max \|difference\| = **3.60e-14** J/m² |
| C1b-recorded common/routing checkpoints match those passed | passed |

The route check returning exactly zero (not merely within 1e-9) means the ASE evaluation
path reproduces the training path bit-for-bit.

## 4. Results, J/m²

| split | model | N | MAE | RMSE | bias | Pearson r | slope |
|---|---|---:|---:|---:|---:|---:|---:|
| all | full Local-MoE | 66 | 0.0055 | 0.0076 | +0.0052 | 0.9251 | 0.9462 |
| all | common part only | 66 | 0.0051 | 0.0076 | +0.0048 | 0.9182 | 0.9573 |
| train | full Local-MoE | 26 | 0.0048 | 0.0067 | +0.0045 | 0.9382 | 0.9780 |
| train | common part only | 26 | 0.0044 | 0.0061 | +0.0042 | 0.9485 | 0.9718 |
| **test** | **full Local-MoE** | **40** | **0.0060** | **0.0082** | **+0.0056** | **0.9175** | **0.9229** |
| **test** | **common part only** | **40** | **0.0056** | **0.0085** | **+0.0052** | **0.9008** | **0.9445** |

Computed twice by independent code paths (the evaluation script and the post-processing
script); the two agree to all printed digits.

Errors are positive on 60 of 66 families and 38 of 40 held-out families: the model
systematically **overestimates** interfacial energy, by +0.0056 J/m² on average on the
held-out set.

## 5. What the residual contributes

| split | N | residual γ mean | mean \|·\| | max \|·\| | helped | hurt | unchanged | mean Δ\|err\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 66 | +0.0004 | 0.0011 | 0.0051 | 23 | 36 | 7 | +0.00038 |
| train | 26 | +0.0003 | 0.0007 | 0.0021 | 8 | 14 | 4 | +0.00039 |
| test | 40 | +0.0004 | 0.0014 | 0.0051 | 15 | 22 | 3 | +0.00038 |

"Helped" / "hurt" use a 1e-4 J/m² tolerance on the change in absolute error.

**The effect is mixed across metrics, and favourable only on the held-out set.**
Relative to the common part alone, computed from the committed per-family CSV by
`make_parity_gamma66_ep13.py`:

| split | MAE | RMSE | signed bias | Pearson r |
|---|---|---|---|---|
| **test** | 0.005628 → 0.006008, **+6.75% worse** | 0.008470 → 0.008173, **−3.51% better** | +0.005243 → +0.005648, more positive | 0.9008 → 0.9175, **+1.86%, improved linear correlation** |
| train | 0.004386 → 0.004771, +8.78% worse | 0.006149 → 0.006738, +9.57% worse | +0.004197 → +0.004530, more positive | 0.9485 → 0.9382, −1.09% worse |
| all | 0.005139 → 0.005521, +7.43% worse | 0.007641 → 0.007640, −0.01%, unchanged | +0.004831 → +0.005207, more positive | 0.9182 → 0.9251, +0.75% better |

On held-out families the residual **lowers RMSE by 3.5%** while **raising MAE by 6.7%**
and making the positive bias larger, and it **improves the linear correlation**. RMSE
falling while MAE rises means it reduces the largest errors at the cost of the typical
one — the same thing the per-family counts show, 22 families worse against 15 better.

The asymmetry between splits is the part that deserves attention. On the training
families every metric degrades, RMSE by 9.6% and the correlation by 1.1%. A residual
that worsens training-set RMSE is not fitting this objective: it was trained on
per-configuration energies and forces, not on interface energies, and γ is a
near-cancelling difference of three large energies divided by twice the area. A
contribution carrying no interface-specific signal would look like noise plus a small
positive offset, which is what the numbers show. The held-out RMSE and correlation gains
should be read against that, and against n = 40.

An earlier version of this report stated that the residual "does not improve interface
energies". That was a one-sided reading of a mixed result: it is wrong about held-out
RMSE and about the correlation, and it is corrected here.

## 6. Descriptor repair

The script cross-checks its common-network part against the stand-alone common
evaluation `gamma66_common_holdout40_ew10000_v1.csv` and **prints a warning**:

```
common part vs stand-alone common evaluation (66 rows): max |difference| = 2.05e-02 J/m2
   <-- WARNING: larger than float noise; the two common-network evaluations disagree
```

This is expected and is not an error. That file dates from 2026-10-18 and predates the
descriptor repair. Evaluated against the same MATPes reference:

| common network | test MAE | bias | Pearson r |
|---|---:|---:|---:|
| pre-repair descriptor (superseded) | 0.0107 | +0.0099 | 0.8403 |
| repaired descriptor | **0.0056** | +0.0052 | **0.9008** |

The repair **halves the error** and the new values move toward the reference on every
large-difference family. The authoritative common-only numbers are the
`gamma_common_part_J_m2` column of this run, computed on identical structures in the
same forward pass as the full model.

No log survives for the stand-alone common run, so its artifacts cannot be documented
from first-hand evidence — the attribution above is inferred from the file's date and
the direction of the discrepancy. Anyone re-running this evaluation will see the same
warning.

## 7. Compute

CPU only. The GPU allocation was exhausted before this task; this evaluation is
single-point on frozen geometries and needs no GPU despite the script's docstring.

| | |
|---|---|
| job | `gamma66_ep13`, shared QOS, CPU |
| elapsed | **00:00:51** on 64 cores |
| GPU-hours used | **0.000** |
| GPU-hours remaining | 0.000 (exhausted prior to this task) |

## 8. Outputs

In `Gamma66_EP13_Deliverable_v1/`:

| file | contents |
|---|---|
| `gamma66_ep13_v1_per_family.csv` | 66 rows: case, origin, pair, split, N_AB/N_A/N_B, area, AB/A/B energies for full, common and residual parts, reference γ, both predictions, residual contribution, per-family errors and the change in absolute error |
| `gamma66_ep13_v1_summary.csv` | N, MAE, RMSE, bias, Pearson, slope for all/train/test × {full, common, pre-repair} |
| `gamma66_ep13_v1_residual_effect.csv` | per-split residual contribution and helped/hurt/unchanged counts |
| `gamma66_ep13_v1_stats.csv` | N, MAE, bias, RMSE, Pearson and slope per split, written alongside the figures. Reproduces §4 to all printed digits, from a separate code path -- retained as a cross-check, not as the authoritative table |
| `gamma66_ep13_v1_rows.csv` | the per-family table with the figures' display columns added (formatted interface label, pool split, drop reason). Same rows as the per-family CSV |
| `table_gamma66_ep13_v1_paper.{pdf,svg,png}` | two-panel table |
| `parity_gamma66_ep13_v1.{pdf,png}` | parity figure, regenerated — see the note below (PDF is the vector copy; no SVG) |
| `parity_gamma66_ep13_v1_residual_metrics.csv` | the §5 table as data: per split and metric, common-only vs full, absolute and percentage change |
| `gamma66_fullmoe_c1b_holdout40_e10000_ep13_v1.csv` | raw script output, before post-processing |
| `scripts/` | the executed code: evaluation script, post-processing, figure script, submit script |
| `logs/` | raw logs, path-scrubbed (see below) |
| `MANIFEST_SHA256.txt` | sha256 of every packaged file, of the four model artifacts, and of the executed code |

The PNG copies of both figures are in this package but are deliberately **not** committed
to the repository: they are raster duplicates of the PDFs, and the table PNG alone is
3.4 MB. The PDFs are vector, so the repository loses nothing.

**The parity figure's generator could not be recovered, and the figure was regenerated.**
The first version of this package named `make_parity_plot.py` as the generator. That was
wrong: it targets the GULP/Oishi comparison and contains no CSV writer, so it cannot
have produced this figure or the statistics CSV beside it. It has been removed from the
package. `interface_energy/make_gamma_parity_figure.py` (`2509906dcdd555d977040448`) was
also checked and excluded: it writes `{prefix}_stats.csv` with one row per *model* and
`index_label="model"`, whereas the committed `gamma66_ep13_v1_stats.csv` has one row per
*split* and an empty index label, and it distinguishes series by model rather than by
split. No script on disk matches the artifacts the original run left behind.

The figure is therefore regenerated by `make_parity_gamma66_ep13.py`
(`aad5a8a4e1bee96f`), written for this purpose and packaged in `scripts/`. It reads only
the committed per-family CSV, recomputes every annotation it draws, and **refuses to
write a figure at all** if those values disagree with the committed summary CSV beyond
5e-6 — so the figure cannot silently drift from the table it accompanies. It also checks
the frozen partition (26 train / 40 test) and aborts if the CSV does not carry it. All
66 points are plotted: training as blue open circles, test as green open squares,
against a 1:1 line, with axes labelled for MACE-MatPES-r2SCAN-0 and Local-MoE and the
legend reading "Training" and "Test".

Raw logs: `gamma66_fullmoe_ep13_v1_<jobid>.log`, `finalize_gamma66_ep13_v1.log`,
`parity_gamma66_ep13_v1.log`. Cluster paths, the username and allocation names in them
are replaced with `<scratch>`, `<user>` and `<alloc>` in the packaged copies; the content
is otherwise verbatim, every substitution is counted in `MANIFEST_SHA256.txt`, and the
originals on the cluster are unmodified. The packaged log hashes are therefore hashes of
the scrubbed copies, not of what SLURM wrote.

## 9. Reproduction

The SLURM log does not echo its own invocation, so what follows is not a reconstruction:
it is `run_gamma66_ep13_cpu.sh` as submitted, packaged in `scripts/`, with the allocation
name replaced. `$PROJ` and `$REPAIR` come from the shell environment file;
`$(dirname $PROJ)` is the directory holding both the branch checkout and
`External_Benchmark_Oishi2025/`, and the job was submitted from the latter.

```bash
#!/bin/bash
#SBATCH --qos=shared
#SBATCH --account=<alloc>
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=gamma66_ep13
#SBATCH --output=gamma66_fullmoe_ep13_v1_%j.log
set -u
source ~/.pilot_env
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MOE_HET="$(dirname $PROJ)"
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
python3 "$(dirname $PROJ)/moe-interface-potential/interface_energy/eval_gamma66_fullmoe.py" \
  --het        "$(dirname $PROJ)" \
  --repo       "$(dirname $PROJ)/moe-interface-potential" \
  --c1b-checkpoint     "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
  --common-checkpoint  "$REPAIR/pilot/common/best_checkpoint.pt" \
  --routing-checkpoint "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
  --torchsoap-module   "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
  --stage-root         "$PROJ/data/expanded_holdout40_v1_c1b_stage" \
  --manifest           gamma66_manifest.csv \
  --device cpu \
  --out gamma66_fullmoe_c1b_holdout40_e10000_ep13_v1.csv
```

Then, from the same directory:

```bash
python3 finalize_gamma66_ep13.py \
  --csv       gamma66_fullmoe_c1b_holdout40_e10000_ep13_v1.csv \
  --prerepair gamma66_common_holdout40_ew10000_v1.csv \
  --out-dir   Gamma66_EP13_Deliverable_v1 --stem gamma66_ep13_v1
```

The parity figure and the §5 residual table are regenerated from the committed CSVs
alone — no model, no checkpoint, no GPU:

```bash
python3 make_parity_gamma66_ep13.py \
  --csv     Gamma66_EP13_Deliverable_v1/gamma66_ep13_v1_per_family.csv \
  --summary Gamma66_EP13_Deliverable_v1/gamma66_ep13_v1_summary.csv \
  --out-dir Gamma66_EP13_Deliverable_v1 \
  --stem    parity_gamma66_ep13_v1
```

Two things here are load-bearing. Passing every checkpoint path explicitly is required,
not optional — see §1. And `--repo` must be the branch checkout that contains both
`interface_energy/` and `training/`, because `--c1b-script` is resolved from it: pointing
it at another checkout of this project's code will either fail on import or, worse,
rebuild the C1b model class from a different definition.

## 10. Limitations

- The reference is MACE-MATPes-r2SCAN, not experiment. These are agreement statistics
  against another model, not accuracy against measurement.
- Geometries are MATPes-relaxed and frozen. Neither model relaxes them, so the numbers
  are single-point energies on a geometry neither model chose.
- No uncertainty estimate accompanies any prediction.
- The pre-repair comparison in §6 rests on an undocumented file; see the caveat there.
- The job log does not record its own invocation. The command in §9 is the submit script
  itself, which is the authoritative record and is packaged, but nothing in the log
  independently confirms that this script is what produced that log beyond the job id in
  its filename and its 17:02 timestamp preceding the 19:54 outputs.
