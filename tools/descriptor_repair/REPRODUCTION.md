# Reproduction

Every command below was executed as given. Paths are written `<proj>`, `<repo>`,
`<ext>` for the data workspace, this repository and the external benchmark
directory; `<alloc_gpu>` and `<alloc_cpu>` for the Slurm accounts. Submit scripts
in each package's `scripts/` carry the resource requests actually used.

## Environment

    source ~/.pilot_env          # exports PROJ, REPAIR, PILOT, CP
    conda activate silica_moe    # batch jobs must source miniconda explicitly;
                                 # NERSC `module load conda` gives a base without it

## Artifact verification

All four artifacts are hash-checked before every run; a mismatch aborts.

    sha256sum <proj>/descriptor_repair/pilot/common/best_checkpoint.pt \
              <proj>/descriptor_repair/pilot/router/best_torchsoap_routing_gate.pt \
              <proj>/descriptor_repair/pilot/c1b/best_c1b_residual_model.pt \
              <proj>/descriptor_repair/torchsoap_corrected_rotinv_v1.py

## Item 6: dynamics

10 ps protocol cohort, 52 configurations.

The 52-case cohort is driven by `run_md_cohort_ep13.slurm`, not by
`run_md_pilot_ep13_cpu.sh` (which runs the single record `external_0219` into
`md_ep13_pilot_v1`). The launcher requires `MD_JOB` and splits the manifest
across five jobs of four 16-thread slots each; a case whose
`nve_final_structure.extxyz` exists is skipped, so resubmission is safe:

    for J in 0 1 2 3 4; do
        MD_JOB=$J sbatch -A <alloc_cpu> \
            dynamics_pilot/scripts/run_md_cohort_ep13.slurm
    done

`OUT_ROOT` defaults to `<proj>/md_ep13_v1`, and the launcher hashes the four
artifacts plus the trainer, runner and calculator into each slot log. Then the
figure:

    python3 <repo>/md/plot_local_moe_nve_drift.py \
        --md-root <proj>/md_ep13_v1 --out-dir <out> \
        --prefix LocalMoE_ep13_nve10ps --duration-ps 10 --limit 0.1 --paper --dpi 600

500 ps cohort, ten configurations.

`run_md_long_ep13.slurm` runs ONE record and requires `REC`
(`REC="${REC:?set REC=<record_id>}"`), so it is submitted once per record from
the pre-registered manifest. Resubmitting the same command resumes that record
from its `restart.npz`:

    for REC in $(tail -n +2 long_dynamics/metrics/long_manifest.csv | cut -d, -f1); do
        REC=$REC sbatch -A <alloc_cpu> \
            long_dynamics/scripts/run_md_long_ep13.slurm
    done

The ten records took 19 executions in four batches because of those resumes;
each writes `<long-root>/logs/${REC}.log` and a Slurm log per execution, and all
19 are in `long_dynamics/logs/`.

Then the figure:

    python3 <repo>/md/plot_local_moe_nve_drift.py \
        --md-root <proj>/md_ep13_long500ps_v1 \
        --manifest <proj>/md_ep13_long500ps_v1/long_manifest_plot.csv \
        --out-dir <out> --prefix LocalMoE_ep13_long500ps \
        --log-name nve_500ps_log.csv --summary-name long_nve_summary.json \
        --duration-ps 500 --limit 0.1 --paper --dpi 600

`--log-name` and `--summary-name` are required for the long runs: they write
`nve_500ps_log.csv` and `long_nve_summary.json`, and the plotter defaults to the
10 ps names.

## Item 5: routing

    sbatch -A <alloc_gpu> routing_diagnostics/scripts/run_routing_ep13_gpu.slurm

The submit script passes all four artifacts explicitly and uses a fresh
`--out-dir` with `--recompute`. Both matter: the script's defaults resolve to the
pre-repair trio and the pre-repair descriptor, and it caches region means per
structure, so a shared output directory would serve stale rows.

## Item 8: invariance and conservativity

    sbatch -A <alloc_gpu> invariance/scripts/run_rotinv_ep13_gpu.slurm

Runs in `<proj>/rotation_audit/` because the harness imports
`test_torchsoap_rotation` and `local_moe_ase` as siblings, and writes its JSON
elsewhere so the October results are untouched. Step two repeats with
`--positive-control`, which must FAIL for the step-one PASS to carry information.

## Item 7: cost

    sbatch -A <alloc_gpu> cost_benchmark/scripts/run_cost_ep13_gpu.slurm

`--constraint=gpu&hbm80g` is required: a 40 GB A100 is not comparable with the
archived 80 GB measurement.

## Verifying the committed statistics

Recompute every 500 ps drift statistic from the raw logs and compare with the
runner's recorded values (tolerance 1e-6 relative):

    python3 - <<'PY'
    import numpy as np, pandas as pd, json, glob
    for case in sorted(glob.glob("long_dynamics/case_*")):
        s = json.load(open(case + "/long_nve_summary.json"))
        d = pd.read_csv(case + "/nve_500ps_log.csv")
        e = d.total_energy_per_atom_eV.to_numpy(float)
        dm = 1000.0 * (e - e[0]); t = d.time_ps.to_numpy(float)
        A = np.vstack([t, np.ones_like(t)]).T
        mine = {"max_abs_drift_meV_per_atom": float(np.abs(dm).max()),
                "final_drift_meV_per_atom": float(dm[-1]),
                "peak_to_peak_meV_per_atom": float(dm.max() - dm.min()),
                "linear_drift_meV_per_atom_per_ps":
                    float(np.linalg.lstsq(A, dm, rcond=None)[0][0])}
        bad = [k for k, v in mine.items()
               if abs(v - s[k]) / max(abs(s[k]), 1e-30) > 1e-6]
        print(s["record_id"], "OK" if not bad else "MISMATCH %s" % bad)
    PY

Check every package against its manifest:

    for m in */metrics/MANIFEST_SHA256.txt; do
        pkg=$(dirname $(dirname "$m"))
        awk 'NR>FNR{next}' /dev/null "$m"
        sed -n '/^Package files/,$p' "$m" | grep -E '^    [0-9a-f]{64}' \
          | while read -r h f; do
              printf '%s  %s\n' "$h" "$pkg/$f"
            done | sha256sum -c -
    done

## Verifying provenance as executed

Each execution log prints the artifact hashes before work begins, so provenance
does not rest on the post-hoc sidecars:

    grep -h -A6 '^=\{20,\}$' long_dynamics/logs/slurm_*.txt | grep -E '^[0-9a-f]{24} '
