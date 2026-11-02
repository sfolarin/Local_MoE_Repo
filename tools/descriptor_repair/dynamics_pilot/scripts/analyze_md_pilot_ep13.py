#!/usr/bin/env python3
"""
analyze_md_pilot_ep13.py -- verify and report the Local-MoE MD pilot.

The runner writes final_protocol_summary.json with its own drift statistics. This
recomputes every one of them from the raw NVE log and aborts if they disagree: a
summary that is not reproducible from the trajectory it describes is not evidence.

Everything is referenced to the INITIAL NVE SAMPLE as the baseline, which is what makes
"drift" mean anything -- the NPT stage ends at a different energy, and quoting an
excursion against the wrong zero silently changes the number.

Reported, with units:
    baseline total energy                        eV and eV/atom
    maximum absolute excursion |E(t) - E0|       meV/atom
    final change E(end) - E0                     meV/atom
    peak-to-peak                                 meV/atom
    fitted linear drift (least squares vs t)     meV/atom/ps
    temperature mean / min / max                 K
    relaxation convergence                       eV/A, steps
    measured rates and wall time                 steps/s, s

It also projects the cost of extending the cohort from the measured rates and the
manifest's atom counts, under stated assumptions rather than as a bare number.

    python3 analyze_md_pilot_ep13.py --case <.../case_external_0219> --out-dir <dir>
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TOL_REL = 1e-6          # agreement required against the runner's own summary
CONSERVE_LIMIT = 0.1    # meV/atom, the runner's criterion


def fit_slope(t_ps, y):
    """Least-squares slope of y against t, returned in y-units per ps."""
    A = np.vstack([t_ps, np.ones_like(t_ps)]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][0])


def relax_convergence(case: Path):
    """Last Fmax reported by each optimiser log, and the step counts."""
    out = {}
    for name in ("relax_fire.log", "relax_polish.log"):
        p = case / name
        if not p.is_file():
            continue
        rows = [l for l in p.read_text().splitlines() if re.search(r"\d\s*$", l)]
        nums = [re.findall(r"[-+]?\d*\.\d+|\d+", l) for l in rows]
        nums = [n for n in nums if len(n) >= 2]
        if nums:
            out[name] = dict(steps=len(nums), final_fmax_eV_per_A=float(nums[-1][-1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="the case_<record> directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stem", default="md_pilot_ep13_v1")
    ap.add_argument("--manifest", default=None,
                    help="md_manifest.csv, for the cohort projection")
    args = ap.parse_args()

    case = Path(args.case).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    summ = json.load(open(case / "final_protocol_summary.json"))
    nve = pd.read_csv(case / "nve_10ps_log.csv")
    npt = pd.read_csv(case / "npt_5ps_log.csv")
    n = int(summ["n_atoms"])
    print("case %s   N=%d   pair=%s   split=%s   dtype=%s"
          % (summ["record_id"], n, summ["phase_pair"], summ["split"], summ["dtype"]))
    print("NVE samples: %d over %.2f ps" % (len(nve), nve.time_ps.iloc[-1]))

    # ---- baseline and drift, recomputed -------------------------------------
    e0_tot = float(nve.total_energy_eV.iloc[0])
    e0_at = float(nve.total_energy_per_atom_eV.iloc[0])
    d_meV = 1000.0 * (nve.total_energy_per_atom_eV.to_numpy(float) - e0_at)

    mine = {
        "nve_max_abs_energy_drift_meV_per_atom": float(np.abs(d_meV).max()),
        "nve_final_energy_drift_meV_per_atom": float(d_meV[-1]),
        "nve_peak_to_peak_energy_meV_per_atom": float(d_meV.max() - d_meV.min()),
        "nve_linear_drift_meV_per_atom_per_ps": fit_slope(nve.time_ps.to_numpy(float), d_meV),
        "nve_mean_temperature_K": float(nve.temperature_K.mean()),
        "nve_min_temperature_K": float(nve.temperature_K.min()),
        "nve_max_temperature_K": float(nve.temperature_K.max()),
    }

    print("\nbaseline (initial NVE sample)")
    print("  total energy            %.9f eV" % e0_tot)
    print("  total energy per atom   %.12f eV/atom" % e0_at)

    print("\nrecomputed vs the runner's summary")
    bad = []
    for k, v in mine.items():
        ref = summ.get(k)
        if ref is None:
            print("  %-42s %+.6e   (not in summary)" % (k, v))
            continue
        rel = abs(v - ref) / max(abs(ref), 1e-30)
        flag = "ok" if rel <= TOL_REL else "MISMATCH"
        if rel > TOL_REL:
            bad.append((k, ref, v, rel))
        print("  %-42s %+.6e  summary %+.6e  rel %.1e  %s" % (k, v, ref, rel, flag))
    if bad:
        print("\nFAIL: the summary is not reproducible from the trajectory")
        for k, ref, v, rel in bad:
            print("  %s: summary %.9e, recomputed %.9e (rel %.2e)" % (k, ref, v, rel))
        return 1

    conserving = mine["nve_max_abs_energy_drift_meV_per_atom"] < CONSERVE_LIMIT
    print("\nenergy conserving: %s  (max |excursion| %.6f meV/atom, limit %.1f)"
          % (conserving, mine["nve_max_abs_energy_drift_meV_per_atom"], CONSERVE_LIMIT))
    if conserving != bool(summ.get("energy_conserving", False)):
        print("FAIL: disagrees with the summary's own verdict")
        return 1

    # ---- NPT and relaxation --------------------------------------------------
    print("\nNPT (5 ps, 300 K, 1 bar)")
    for k in ("initial_pressure_GPa", "npt_mean_pressure_GPa", "npt_final_pressure_GPa",
              "npt_mean_temperature_K", "npt_volume_change_percent"):
        print("  %-34s %+.6f" % (k, summ[k]))
    print("  pressure mean from log             %+.6f GPa" % npt.pressure_GPa.mean())
    print("\nvelocity rescale  %.6f K -> %.6f K  (factor %.9f)"
          % (summ["temperature_before_velocity_rescale_K"],
             summ["temperature_after_velocity_rescale_K"],
             summ.get("velocity_scale_factor", float("nan"))))

    rc = relax_convergence(case)
    print("\nrelaxation")
    for k, v in rc.items():
        print("  %-18s %d steps, final Fmax %.4f eV/A" % (k, v["steps"], v["final_fmax_eV_per_A"]))

    # ---- cost ----------------------------------------------------------------
    npt_s = float(summ["npt_walltime_seconds"])
    nve_s = float(summ["nve_walltime_seconds"])
    print("\nmeasured cost at N=%d on CPU" % n)
    print("  NPT %8.2f s  (%.2f steps/s)" % (npt_s, summ["npt_steps_per_second"]))
    print("  NVE %8.2f s  (%.2f steps/s)" % (nve_s, summ["nve_steps_per_second"]))
    print("  per case (15 ps) %.1f min" % ((npt_s + nve_s) / 60.0))

    rows = [dict(quantity=k, value=v) for k, v in mine.items()]
    rows += [dict(quantity="baseline_total_energy_eV", value=e0_tot),
             dict(quantity="baseline_total_energy_per_atom_eV", value=e0_at),
             dict(quantity="n_atoms", value=n),
             dict(quantity="npt_walltime_seconds", value=npt_s),
             dict(quantity="nve_walltime_seconds", value=nve_s),
             dict(quantity="npt_steps_per_second", value=summ["npt_steps_per_second"]),
             dict(quantity="nve_steps_per_second", value=summ["nve_steps_per_second"]),
             dict(quantity="energy_conserving", value=int(conserving))]
    pd.DataFrame(rows).to_csv(out / f"{args.stem}_nve_stats.csv", index=False)

    if args.manifest and Path(args.manifest).is_file():
        man = pd.read_csv(args.manifest)
        tot_atoms = int(man.n_atoms.sum())
        per_atom_s = (npt_s + nve_s) / n
        print("\ncohort projection, %d cases, %d atoms in total" % (len(man), tot_atoms))
        print("  measured %.3f s per atom per 15 ps at N=%d" % (per_atom_s, n))
        print("  LINEAR in N:        %.1f core-node-hours wall if run one at a time"
              % (per_atom_s * tot_atoms / 3600.0))
        print("  largest single case N=%d -> %.1f h"
              % (man.n_atoms.max(), per_atom_s * man.n_atoms.max() / 3600.0))
        print("  500 ps NVE at N=%d  -> %.1f h (1e6 steps at the measured NVE rate)"
              % (n, 1e6 / summ["nve_steps_per_second"] / 3600.0))
        print("\n  Assumption: cost linear in N, which the neighbour-list descriptor")
        print("  supports but 64-thread efficiency does not -- a 162-atom cell")
        print("  underuses the node, so larger cases will cost LESS per atom than")
        print("  this. Treat the linear figure as an upper bound and measure one")
        print("  mid-size case before committing the cohort.")
        pd.DataFrame([dict(cases=len(man), total_atoms=tot_atoms,
                           seconds_per_atom_15ps=per_atom_s,
                           linear_cohort_hours=per_atom_s * tot_atoms / 3600.0,
                           largest_case_atoms=int(man.n_atoms.max()),
                           nve_500ps_hours_at_pilot_size=1e6 / summ["nve_steps_per_second"] / 3600.0)
                      ]).to_csv(out / f"{args.stem}_cost_projection.csv", index=False)

    print("\nwrote %s_nve_stats.csv" % args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
