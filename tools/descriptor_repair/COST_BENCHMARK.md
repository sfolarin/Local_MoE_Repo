# Computational cost on the repaired model

Timing and memory for the selected Local-MoE against MACE, remeasured because
the archived cost figure was produced with pre-repair artifacts.

## Model

| role | path | sha256 (first 24) |
|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |

## Why it was remeasured

The archived S8 cost run used the pre-repair trio and the pre-repair descriptor.
`benchmark_cost_holdout40_nl.py` resolves all four to those by default. Since
the repaired descriptor is measurably slower, the archived numbers are not a
description of the model the paper reports.

The first replacement attempt also had to be discarded: it landed on a 40 GB
A100, which is not comparable with the archived 80 GB run. The run in this
package used `--constraint=gpu&hbm80g` and reports
`NVIDIA A100-SXM4-80GB, 81920 MiB, driver 580.178.04`, 11 m 34 s wall.

## Gates

The benchmark refuses to report unless two internal gates pass. G1 checks the
common-only path against the full model with a zeroed residual (tolerance 1e-7);
G2 checks predictions against the stored export (tolerance 1e-4). Both passed at
approximately 1e-16.

## Result

| model | time per evaluation | peak memory |
|---|---|---|
| MACE float64 | 112.99 ms | 30.80 GiB |
| MACE float32 | 90.03 ms | 15.42 GiB |
| Local-MoE, common only | 12.10 ms | 0.55 GiB |
| Local-MoE, full | 13.05 ms | 0.55 GiB |

Speedups of the full Local-MoE over MACE: **mean 11.58x, median 8.68x, range
2.32x to 39.56x**. Memory reduction **98.1%**.

## Three qualifications that belong in the text

**The repaired descriptor is about 14% slower than the pre-repair one.** Any
speedup quoted from the archived figure is therefore optimistic, and the numbers
above supersede it.

**The measured time exponent of 0.08 for N >= 1000 is not an asymptotic
complexity.** It describes this hardware over this size range, where the model
is far from compute-bound. It should not be presented as a scaling law.

**Ten of the 63 markers enumerate periodic images.** The 105 test records merge
into 63 family-level markers across 62 families; ten of those involve
structures whose evaluation enumerates periodic images, and they carry a median
speedup of 3.07x against 9.39x for the rest. Reporting a single median across
all markers mixes two regimes.

## Limitations

- One GPU model (A100-80GB), one driver, one software stack: torch 2.8.0+cu128.
- `benchmark_cost_holdout40_nl.py` records the script and descriptor hashes, the
  GPU, torch, CUDA, Python, host and Slurm job, but **does not hash the
  checkpoints**. The artifact hashes above were verified separately, before and
  after, and are recorded in the run log.
- Timing is per evaluation of a single configuration, not an MD throughput
  measurement. The 500 ps runs in `long_dynamics` are the throughput evidence.

## Files

`figures/` the common-only and full-model cost figures as PDF, PNG and SVG with
their plot data. `metrics/` the raw per-structure timings and the summary JSON.
`logs/`, `scripts/` the run log and submit script. Hashes in
`metrics/MANIFEST_SHA256.txt`.
