# Computational cost on the repaired model

Timing and memory for the selected Local-MoE against MACE, remeasured because the
archived cost figure was produced with pre-repair artifacts.

## Model

| role | path | sha256 (first 24) |
|---|---|---|
| common network | `pilot/common/best_checkpoint.pt` | `5b17849752a535739b16cfc8` |
| routing gate | `pilot/router/best_torchsoap_routing_gate.pt` | `9e8e0ca6af4976ef8c74c34e` |
| C1b residual | `pilot/c1b/best_c1b_residual_model.pt` | `c7ad2a89c570549309ad8906` |
| descriptor | `torchsoap_corrected_rotinv_v1.py` | `0afd9a1ba7b9a24f8bf8151a` |

## Why it was remeasured

The archived S8 run used the pre-repair trio and descriptor, which is what
`benchmark_cost_holdout40_nl.py` resolves to by default. A first replacement
landed on a 40 GB A100 and was discarded as not comparable; this run pins
`--constraint=gpu&hbm80g` and reports `NVIDIA A100-SXM4-80GB, 81920 MiB, driver
580.178.04`, 11 m 34 s wall, Slurm job 60881973, torch 2.8.0+cu128, 105
configurations, 5 warmup and 30 repeat evaluations each.

Both internal gates passed at approximately 1e-16: common-only against the full
model with a zeroed residual (tolerance 1e-7), and predictions against the
stored export (tolerance 1e-4).

## Three statistic families, which are not interchangeable

An earlier version of this report mixed these. They are reported separately.

### 1. Aggregate timings: median over structures of each per-structure median

| model | median_of_medians_ms | n_ok | n_oom | max N |
|---|---|---|---|---|
| MACE r2SCAN float64 | 112.993577 | 105 | 0 | 4608 |
| MACE r2SCAN float32 | 90.032201 | 105 | 0 | 4608 |
| Local-MoE common only | 12.099392 | 105 | 0 | 4608 |
| Local-MoE full | 13.046559 | 105 | 0 | 4608 |

The ratio of two entries in this column is a **ratio of aggregate statistics**
and is not the median of paired per-configuration ratios below, even where the
two happen to be numerically close.

### 2. Peak memory: MAXIMUM over configurations, not a median

| model | peak_mem_max_GiB |
|---|---|
| MACE r2SCAN float64 | 30.803709 |
| MACE r2SCAN float32 | 15.415611 |
| Local-MoE common only | 0.554940 |
| Local-MoE full | 0.549447 |

### 3. Paired per-configuration speedups and memory reductions, n = 105

| pairing | mean | median | min | max | median paired memory reduction |
|---|---|---|---|---|---|
| MACE float64 / Local-MoE full | 11.579194 | **8.676736** | 2.320586 | 39.556375 | **98.081212%** |
| MACE float32 / Local-MoE full | 9.156812 | **6.929799** | 2.314945 | 30.072631 | **96.175662%** |
| MACE float64 / Local-MoE common | 12.498004 | 9.399082 | 2.526027 | 42.478261 | 98.065092% |
| MACE float32 / Local-MoE common | 9.883870 | 7.477505 | 2.511203 | 32.293987 | 96.143531% |

Any speedup quoted in the text must name both the baseline (float64 or float32)
and the model (full or common only).

## Configurations that enumerate periodic images

Per configuration, against MACE float64, full model:

| subset | n | median | min | max |
|---|---|---|---|---|
| images enumerated | **14** | 2.912119 | 2.320586 | 15.804266 |
| no images enumerated | **91** | 9.173889 | 3.319295 | 39.556375 |
| all | 105 | 8.676736 | 2.320586 | 39.556375 |

These are per-configuration populations. The published figure merges
configurations into **63 family-level markers**, of which ten involve
image-enumerating structures with a grouped median of about 3.07x against 9.39x
for the rest. The two groupings give different counts (14 against 10) and
different medians because they are different populations, and a text drawing on
either must say which.

## Two further qualifications

**The repaired descriptor is about 14% slower than the pre-repair one**, so any
speedup quoted from the archived figure is optimistic. The numbers above
supersede it.

**The measured time exponent of 0.08 for N >= 1000 is not an asymptotic
complexity.** It describes this hardware over this size range, where the model is
far from compute-bound, and should not be presented as a scaling law.

## Limitations

- One GPU model, one driver, one software stack.
- `benchmark_cost_holdout40_nl.py` records the script and descriptor hashes, the
  GPU, torch, CUDA, Python, host and Slurm job, but **does not hash the
  checkpoints**. The run log prints all four artifact hashes contemporaneously.
- Timing is per single-configuration evaluation, not MD throughput. The 500 ps
  runs in `long_dynamics` are the throughput evidence.

## Files

`figures/` common-only and full-model figures as PDF, PNG and SVG with plot data.
`metrics/` the raw per-structure timings (420 rows = 105 configurations x 4
models) and the summary JSON. `logs/`, `scripts/` the run log and submit script.
Hashes in `metrics/MANIFEST_SHA256.txt`.
