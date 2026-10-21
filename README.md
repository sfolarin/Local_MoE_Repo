# Local-MoE interface potential for SiO2

Code to reproduce the results of: **[paper title]** ([authors], [year]).

[One paragraph: a local mixture-of-experts interatomic potential, a common network plus a router over seven
classical-potential channels and a trained residual; trained on MACE-MatPES-r2SCAN labels of silica
interfaces; interface energies comparable to MACE-MatPES-r2SCAN at a fraction of the cost.]

## Layout

| folder | contents |
|---|---|
| `src/local_moe/` | TorchSOAP descriptor (all-pairs and neighbour-list versions), ASE calculator |
| `data_splits/` | holdout-by-pair split, split audits, manifests |
| `training/` | common network, router distillation, residual trainers and submit scripts |
| `evaluation/` | parity export and energy/force parity figure |
| `interface_energy/` | interfacial-energy evaluation, tables, parity figures, GULP check |
| `cost/` | cost benchmark and figure |
| `md/` | NVE molecular dynamics (10 ps and 500 ps), analysis, routing from MD |
| `deploy/` | export of the model for the LAMMPS port, NumPy reference implementation |
| `tests/` | checks that back specific claims (no split leakage, neighbour list equals all-pairs) |
| `figures/` | schematic figures |
| `data_construction/` | (optional) how the interface structures were built |
| `docs/expected_results.md` | the numbers each stage should reproduce |

## Environment

`environment/` pins the conda environments: [`silica_moe.yml`, `silica_mpi.yml`]. MD runs used ASE 3.26.0 and
torch 2.8.0+cu128. [GPU: NVIDIA A100 80 GB.]

## Data and paths

The data are an input, not part of the repository (see `docs/data.md` [to write]). Point the scripts at them with

    export MOE_HET=/path/to/workspace      # holds Local_MoE_MATPES_R2SCAN_Expanded_v1/ and External_Benchmark_Oishi2025/
    export CONDA_ENV=/path/to/conda/envs/silica_mpi    # used by the SLURM scripts

Run every command from the repository root. SLURM scripts contain `--account=YOUR_ALLOCATION`: replace it or pass
`sbatch --account=<yours> ...`. No other path is machine specific.

## Reproducing the results

Run the stages in order. `docs/expected_results.md` lists what each stage should print.

1. **Splits:** `data_splits/` [commands]
2. **Training:** `training/` [commands]
3. **Evaluation:** `evaluation/` [commands]
4. **Interface energies:** `interface_energy/` [commands]
5. **Cost:** `cost/` [commands]
6. **MD:** `md/` [commands]

## Paper element -> script

| paper element | script | input | output |
|---|---|---|---|
| Architecture figure | `figures/make_architecture_figure.py` | `routing_nve_region_means.csv` | `Local_MoE_architecture_v2.*` |
| Energy / force parity | `evaluation/plot_energy_force_parity.py --paper` | `raw_predictions` of the full model | `energy_force_parity_paper.*` |
| Interfacial-energy parity (Local-MoE vs MACE-MatPES) | `interface_energy/make_gamma_parity_figure.py` | `gamma66_fullmoe_*.csv` | `gamma_parity_*.*` |
| Interfacial-energy table | `interface_energy/make_oishi_gamma_table_and_parity.py --paper` | `gamma66_fullmoe_*.csv` | `table_*_paper.*` |
| GULP versus Oishi parity | `interface_energy/make_parity_plot.py` | comparison CSV | `parity_plot_20.png` |
| Cost figure | `cost/plot_computational_cost_holdout40.py --paper` | `computational_cost_raw.csv` | `computational_cost_holdout40_paper.*` |
| NVE energy conservation | `md/plot_local_moe_nve_drift.py --paper` | `md_holdout40_v1/` | `LocalMoE_holdout40_nve_paper.*` |
| NVE protocol facts | `md/nve_caption_facts.py` | `md_holdout40_v1/` | printed |
| Routing from MD | `md/routing_from_md.py` | `md_holdout40_v1/` | routing figures |

## Licence and citation

[Licence: TBD with the advisor / institution.] [Citation.]
