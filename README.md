# Local-MoE interface potential for SiO2

Code to reproduce the results of: **[paper title]** ([authors], [year]).

A local mixture-of-experts (MoE) interatomic potential for silica interfaces. A common network gives the base
prediction. A router, distilled from a dense-gate teacher and then frozen, weights seven classical-potential channels
atom by atom (ML-BKS, Original BKS, Chik BKS, PSO1 BKS, PSO2 BKS, Tersoff, Vashishta). A trained residual closes the
remaining gap. The model is trained on MACE-MatPES-r2SCAN energies and forces of 512 silica structures and gives
interfacial energies comparable to MACE-MatPES-r2SCAN on held-out interface families at a fraction of the cost.

## Layout

| folder | contents |
|---|---|
| `src/local_moe/` | TorchSOAP descriptor (all-pairs and neighbour-list versions) and the ASE calculators |
| `data_splits/` | the holdout-40 split (tables shipped here) and the scripts that built and audited it |
| `training/` | features, classical-expert forces, common network, router (dense gate, distillation), residual |
| `potentials/` | Tersoff and Vashishta parameter files used by the classical channels |
| `evaluation/` | parity export and energy/force parity figure |
| `interface_energy/` | interfacial-energy evaluation, tables and parity figures |
| `cost/` | cost benchmark and figure |
| `md/` | NVE molecular dynamics, analysis, routing from MD |
| `deploy/` | export of the model for the LAMMPS port, NumPy reference implementation |
| `tests/` | checks that back specific claims (no split leakage, neighbour list equals all-pairs, scripts equal those saved in the run folders) |
| `environment/` | conda environments and the LAMMPS environment template |
| `docs/data.md` | the data the scripts expect (folders, sizes, splits, checkpoints) |
| `docs/expected_results.md` | the numbers each stage should reproduce |
| `MANIFEST.tsv` | SHA-256 of every script and of the original file it was made from |

## Requirements

- **Python environment:** `environment/silica_mpi.yml` (training, expert forces, MD; the SLURM scripts use it) and
  `environment/silica_moe.yml` (analysis and figures). Training runs used Python 3.9.19, PyTorch 2.8.0+cu128
  (CUDA 12.8) and NumPy 1.26.4; MD used ASE 3.26.0. GPU: NVIDIA A100 80 GB (the largest structures have 4608 atoms;
  `training/submit_common_holdout40.sh` requests 80 GB cards).
- **LAMMPS** (only for the classical-expert forces, stage 3 below): unmodified LAMMPS, commit
  `36f7c81ced8dd456430b13647e48f0c7d7bd8faa` (28 Oct 2025), built with CMake in `Release` mode as an MPI shared
  library with the Python module. Packages enabled: MANYBODY, KSPACE, MOLECULE, RIGID and EXTRA-COMPUTE (all others
  off). The build provides the pair styles `tersoff`, `vashishta` and `buck/coul/long`, `table`, and the `ewald` and
  `pppm` solvers. `environment/env_perlmutter.sh` shows the modules and library path used
  on NERSC Perlmutter; edit the `/path/to/...` placeholders for your machine.
- **Potential files:** `potentials/Tersoff.pot` (Munetoh et al., Comput. Mater. Sci. 39, 334 (2007)) and
  `potentials/Vashishta.pot` (Broughton, Meli, Vashishta and Kalia, Phys. Rev. B 56, 611 (1997)) are the files
  distributed with LAMMPS, with their original headers. The expert-force stage checks their SHA-256.

## Data and paths

The data are an input and are not part of the repository (`docs/data.md` describes them). Point the scripts at them:

    export REPO=/path/to/this/repository
    export MOE_HET=/path/to/workspace          # holds Local_MoE_MATPES_R2SCAN_Expanded_v1/ and External_Benchmark_Oishi2025/
    export PROJ=$MOE_HET/Local_MoE_MATPES_R2SCAN_Expanded_v1
    export CONDA_ENV=/path/to/conda/envs/silica_mpi    # used by the SLURM scripts

Run the commands below from `$PROJ`. Submit the SLURM scripts from `$REPO` (they locate the repository from the submit
directory, or from `MOE_REPO`). They contain `--account=YOUR_ALLOCATION` (replace it or pass `sbatch --account=<yours>`)
and Perlmutter queue settings (`-q premium`, `--constraint="gpu&hbm80g"`) to adapt to your machine.

## Reproducing the paper model

The model in the paper is the holdout-40 model: 369 training, 38 validation and 105 test structures. Checkpoints are
chosen on the validation split; the test split is never used for selection. Folder names follow the runs in the paper.

**0. Inputs.** `data/expanded_holdout40_v1/` (24 MB: 512 labelled structures and the split) and the snapshot folder of
the earlier baseline run, `runs/size_pilot_prep_20260918T223242_204092Z/` (5.7 MB), which supplies the energy
reference (-9.668443844547909 eV/atom). The split tables are also in `data_splits/`. To rebuild the dataset folder
from the label run:

    python3 $REPO/data_splits/prepare_expanded_moe_splits_holdout.py \
        --label-run runs/matpes_labels_4608_20260918_v1 --out data/expanded_holdout40_v1 \
        --split-table $REPO/data_splits/holdout_split_n40.csv

It prints `train 369, val 38, test 105`.

**1. Descriptor features** (SOAP, n_max 6, l_max 4, r_cut 5 A, plus a species one-hot: 392 inputs per atom):

    export SNAP=$PROJ/runs/size_pilot_prep_20260918T223242_204092Z
    python3 $REPO/training/prepare_expanded_moe_features.py compute  --dataset data/expanded_holdout40_v1 --snapshot $SNAP --out data/expanded_holdout40_v1_features --threads 16
    python3 $REPO/training/prepare_expanded_moe_features.py finalize --dataset data/expanded_holdout40_v1 --snapshot $SNAP --out data/expanded_holdout40_v1_features

**2. Common network.** Four energy weights were trained (10, 1000, 3200, 10000); the paper uses 10000.

    cd $REPO && sbatch --account=<yours> training/submit_common_holdout40.sh     # the four weights in parallel on one node
    # or one weight, from $PROJ:
    python3 $REPO/training/train_expanded_moe_common.py train --features data/expanded_holdout40_v1_features \
        --out runs/common_holdout40_ew10000_v1 --energy-weight 10000 --resume --threads 16 --max-hours 5.3

Result: `runs/common_holdout40_ew10000_v1/best_checkpoint.pt`.

**3. Classical-potential energies and forces (LAMMPS).** Run as recorded for the paper; its output (936 MB) is an input to
stage 4 and is described in `docs/data.md`. Re-running needs the LAMMPS build above, `potentials/`, and the earlier expert dataset's
`dataset_info.json` (see the notes below).

    export LMP_MOE=/path/to/lammps_moe
    source $REPO/environment/env_perlmutter.sh
    RUN=$PROJ/data/expanded_holdout40_v1_experts_v2
    python3 $REPO/training/prepare_expanded_moe_experts.py prepare --run $RUN --het $MOE_HET --dataset $PROJ/data/expanded_holdout40_v1 \
        --lmp-python "$LMP_MOE/python" --lmp-lib "$LMP_MOE/build-mpi/liblammps.so.0"
    S="srun --exact -N1 -n1 -c32 --cpu-bind=cores env MPICH_GPU_SUPPORT_ENABLED=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16"
    $S python3 -u $REPO/training/prepare_expanded_moe_experts.py pilot   --run $RUN
    $S python3 -u $REPO/training/prepare_expanded_moe_experts.py compute --run $RUN --max-hours 1     # bounded per call; see --help
    python3 $REPO/training/prepare_expanded_moe_experts.py summary --run $RUN

**4. Router.** Dense gate trained on the expert forces, exported as a per-atom routing teacher, then distilled into a gate
that works on differentiable TorchSOAP features.

    python3 $REPO/training/prepare_expanded_routing_dataset_holdout40.py --dataset data/expanded_holdout40_v1 \
        --expert-run data/expanded_holdout40_v1_experts_v2 --out data/expanded_holdout40_v1_routing
    python3 -u $REPO/training/train_local_gate_from_expert_forces_holdout40.py --dataset_root data/expanded_holdout40_v1_routing \
        --output_dir runs/dense_gate_holdout40_v1 --preserve_manifest_split --device cuda
    python3 $REPO/training/export_dense_gate_routing_teacher_expanded.py --dataset_root data/expanded_holdout40_v1 \
        --trained_root runs/dense_gate_holdout40_v1 --output_root data/expanded_holdout40_v1_teachers --device cuda
    python3 $REPO/training/prepare_expanded_torchsoap_distillation_dataset_holdout40.py --features data/expanded_holdout40_v1_features \
        --teachers data/expanded_holdout40_v1_teachers --routing-dataset data/expanded_holdout40_v1_routing \
        --out data/expanded_holdout40_v1_distillation
    python3 -u $REPO/training/train_local_torchsoap_gate_distillation_expanded.py --dataset_root data/expanded_holdout40_v1_distillation \
        --output_root runs/torchsoap_gate_distillation_holdout40_v1 --device cuda --expected_split_sizes 369 38 105

Result: `runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt` (the router; frozen from here on).

**5. Residual (C1b).** Trained with the common network and the router frozen. Energy weights 10 (the default) and 10000
were trained; the paper uses 10000.

    python3 $REPO/training/prepare_expanded_c1b_stage_root_holdout40.py --dataset data/expanded_holdout40_v1 --out data/expanded_holdout40_v1_c1b_stage
    python3 -u $REPO/training/train_c1b_holdout40.py --stage_b1_root data/expanded_holdout40_v1_c1b_stage \
        --common_checkpoint runs/common_holdout40_ew10000_v1/best_checkpoint.pt \
        --routing_checkpoint runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt \
        --torchsoap_module $REPO/src/local_moe/torchsoap_autograd_module_periodic_regnorm.py \
        --output_root runs/c1b_holdout40_e10000_v1 --energy_weight 10000 --device cuda

Result: `runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt`, the full Local-MoE model. The four checkpoints of the
paper model together are about 3.3 MB.

## Evaluation, interface energies, cost and MD

`docs/expected_results.md` lists what each script should print. Every script documents its options with `--help`.

| paper element | script | input | output |
|---|---|---|---|
| Split audit | `tests/check_gate_split.py` | `GATE_SPLIT HOLDOUT_TABLE ROUTING_MANIFEST` (three CSVs) | printed |
| Energy / force parity | `evaluation/export_c1b_parity_predictions.py`, `evaluation/export_common_parity_predictions_v2.py`, then `evaluation/plot_energy_force_parity.py --paper` | the checkpoints above | `energy_force_parity_paper.*` |
| Error by region | `evaluation/eval_test_regions_common_vs_full.py` | the checkpoints above | region metrics |
| Interfacial energy, full model and common part only | `interface_energy/eval_gamma66_fullmoe.py`, `interface_energy/eval_gamma66_common_only.py` | the checkpoints above, `gamma66_manifest.csv` (from `data_splits/make_gamma66_manifest.py`) | `gamma66_*.csv` |
| Interfacial-energy parity and table | `interface_energy/make_gamma_parity_figure.py`, `interface_energy/make_oishi_gamma_table_and_parity.py --paper` | `gamma66_*.csv` | `gamma_parity_*`, `table_*_paper.*` |
| Cost figure | `cost/benchmark_cost_holdout40_nl.py`, then `cost/plot_computational_cost_holdout40.py --paper` | the checkpoints above | `computational_cost_holdout40_paper.*` |
| NVE energy conservation | `md/make_md_manifest.py`, `md/submit_md_jobs.sh`, `md/md_status.py`, `md/plot_local_moe_nve_drift.py --paper` | the checkpoints above | `LocalMoE_holdout40_nve_paper.*` |
| NVE protocol facts, routing from MD | `md/nve_caption_facts.py`, `md/routing_from_md.py` | the MD runs | printed, routing figures |
| Export for the LAMMPS port | `deploy/export_local_moe_spec.py`, `deploy/local_moe_reference_numpy.py` | the checkpoints above | model spec, NumPy check |

## Notes on re-running

- **Stages that check hashes.** Several stages record the SHA-256 of the scripts and inputs they ran with and refuse to
  continue when a later stage sees something different. The scripts here are the originals with one change: a short
  header that replaces machine-specific paths, so their hashes differ from the originals. `MANIFEST.tsv` gives both
  hashes for every file, and `tests/verify_provenance.py` compares the originals with the copies of the scripts that
  each run folder saved (`src_snapshot/`). A fresh run of a stage with these scripts is consistent with itself;
  re-checking an existing paper run folder with them is refused by design.
- **Stage 3 (expert forces)** compares its inputs with an earlier expert dataset (`dataset_info.json`), the original
  calculator and the two potential files. Treat it as provenance and use the stored expert forces to run stages 4-5.
- **Older names.** Some modules carry the name of the first training set they were written for
  (`..._196train.py`). The feature and common-network code load them by name, so they are kept unchanged.
- **Earlier split.** The same 512 structures were first split 410 / 44 / 58 (`expanded_grouped_4608_v1`). The paper
  model does not use it; it is only the pool from which `data_splits/make_oishi_holdout_split.py` derived the holdout.

## Licence and citation

[Licence: to be added.] [Citation: to be added.] The two potential files keep the headers they have in the LAMMPS
distribution.
