# Data

The scripts read a workspace folder (`$MOE_HET`) that is not part of this repository. This page lists what is in it,
which stage makes or uses each folder, and what is shipped in `data_splits/`. Sizes are those of the paper runs.

[Availability and licence of the data: to be added.]

## Workspace layout

    $MOE_HET/
      Local_MoE_MATPES_R2SCAN_Expanded_v1/        ($PROJ: every command in the README runs from here)
        runs/
          matpes_labels_4608_20260918_v1/         labelled structures (input of stage 0)
          size_pilot_prep_20260918T223242_204092Z/  snapshot of the earlier baseline run (5.7 MB)
          common_holdout40_ew{10,1000,3200,10000}_v1/   stage 2
          dense_gate_holdout40_v1/                stage 4
          torchsoap_gate_distillation_holdout40_v1/     stage 4
          c1b_holdout40_e{10,10000}_v1/           stage 5
        data/
          expanded_holdout40_v1/                  stage 0 (24 MB)
          expanded_holdout40_v1_features/         stage 1 (825 MB)
          expanded_holdout40_v1_experts_v2/       stage 3 (936 MB)
          expanded_holdout40_v1_routing/          stage 4 (260 KB)
          expanded_holdout40_v1_teachers/         stage 4 (31 MB)
          expanded_holdout40_v1_distillation/     stage 4 (838 MB)
          expanded_holdout40_v1_c1b_stage/        stage 5 (9.2 MB)
      External_Benchmark_Oishi2025/               interface-energy benchmark inputs and results

`data/expanded_holdout40_v1_experts/` (no `_v2`) is a pilot run of the expert stage that holds no shards. The
paper uses `_experts_v2` only.

## The dataset: `data/expanded_holdout40_v1`

512 silicon-dioxide structures with MACE-MatPES-r2SCAN energies and forces. One NumPy `.npz` file per structure in
`shards/{train,val,test}/<record_id>.npz`, plus these tables:

| file | content |
|---|---|
| `manifest.csv` | one row per structure (columns below) |
| `train.csv`, `val.csv`, `test.csv` | the same rows, split by `split` |
| `group_assignments.csv` | one row per group (261): group, split, reason, record ids |
| `split_summary.json` | counts, seeds and hashes of the inputs of the split |

Columns of `manifest.csv`: `record_id`, `source`, `source_record`, `family`, `phase_pair_group`, `split_group`,
`split`, `existing_split`, `n_atoms`, `geometry_sha256`, `data_file`, `data_sha256`, `source_label_file`,
`source_label_sha256`, `label_provenance`, `has_region_metadata`, `energy_eV_per_atom`, `max_force_eV_A`.

Structures are assigned to a split by group (`split_group`, 261 groups), so no group is divided between train,
validation and test.

| source | train | val | test | total |
|---|---|---|---|---|
| Oishi_all179 | 112 | 15 | 52 | 179 |
| Oishi_relaxed52 | 20 | 1 | 31 | 52 |
| Own_augmented | 196 | 22 | 22 | 240 |
| Siddharth | 41 | 0 | 0 | 41 |
| **total** | **369** | **38** | **105** | **512** |

`Own_augmented` kept its original assignment (196 / 22 / 22). The split is the one in `split_summary.json`;
checkpoints are chosen on the validation split only.

## What is shipped in `data_splits/`

Copies of the split definition without cluster paths or label values:

| file | content | sha-256 (first 16) |
|---|---|---|
| `holdout40_records.csv` | `manifest.csv` without `source_label_file`, `energy_eV_per_atom`, `max_force_eV_A` (15 columns, 512 rows) | `b3ea344bb95fbfe3` |
| `holdout40_groups.csv` | `group_assignments.csv` (261 rows) | `6762117d7c456bab` |
| `holdout40_split_summary.json` | `split_summary.json` with paths made relative | `c05fab2676a70e37` |
| `holdout_split_n40.csv` | the split table passed to `prepare_expanded_moe_splits_holdout.py --split-table` (512 rows: `record_id`, `source`, `family`, `pair`, `old_split`, `new_split`, `gamma_family`, `n_atoms`, `geometry_sha256`) | `0591f8fdfbb9a046` |

`data_splits/export_split_tables.py` writes them from the dataset folder. The shipped tables hold identifiers, groups,
splits, atom counts and geometry hashes. They hold no energies, forces or structures.

## Derived folders

| folder | made by | content |
|---|---|---|
| `expanded_holdout40_v1_features` | `prepare_expanded_moe_features.py` | TorchSOAP features per structure (390 SOAP values + 2 = 392 per atom), train-only normalisation statistics |
| `expanded_holdout40_v1_experts_v2` | `prepare_expanded_moe_experts.py` (LAMMPS) | energies and forces of the seven classical channels for all 512 structures (512 shards) |
| `expanded_holdout40_v1_routing` | `prepare_expanded_routing_dataset_holdout40.py` | the dense-gate training set, with the split of the dataset preserved |
| `expanded_holdout40_v1_teachers` | `export_dense_gate_routing_teacher_expanded.py` | per-atom routing weights of the trained dense gate |
| `expanded_holdout40_v1_distillation` | `prepare_expanded_torchsoap_distillation_dataset_holdout40.py` | features and teacher weights for the distilled router |
| `expanded_holdout40_v1_c1b_stage` | `prepare_expanded_c1b_stage_root_holdout40.py` | the root folder the residual trainer reads |

The seven classical channels are ML-BKS, Original BKS, Chik BKS, PSO1 BKS and PSO2 BKS (pair potentials), Tersoff and
Vashishta (parameter files in `potentials/`).

## Checkpoints of the paper model

All four are small. Checkpoints are not in the repository.

| file | stage | size |
|---|---|---|
| `runs/common_holdout40_ew10000_v1/best_checkpoint.pt` | 2, common network (energy weight 10000) | 546 KB |
| `runs/dense_gate_holdout40_v1/` best checkpoint | 4, dense gate (135,175 parameters, best epoch 70) | 1.64 MB |
| `runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt` | 4, router (best epoch 93) | 549 KB |
| `runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt` | 5, residual (best epoch 3) | 546 KB |

The energy reference used by the common network is -9.668443844547909 eV/atom. It is inherited from the baseline run
whose snapshot is in `runs/size_pilot_prep_20260918T223242_204092Z/baseline/common.pt`.

## Interface-energy benchmark

`External_Benchmark_Oishi2025/` holds the inputs of the interface-energy evaluation: the family-level split table, the
candidate list, the relaxed interface and component structures, and the MACE-MatPES-r2SCAN reference energies. The
manifest `gamma66_manifest.csv` is built from them by `data_splits/make_gamma66_manifest.py`.
[Describe the 66 interfaces and the reference they are compared with.]
