#!/bin/bash -l
#SBATCH -A YOUR_ALLOCATION
#SBATCH --constraint="gpu&hbm80g"
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -t 06:00:00
#SBATCH -J common_holdout40
#SBATCH -o common_holdout40_%j.log
# repository root: submit/run from the repository root, or set MOE_REPO
export REPO="${MOE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -d "$REPO/training" ] || { echo "run this from the repository root (or set MOE_REPO)" >&2; exit 1; }
export PROJ=${MOE_HET:?set_MOE_HET_to_the_data_workspace_folder_see_README}/Local_MoE_MATPES_R2SCAN_Expanded_v1
export PATH=${CONDA_ENV:?set_CONDA_ENV_to_the_path_of_the_conda_environment_see_environment}/bin:$PATH
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $PROJ
srun --ntasks=4 --gpus-per-task=1 --gpu-bind=single:1 bash -c '
EW=(10 1000 3200 10000); W=${EW[$SLURM_PROCID]}
python3 ${REPO}/training/train_expanded_moe_common.py train --features data/expanded_holdout40_v1_features \
  --out runs/common_holdout40_ew${W}_v1 --energy-weight $W --resume --threads 16 --max-hours 5.3 \
  >> logs_common_holdout40_ew${W}.log 2>&1'
