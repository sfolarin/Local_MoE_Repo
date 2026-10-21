#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --qos=premium
#SBATCH --constraint="gpu&hbm80g"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#
# One premium job = one node = 4 GPU slots. Slot s of job J runs the manifest rows with job == J and slot == s,
# one structure after another (relax -> NPT 5 ps -> NVE 10 ps). Submit with submit_md_jobs.sh, or by hand:
#   MD_JOB=0 sbatch run_md_job.slurm
# Resubmitting a job is safe: a structure whose nve_final_structure.extxyz exists is skipped.

# repository root: submit/run from the repository root, or set MOE_REPO
export REPO="${MOE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -d "$REPO/training" ] || { echo "run this from the repository root (or set MOE_REPO)" >&2; exit 1; }
set -euo pipefail

MD_JOB="${MD_JOB:?Set MD_JOB=0..4}"

HET="${MOE_HET:?set_MOE_HET_to_the_data_workspace_folder_see_README}"
PROJ="$HET/Local_MoE_MATPES_R2SCAN_Expanded_v1"
OUT_ROOT="${OUT_ROOT:-$PROJ/md_holdout40_v1}"
MANIFEST="${MANIFEST:-$OUT_ROOT/md_manifest.csv}"
CODE="${CODE:-${REPO}/md}"          # run_local_moe_md.py and local_moe_ase.py live here
CONDA_ENV="${CONDA_ENV:?set_CONDA_ENV_to_the_path_of_the_conda_environment_see_environment}"   # the environment of the earlier MD run
DTYPE="${DTYPE:-float64}"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_ROOT/logs"
python3 - <<'PY'
import ase, torch
print("ase", ase.__version__, "| torch", torch.__version__, "| cuda available on the batch host:", torch.cuda.is_available())
PY

export HET PROJ OUT_ROOT MANIFEST CODE DTYPE MD_JOB
export PYTHONPATH="$CODE:${PYTHONPATH:-}"

srun \
  --exact \
  -N1 \
  -n4 \
  -c32 \
  --gpus-per-task=1 \
  --mem-per-gpu=55G \
  --cpu-bind=cores \
  bash -c '
    set -euo pipefail
    export OMP_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    export OPENBLAS_NUM_THREADS=16
    export NUMEXPR_NUM_THREADS=16
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores
    export MPICH_GPU_SUPPORT_ENABLED=0

    LOG="$OUT_ROOT/logs/job${MD_JOB}_slot${SLURM_PROCID}.log"
    {
        echo "JOB=$MD_JOB SLOT=$SLURM_PROCID HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} started $(date)"
    } >> "$LOG"

    python3 -u "$CODE/run_local_moe_md.py" \
      --manifest "$MANIFEST" \
      --job "$MD_JOB" \
      --slot "$SLURM_PROCID" \
      --out-root "$OUT_ROOT" \
      --dtype "$DTYPE" \
      --device cuda \
      >> "$LOG" 2>&1

    echo "SLOT FINISHED $(date)" >> "$LOG"
  '
