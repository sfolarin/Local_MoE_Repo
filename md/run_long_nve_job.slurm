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
# 500 ps NVE for the 10 structures of long_nve_manifest.csv on ONE node (4 GPU slots, one process per GPU).
# Slot s runs the manifest rows with slot == s, one after another: slots 0 and 1 have three structures, slots 2 and 3 two.
# At 70-91 steps/s (measured in the 10 ps runs) one structure takes about 3.1-3.9 h, so the longest slot needs ~11 h.
#
# Resubmitting is safe: a finished structure is skipped and an unfinished one resumes from its last checkpoint
# (every 50,000 steps = 25 ps). Use "sbatch --qos=premium ..." to jump the queue.
#
# One-time setup (from $PROJ):
#   mkdir -p md_long500ps_v1/logs && cp scripts_md/long_nve_manifest.csv md_long500ps_v1/long_manifest.csv
#   sbatch ${REPO}/md/run_long_nve_job.slurm

# repository root: submit/run from the repository root, or set MOE_REPO
export REPO="${MOE_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -d "$REPO/training" ] || { echo "run this from the repository root (or set MOE_REPO)" >&2; exit 1; }
set -euo pipefail

HET="${MOE_HET:?set_MOE_HET_to_the_data_workspace_folder_see_README}"
PROJ="$HET/Local_MoE_MATPES_R2SCAN_Expanded_v1"
MD_ROOT="${MD_ROOT:-$PROJ/md_holdout40_v1}"            # the finished 10 ps runs (start states)
OUT_ROOT="${OUT_ROOT:-$PROJ/md_long500ps_v1}"
MANIFEST="${MANIFEST:-$OUT_ROOT/long_manifest.csv}"
CODE="${CODE:-${REPO}/md}"                       # run_local_moe_long_nve.py, run_local_moe_md.py, local_moe_ase.py
CONDA_ENV="${CONDA_ENV:?set_CONDA_ENV_to_the_path_of_the_conda_environment_see_environment}"
DTYPE="${DTYPE:-float64}"
NVE_PS="${NVE_PS:-500}"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$OUT_ROOT/logs"
python3 - <<'PY'
import ase, torch
print("ase", ase.__version__, "| torch", torch.__version__, "| cuda available on the batch host:", torch.cuda.is_available())
PY

export HET PROJ MD_ROOT OUT_ROOT MANIFEST CODE DTYPE NVE_PS
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

    LOG="$OUT_ROOT/logs/slot${SLURM_PROCID}.log"
    {
        echo "SLOT=$SLURM_PROCID HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} started $(date)"
    } >> "$LOG"

    python3 -u "$CODE/run_local_moe_long_nve.py" \
      --manifest "$MANIFEST" \
      --slot "$SLURM_PROCID" \
      --md-root "$MD_ROOT" \
      --out-root "$OUT_ROOT" \
      --nve-ps "$NVE_PS" \
      --dtype "$DTYPE" \
      --device cuda \
      >> "$LOG" 2>&1

    echo "SLOT FINISHED $(date)" >> "$LOG"
  '
