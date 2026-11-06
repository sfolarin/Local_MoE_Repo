#!/bin/bash
#SBATCH --qos=shared
#SBATCH --account=<alloc_cpu>
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# One job = 4 slots x 16 threads = 64 cores (half a CPU node, billed proportionally
# under shared QOS). 16 threads is the measured optimum: at N=1008 the wall time is
# 6.12 s against 6.89 s at 64 threads, for one quarter of the cores.
# Slot s of job J runs the manifest rows with job == J and slot == s, serially.
# A case whose nve_final_structure.extxyz exists is skipped, so external_0219 is
# reused and resubmission is safe.
set -euo pipefail
source ~/.pilot_env
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
HETDIR="$(dirname "$PROJ")"; B="$HETDIR/moe-interface-potential"
MD_JOB="${MD_JOB:?set MD_JOB=0..4}"
OUT="${OUT_ROOT:-$PROJ/md_ep13_v1}"
mkdir -p "$OUT/logs"

echo "job $MD_JOB  out-root $OUT"
for f in "$REPAIR/pilot/common/best_checkpoint.pt" \
         "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
         "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
         "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
         "$B/training/train_c1b_holdout40.py" \
         "$B/md/run_local_moe_md.py" \
         "$B/src/local_moe/local_moe_ase.py"; do
  echo "  $(sha256sum "$f" | cut -c1-24)  $(basename "$f")"
done

export HETDIR B OUT MD_JOB REPAIR PROJ
srun --exact -N1 -n4 -c16 --cpu-bind=cores bash -c '
  set -euo pipefail
  export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
         NUMEXPR_NUM_THREADS=16 OMP_PROC_BIND=true OMP_PLACES=cores
  LOG="$OUT/logs/job${MD_JOB}_slot${SLURM_PROCID}.log"
  echo "JOB=$MD_JOB SLOT=$SLURM_PROCID HOST=$(hostname) started $(date)" >> "$LOG"
  python3 -u "$B/md/run_local_moe_md.py" \
    --het "$HETDIR" --repo "$B" \
    --trainer-script     "$B/training/train_c1b_holdout40.py" \
    --stage-root         "$PROJ/data/expanded_holdout40_v1_c1b_stage" \
    --common-checkpoint  "$REPAIR/pilot/common/best_checkpoint.pt" \
    --routing-checkpoint "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
    --c1b-checkpoint     "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
    --torchsoap-module   "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
    --manifest "$OUT/md_manifest.csv" --job "$MD_JOB" --slot "$SLURM_PROCID" \
    --out-root "$OUT" --device cpu --dtype float64 \
    >> "$LOG" 2>&1
  echo "SLOT FINISHED $(date)" >> "$LOG"
'
