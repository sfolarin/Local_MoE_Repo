#!/bin/bash
#SBATCH --qos=shared
#SBATCH --account=<alloc_cpu>
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# One 500 ps NVE trajectory per job, 16 threads (the measured optimum).
# Starts from the SELECTED model's post-NPT state in md_ep13_v1, not the pre-repair
# root the script defaults to. Restartable: a job that hits the wall limit resumes
# from restart.npz with the original energy baseline intact and no velocity rescale.
set -euo pipefail
source ~/.pilot_env
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
HETDIR="$(dirname "$PROJ")"; B="$HETDIR/moe-interface-potential"
REC="${REC:?set REC=<record_id>}"
LONG="${LONG_ROOT:-$PROJ/md_ep13_long500ps_v1}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
       NUMEXPR_NUM_THREADS=16 OMP_PROC_BIND=true OMP_PLACES=cores

echo "record $REC   long-root $LONG   10ps source $PROJ/md_ep13_v1"
for f in "$REPAIR/pilot/common/best_checkpoint.pt" \
         "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
         "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
         "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
         "$B/md/run_local_moe_long_nve.py" \
         "$B/training/train_c1b_holdout40.py"; do
  echo "  $(sha256sum "$f" | cut -c1-24)  $(basename "$f")"
done

python3 -u "$B/md/run_local_moe_long_nve.py" \
  --het "$HETDIR" \
  --md-root  "$PROJ/md_ep13_v1" \
  --out-root "$LONG" \
  --common-checkpoint  "$REPAIR/pilot/common/best_checkpoint.pt" \
  --routing-checkpoint "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
  --c1b-checkpoint     "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
  --torchsoap-module   "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
  --records "$REC" --device cpu --dtype float64 \
  2>&1 | tee -a "$LONG/logs/${REC}.log"
