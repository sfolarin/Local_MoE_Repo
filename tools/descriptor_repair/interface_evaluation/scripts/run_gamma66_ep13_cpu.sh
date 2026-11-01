#!/bin/bash
#SBATCH --qos=shared
#SBATCH --account=<alloc>
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --job-name=gamma66_ep13
#SBATCH --output=gamma66_fullmoe_ep13_v1_%j.log
# Full Local-MoE interface energies, 66-family manifest, epoch-13 residual.
# CPU: GPU allocation exhausted, and this is single-point evaluation on frozen
# geometries. Shared QOS: one process, 64 threads, no need for a whole node.
# Every artifact path is passed explicitly -- the script's defaults point at
# DIFFERENT checkpoints (fde94fa6.../625915d4...) than the ones specified.
set -u
source ~/.pilot_env
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export MOE_HET="$(dirname $PROJ)"
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
python3 "$(dirname $PROJ)/moe-interface-potential/interface_energy/eval_gamma66_fullmoe.py" \
  --het        "$(dirname $PROJ)" \
  --repo       "$(dirname $PROJ)/moe-interface-potential" \
  --c1b-checkpoint     "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
  --common-checkpoint  "$REPAIR/pilot/common/best_checkpoint.pt" \
  --routing-checkpoint "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
  --torchsoap-module   "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
  --stage-root         "$PROJ/data/expanded_holdout40_v1_c1b_stage" \
  --manifest           gamma66_manifest.csv \
  --device cpu \
  --out gamma66_fullmoe_c1b_holdout40_e10000_ep13_v1.csv
