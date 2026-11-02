#!/bin/bash
#SBATCH --qos=premium
#SBATCH --account=<alloc>
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --job-name=md_preflight_ep13
#SBATCH --output=md_preflight_ep13_v1_%j.log
# CPU preflight for the selected Local-MoE: common ep116, router ep110, residual ep13,
# repaired descriptor. Route parity against the trainer, then finite-difference forces
# and stress. Every artifact path explicit -- the runner's defaults point elsewhere.
set -u
source ~/.pilot_env
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
HETDIR="$(dirname "$PROJ")"; B="$HETDIR/moe-interface-potential"
export OMP_NUM_THREADS=64 MKL_NUM_THREADS=64

echo "== artifact hashes as used =="
for f in "$REPAIR/pilot/common/best_checkpoint.pt" \
         "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
         "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
         "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
         "$B/training/train_c1b_holdout40.py"; do
  echo "  $(sha256sum "$f" | cut -c1-24)  $(basename "$f")"
done
echo "== stage root present: $([ -d "$PROJ/data/expanded_holdout40_v1_c1b_stage" ] && echo yes || echo NO) =="

A=( --het "$HETDIR" --repo "$B"
    --trainer-script     "$B/training/train_c1b_holdout40.py"
    --stage-root         "$PROJ/data/expanded_holdout40_v1_c1b_stage"
    --common-checkpoint  "$REPAIR/pilot/common/best_checkpoint.pt"
    --routing-checkpoint "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt"
    --c1b-checkpoint     "$REPAIR/pilot/c1b/best_c1b_residual_model.pt"
    --torchsoap-module   "$REPAIR/torchsoap_corrected_rotinv_v1.py"
    --device cpu --dtype float64 )

echo; echo "== 1. route parity vs the trainer =="
time python3 "$B/md/run_local_moe_md.py" "${A[@]}" --parity external_0219

echo; echo "== 2. finite-difference forces and stress =="
time python3 "$B/md/run_local_moe_md.py" "${A[@]}" --check external_0219
