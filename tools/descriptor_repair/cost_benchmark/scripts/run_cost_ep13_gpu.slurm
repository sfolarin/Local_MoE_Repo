#!/bin/bash
#SBATCH --qos=premium
#SBATCH --account=<alloc_gpu>
#SBATCH --constraint=gpu&hbm80g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --job-name=cost_ep13
#SBATCH --output=cost_ep13_v1_%j.log
# S8 computational cost for the SELECTED model. The archived eval_cost_v2_nl run used
# the pre-repair artifacts (fde94fa6 / 625915d4 / 99d0cfae, descriptor 30acc083) and
# therefore measures a different model; every artifact path is explicit here.
set -u
source ~/.pilot_env
export CUDA_VISIBLE_DEVICES=0
source "<scratch>/sd/f/$USER/miniconda3/etc/profile.d/conda.sh"
conda activate silica_moe
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
HETDIR="$(dirname "$PROJ")"; B="$HETDIR/moe-interface-potential"

echo "== artifacts =="
for f in "$REPAIR/pilot/common/best_checkpoint.pt" \
         "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
         "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
         "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
         "$B/training/train_c1b_holdout40.py" \
         "$B/cost/benchmark_cost_holdout40_nl.py"; do
  echo "  $(sha256sum "$f" | cut -c1-24)  $(basename "$f")"
done
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python3 -u "$B/cost/benchmark_cost_holdout40_nl.py" \
  --het "$HETDIR" --repo "$B" \
  --c1b-script         "$B/training/train_c1b_holdout40.py" \
  --stage-root         "$PROJ/data/expanded_holdout40_v1_c1b_stage" \
  --common-checkpoint  "$REPAIR/pilot/common/best_checkpoint.pt" \
  --routing-checkpoint "$REPAIR/pilot/router/best_torchsoap_routing_gate.pt" \
  --c1b-checkpoint     "$REPAIR/pilot/c1b/best_c1b_residual_model.pt" \
  --torchsoap-module   "$REPAIR/torchsoap_corrected_rotinv_v1.py" \
  --reference-dir      "$PROJ/descriptor_repair/final/moe/raw_predictions/test" \
  --out                "$PROJ/eval_cost_ep13_v1" \
  --max-atoms 5000 --device cuda:0
