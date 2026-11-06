#!/bin/bash
#SBATCH -J rotinv_ep13
#SBATCH --constraint=gpu&hbm80g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00
#SBATCH -o rotinv_ep13_%j.out
#SBATCH -e rotinv_ep13_%j.err
# ---------------------------------------------------------------------------
# run_rotinv_ep13_gpu.slurm -- rotational invariance, translation and
# conservativity of the SELECTED (descriptor-repaired) Local-MoE.
#
#     sbatch -A <alloc> run_rotinv_ep13_gpu.slurm
#
# WHY THIS RUN EXISTS.  rotation_audit/C0_descriptor_verdicts.txt (2026-10-25
# 18:37 PT) found BOTH pre-repair descriptors non-invariant: per-centre
# rel-L2 1.250, global 0.977, power spectrum failing at l=2,3,4 while passing
# at l=0,1 -- identical in float32 and float64, so a formulation error rather
# than precision.  torchsoap_corrected_rotinv_v1.py was written at
# 2026-10-26 01:05 PT, i.e. AFTER those verdicts, and has never itself been
# put through the harness.  The entire premise of the repair is therefore
# unmeasured.  This run measures it.
#
# WHY EVERY PATH IS EXPLICIT.  The script resolves unset checkpoints to
#     runs/common_holdout40_ew10000_v1/best_checkpoint.pt
#     runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt
#     runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt
# (all pre-repair), --repo to $REPO (the WRONG repository in this project),
# and --reference-dir to eval_parity_v1/c1b_holdout40_e10000_v1/
# raw_predictions/test -- pre-repair stored predictions that line 377 compares
# against, which would manufacture a meaningless mismatch.
#
# RUN LOCATION.  The script does sys.path.insert(0, Path(__file__).parent) and
# imports test_torchsoap_rotation and local_moe_ase as SIBLINGS, so it must run
# in rotation_audit/.  Its --json goes to a new directory instead, so the
# October results stay intact for comparison.  Nothing in rotation_audit/ is
# overwritten.
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/.pilot_env
HETDIR="$(dirname "$PROJ")"
B="$HETDIR/moe-interface-potential"

source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
conda activate silica_moe

export OMP_NUM_THREADS=32
export MPLCONFIGDIR="$SCRATCH/.mplconfig"; mkdir -p "$MPLCONFIGDIR"

RA="$PROJ/rotation_audit"
OUT="$PROJ/rotation_audit_ep13_v1"
STAGE="$PROJ/data/expanded_holdout40_v1_c1b_stage"
REFDIR="$REPAIR/final/moe/raw_predictions/test"

COMMON="$REPAIR/pilot/common/best_checkpoint.pt"
ROUTER="$REPAIR/pilot/router/best_torchsoap_routing_gate.pt"
C1B="$REPAIR/pilot/c1b/best_c1b_residual_model.pt"
SOAP="$REPAIR/torchsoap_corrected_rotinv_v1.py"
TRAINER="$B/training/train_c1b_holdout40.py"

echo "job     : ${SLURM_JOB_ID:-interactive} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

check () {   # $1 role   $2 path   $3 expected 24-hex prefix
    if [ ! -f "$2" ]; then echo "FAIL: missing $1 -> $2"; exit 1; fi
    h="$(sha256sum "$2" | cut -c1-24)"
    if [ "$h" != "$3" ]; then echo "FAIL: $1 sha256 $h, expected $3"; exit 1; fi
    printf "  ok  %-13s %s\n" "$1" "$h"
}
echo "artifact verification (selected model):"
check common     "$COMMON" 5b17849752a535739b16cfc8
check routing    "$ROUTER" 9e8e0ca6af4976ef8c74c34e
check c1b        "$C1B"    c7ad2a89c570549309ad8906
check descriptor "$SOAP"   0afd9a1ba7b9a24f8bf8151a
# The audit imports its own sibling copy of the MD calculator. It was verified
# byte-identical to the repository's src/local_moe/local_moe_ase.py; if that
# ever drifts, the audit stops measuring the code path the MD and cost runs use.
check calculator "$RA/local_moe_ase.py" 787eb6fc6a98b678a9162ed6
if ! cmp -s "$RA/local_moe_ase.py" "$B/src/local_moe/local_moe_ase.py"; then
    echo "FAIL: rotation_audit/local_moe_ase.py differs from the repository copy"
    exit 1
fi
echo "  ok  calculator matches $B/src/local_moe/local_moe_ase.py"
for f in "$RA/test_rotational_invariance_local_moe.py" "$RA/test_torchsoap_rotation.py" "$TRAINER"; do
    [ -f "$f" ] || { echo "FAIL: missing $f"; exit 1; }
    printf "  ok  %-13s %s  %s\n" "support" "$(sha256sum "$f" | cut -c1-24)" "$(basename "$f")"
done
echo

mkdir -p "$OUT"
A=( --proj "$PROJ" --repo "$B"
    --c1b-script         "$TRAINER"
    --common-checkpoint  "$COMMON"
    --routing-checkpoint "$ROUTER"
    --c1b-checkpoint     "$C1B"
    --torchsoap-module   "$SOAP"
    --stage-root         "$STAGE"
    --device cuda:0
    --routes trainer32,calc32,calc64
    --n-test 6 --max-atoms 1500 --nrot 8 --seed 12345 )

# Line 377 compares an identity evaluation against stored predictions. Pointing
# this at the pre-repair export would report a mismatch that says nothing about
# invariance, so it is either the repaired model's own export or nothing at all.
if [ -d "$REFDIR" ]; then
    A+=( --reference-dir "$REFDIR" )
    echo "reference predictions: $(basename "$(dirname "$(dirname "$REFDIR")")")/raw_predictions/test"
    echo "  npz files: $(find "$REFDIR" -name '*.npz' | wc -l)"
else
    echo "WARNING: $REFDIR absent -- running WITHOUT --reference-dir."
    echo "         The stored-prediction cross-check is skipped; it is NOT being"
    echo "         compared against the pre-repair export."
fi
echo

echo "=============== 1. the repaired descriptor, as shipped ==============="
cd "$RA"
srun -n 1 python3 "$RA/test_rotational_invariance_local_moe.py" "${A[@]}" \
    --json "$OUT/rotinv_ep13_v1.json"

echo
echo "=============== 2. positive control (harness exactness) =============="
# Replaces the descriptor with a subclass carrying a correctly weighted angular
# basis. Diagnostic only: it is NOT the trained input. Its purpose is to show the
# harness still separates a correct basis from a broken one, so that a PASS in
# step 1 carries information.
srun -n 1 python3 "$RA/test_rotational_invariance_local_moe.py" "${A[@]}" \
    --positive-control --json "$OUT/rotinv_ep13_v1_positive_control.json" || \
    echo "(positive control exited non-zero; step 1 above is the result that matters)"

echo
echo "=== outputs ==="
for f in "$OUT"/*; do
    printf "  %-52s %s\n" "$(basename "$f")" "$(sha256sum "$f" | cut -c1-16)"
done
echo
echo "October results in $RA are untouched:"
ls "$RA"/C_*.json 2>/dev/null | while read -r f; do echo "  $(basename "$f")"; done
echo
echo "NOTE: measurements only. Nothing staged, committed or pushed."
