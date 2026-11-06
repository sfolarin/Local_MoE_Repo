#!/bin/bash
#SBATCH -J routing_ep13
#SBATCH --constraint=gpu&hbm80g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --gpus-per-node=4
#SBATCH -t 04:00:00
#SBATCH -o routing_ep13_%j.out
#SBATCH -e routing_ep13_%j.err
# ---------------------------------------------------------------------------
# run_routing_ep13_gpu.slurm -- routing diagnostics for the SELECTED
# (descriptor-repaired) Local-MoE, on the repaired NVE trajectories.
#
# Submit with the account on the command line, so this file carries no
# allocation identifier, no username and no scratch path and can be
# committed as it stands:
#
#     sbatch -A <alloc> run_routing_ep13_gpu.slurm
#
# WHY EVERY PATH IS PASSED EXPLICITLY.  routing_from_md.py lines 95-98 default
# to the pre-repair artifacts:
#     runs/common_holdout40_ew10000_v1/best_checkpoint.pt
#     runs/torchsoap_gate_distillation_holdout40_v1/best_torchsoap_routing_gate.pt
#     runs/c1b_holdout40_e10000_v1/best_c1b_residual_model.pt
#     src/local_moe/torchsoap_autograd_module_periodic_regnorm_nl.py
# and --md-root defaults to md_holdout40_v1, the pre-repair trajectories.
# Accepting any one of those defaults measures a model the paper does not
# describe; that is how four earlier artifact families had to be discarded.
# All four are therefore given from $REPAIR and hashed against their pinned
# prefixes BEFORE the model is loaded.  A mismatch aborts the job.
#
# The script caches per structure in <out-dir>/routing_nve_region_means.csv
# and reuses rows for structures already present, so the output directory
# must be new AND --recompute is passed.  --per-frame additionally forces a
# full evaluation and writes the per-frame weights, which is what decides
# whether a caption's error bars are the spread over the 101 frames
# (w_std_frames_*) or the spread across structures (the console table).
# ---------------------------------------------------------------------------
set -euo pipefail

source ~/.pilot_env
HETDIR="$(dirname "$PROJ")"
B="$HETDIR/moe-interface-potential"

# Batch jobs do not inherit conda, and NERSC's `module load conda` gives a
# different base with no silica_moe environment.
source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"
conda activate silica_moe

export MOE_HET="$HETDIR"
export OMP_NUM_THREADS=32
export MPLCONFIGDIR="$SCRATCH/.mplconfig"
mkdir -p "$MPLCONFIGDIR"

MD_ROOT="$PROJ/md_ep13_v1"
OUT="$PROJ/md_ep13_v1/figures_routing_ep13_v1"
TEST_CSV="$PROJ/data/expanded_holdout40_v1/test.csv"

COMMON="$REPAIR/pilot/common/best_checkpoint.pt"
ROUTER="$REPAIR/pilot/router/best_torchsoap_routing_gate.pt"
C1B="$REPAIR/pilot/c1b/best_c1b_residual_model.pt"
SOAP="$REPAIR/torchsoap_corrected_rotinv_v1.py"

echo "job     : ${SLURM_JOB_ID:-interactive} on $(hostname)"
echo "md-root : $MD_ROOT"
echo "out-dir : $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# ---- the four artifacts, verified against the pinned hashes ---------------
check () {   # $1 role   $2 path   $3 expected 24-hex prefix
    if [ ! -f "$2" ]; then
        echo "FAIL: missing $1 -> $2"
        exit 1
    fi
    h="$(sha256sum "$2" | cut -c1-24)"
    if [ "$h" != "$3" ]; then
        echo "FAIL: $1 sha256 $h, expected $3"
        exit 1
    fi
    printf "  ok  %-11s %s\n" "$1" "$h"
}
echo "artifact verification (selected model):"
check common     "$COMMON" 5b17849752a535739b16cfc8
check routing    "$ROUTER" 9e8e0ca6af4976ef8c74c34e
check c1b        "$C1B"    c7ad2a89c570549309ad8906
check descriptor "$SOAP"   0afd9a1ba7b9a24f8bf8151a
echo "  executed script $(sha256sum "$B/md/routing_from_md.py" | cut -c1-24)"
echo

# ---- refuse a stale per-structure cache ----------------------------------
if [ -f "$OUT/routing_nve_region_means.csv" ]; then
    echo "FAIL: $OUT already holds routing_nve_region_means.csv."
    echo "      Rows for structures already in it would be reused rather than"
    echo "      recomputed.  Point --out-dir at a new directory."
    exit 1
fi

N="$(find "$MD_ROOT" -maxdepth 1 -type d -name 'case_*' | wc -l)"
echo "case directories under md-root: $N"
if [ "$N" -lt 52 ]; then
    echo "FAIL: expected 52 finished cases, found $N"
    exit 1
fi
if [ ! -f "$TEST_CSV" ]; then
    echo "WARNING: $TEST_CSV absent; the three-region figure will be labelled"
    echo "         by record id instead of interface name."
fi

mkdir -p "$OUT"

srun -n 1 python3 "$B/md/routing_from_md.py" \
    --md-root  "$MD_ROOT" \
    --out-dir  "$OUT" \
    --prefix   LocalMoE_ep13_routing \
    --manifest "$MD_ROOT/md_manifest.csv" \
    --common-checkpoint  "$COMMON" \
    --routing-checkpoint "$ROUTER" \
    --c1b-checkpoint     "$C1B" \
    --torchsoap-module   "$SOAP" \
    --test-csv "$TEST_CSV" \
    --device cuda --dtype float64 \
    --per-frame --recompute --dpi 600

echo
echo "=== outputs ==="
for f in "$OUT"/*; do
    printf "  %-62s %s\n" "$(basename "$f")" "$(sha256sum "$f" | cut -c1-16)"
done
echo
echo "NOTE: figures and CSVs only.  Nothing is staged, committed or pushed."
