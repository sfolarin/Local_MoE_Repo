#!/bin/bash
# Submit the five premium MD jobs (job index 0..4 of md_manifest.csv).
#   bash submit_md_jobs.sh            submit all five
#   bash submit_md_jobs.sh 0 3        submit only jobs 0 and 3
# Run the smoke test and the finite-difference check first (see the notes), and read md_manifest.csv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HET="${MOE_HET:?set_MOE_HET_to_the_data_workspace_folder_see_README}"
OUT_ROOT="${OUT_ROOT:-$HET/Local_MoE_MATPES_R2SCAN_Expanded_v1/md_holdout40_v1}"
mkdir -p "$OUT_ROOT/logs"
[[ -s "$OUT_ROOT/md_manifest.csv" ]] || { echo "no $OUT_ROOT/md_manifest.csv - run make_md_manifest.py first"; exit 1; }
JOBS=("$@"); [[ ${#JOBS[@]} -gt 0 ]] || JOBS=(0 1 2 3 4)
for j in "${JOBS[@]}"; do
  MD_JOB=$j OUT_ROOT="$OUT_ROOT" sbatch --job-name="lmoe_md_$j" --output="$OUT_ROOT/logs/slurm_job${j}_%j.out" \
        --export=ALL "$HERE/run_md_job.slurm"
done
