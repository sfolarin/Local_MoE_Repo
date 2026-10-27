#!/bin/bash
# final_eval.sh -- FROZEN static evaluation of the Local-MoE checkpoints over the
# preserved 369 train / 38 val / 105 test split.  Produces the paper's energy and
# force prediction results.
#
# NO fitting, NO tuning, NO checkpoint selection.  Every network is loaded from a
# hash-pinned checkpoint and put in eval(); the descriptor has requires_grad_(False)
# on all parameters.  Normalisation statistics come from the checkpoints and the
# training contract, never recomputed from the evaluated data.
#
# THIS IS THE FIRST AUTHORISED TEST-PARTITION EVALUATION.  Every earlier run in this
# work excluded test by construction (adapter patches X1/X3, R1/R3).  Those adapters
# are deliberately NOT used here: this script drives the ORIGINAL, unpatched
# train_c1b_holdout40.py, whose load_records() returns all three splits.  The
# exclusion was the right default while checkpoints were still being selected; it
# would now be wrong, because the holdout is what the paper reports.
#
# Precision is the training evaluation precision by construction: float64 descriptor,
# float32 networks, N*e_ref accumulated in float64.  Nothing here casts or alters it.
#
#   source ~/.pilot_env && bash final_eval.sh 2>&1 | tee final_eval.txt
#
# Run inside:  salloc -A $ALLOC -q shared_interactive -C gpu -G 1 -t 0:40:00
set -u
: "${PROJ:?}" ; : "${REPO:?}" ; : "${REPAIR:?}"
# The verification block runs as a child python process and reads these from the
# environment, so they must be exported, not merely set in this shell.
export PROJ REPO REPAIR
cd "$REPAIR"

FEATC="$PROJ/data/expanded_holdout40_v1_features_rotinv_v1"
STAGE="$PROJ/data/expanded_holdout40_v1_c1b_stage"
CORR="$REPAIR/torchsoap_corrected_rotinv_v1.py"
CK_COMMON="$REPAIR/pilot/common/best_checkpoint.pt"
CK_ROUTER="$REPAIR/pilot/router/best_torchsoap_routing_gate.pt"
CK_RESID="$REPAIR/pilot/c1b/best_c1b_residual_model.pt"

FIN="$REPAIR/final"
TIM="$FIN/final_eval_timings.json"
mkdir -p "$FIN"
echo "{" > "$TIM"

# ---- the four frozen artefacts, exactly as authorised -----------------------
H_COMMON=5b17849752a535739b16cfc8e390e9dd53bbe27c8e50b975235142cf16e346e1
H_ROUTER=9e8e0ca6af4976ef8c74c34e1ded97b2aebeabb51c278037ab3ecab1145e1c66
H_RESID=c7ad2a89c570549309ad8906e477a42adad298b51251da5aa33a09c878ade451
H_DESC=0afd9a1ba7b9a24f8bf8151a7a3bdfc409544ed0920355de0598ffae79d51d21

verify_frozen () {          # verify_frozen <label>
  local bad=0
  echo "---- frozen-artefact verification ($1) ----"
  for pair in "$CK_COMMON:$H_COMMON" "$CK_ROUTER:$H_ROUTER" \
              "$CK_RESID:$H_RESID" "$CORR:$H_DESC"; do
    local f="${pair%:*}" want="${pair##*:}"
    if [ ! -f "$f" ]; then echo "  MISSING $f"; bad=1; continue; fi
    local got=$(sha256sum "$f" | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
      printf '  %-26s %s  ok\n' "$(basename $f)" "${got:0:16}"
    else
      printf '  %-26s %s  MISMATCH (want %s)\n' "$(basename $f)" "${got:0:16}" "${want:0:16}"
      bad=1
    fi
  done
  return $bad
}

stage () {                  # stage <name> <command...>
  local n="$1"; shift
  echo ""; echo "################ $n ################"
  local t0=$(date +%s)
  "$@"; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '  "%s": {"seconds": %d, "exit": %d},\n' "$n" "$dt" "$rc" >> "$TIM"
  echo "[$n] ${dt}s, exit $rc"
  if [ $rc -ne 0 ]; then
    echo "ABORT: $n failed (exit $rc).  Later stages would report numbers built on it."
    printf '  "aborted_at": "%s"\n}\n' "$n" >> "$TIM"
    exit $rc
  fi
  return 0
}

echo "features  $FEATC"
echo "stage     $STAGE"
echo "trainer   $REPO/scripts/train_c1b_holdout40.py   (ORIGINAL -- test included)"
echo ""

verify_frozen "before" || { echo "ABORT: a frozen artefact is not the authorised one"; exit 1; }

# Refuse to overwrite a previous evaluation: a half-replaced output directory is the
# one state that could mix predictions from two different runs into one figure.
for d in "$FIN/moe" "$FIN/compare" "$FIN/figs_moe" "$FIN/figs_common"; do
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
    echo "ABORT: $d exists and is not empty.  Move it aside or delete it."
    exit 1
  fi
done
mkdir -p "$FIN/moe" "$FIN/compare" "$FIN/figs_moe" "$FIN/figs_common"

# ---- 1. common-only: evaluator metrics AND raw predictions, all three splits ----
# One run: the wrapper monkeypatches evaluate_records so the trainer's own evaluate
# path additionally writes per-structure npz.  Settings (energy_weight 10000,
# force_weight 1000) are reloaded from training_contract.json, not from flags, so the
# evaluation cannot drift from the run that produced the checkpoint.
stage common_eval_and_export \
  python3 "$PROJ/export_common_parity_predictions_v2.py" \
    --common-script "$PROJ/scripts/train_expanded_moe_common.py" \
    --features "$FEATC" --out "$REPAIR/pilot/common" \
    --splits train val test --threads 16

# ---- 2. full Local-MoE per-configuration predictions, all 512 ------------------
stage moe_export \
  python3 "$PROJ/scripts/export_c1b_parity_predictions_holdout40.py" \
    --c1b-script "$REPO/scripts/train_c1b_holdout40.py" \
    --stage_b1_root "$STAGE" \
    --common_checkpoint "$CK_COMMON" \
    --routing_checkpoint "$CK_ROUTER" \
    --c1b_checkpoint "$CK_RESID" \
    --torchsoap_module "$CORR" \
    --residual-hidden-dims 256 128 \
    --out "$FIN/moe" --device cuda

# ---- 3. common-only vs full Local-MoE on identical test configurations ---------
stage common_vs_full \
  python3 "$PROJ/scripts/eval_test_regions_common_vs_full.py" \
    --c1b-script "$REPO/scripts/train_c1b_holdout40.py" \
    --c1b-checkpoint "$CK_RESID" \
    --common-checkpoint "$CK_COMMON" \
    --routing-checkpoint "$CK_ROUTER" \
    --torchsoap-module "$CORR" \
    --stage-root "$STAGE" \
    --common-eval-dir "$REPAIR/pilot/common/evaluation_train_val_test" \
    --device cuda:0 --expect-n-test 105 \
    --out-prefix "$FIN/compare/local_moe_vs_common"

# ---- 4. the six-panel parity figures, published configuration ------------------
# Default options reproduce the published layout exactly: verified by regenerating
# the Oct-18 figure from the original predictions and diffing its stats CSV to zero.
# No --zoom-row (so nrows=2 -> six panels), no --dedup-csv (the published stats show
# structures_used == records_in_files).  --model-label is "Local-MoE": no "corrected",
# no "pilot", as requested.
stage figure_moe \
  python3 "$PROJ/scripts/plot_energy_force_parity_holdout40.py" \
    --raw "$FIN/moe/raw_predictions" \
    --model-label "Local-MoE" \
    --out-dir "$FIN/figs_moe" --stem energy_force_parity

stage figure_common \
  python3 "$PROJ/scripts/plot_energy_force_parity_holdout40.py" \
    --raw "$REPAIR/pilot/common/raw_predictions" \
    --metrics "$REPAIR/pilot/common/evaluation_train_val_test/metrics.csv" \
    --model-label "Common-network" \
    --out-dir "$FIN/figs_common" --stem energy_force_parity

# ---- 5. independent verification ----------------------------------------------
# The export reaches the energy through predict_energy_forces; the pilot training run
# reached it through evaluate_split.  Two code paths, identical frozen weights.  If
# train and val disagree, the export is wrong and the test number must not be trusted.
echo ""; echo "################ verification ################"
python3 - <<'PY'
import csv, json, os
from pathlib import Path
import numpy as np

FIN = Path(os.environ["REPAIR"]) / "final"
REPAIR = Path(os.environ["REPAIR"])

# a. recompute every metric straight from the exported npz, independent of the plot script
def metrics(raw_root, split):
    de, fmse, na = [], [], []
    for f in sorted((raw_root / split).glob("*.npz")):
        with np.load(f, allow_pickle=False) as d:
            n = int(d["n_atoms"])
            de.append((float(d["E_predicted_eV"]) - float(d["E_reference_eV"])) / n)
            fr = np.asarray(d["reference_forces"], float)
            fp = np.asarray(d["predicted_forces"], float)
            fmse.append(float(np.mean((fp - fr) ** 2)))
            na.append(n)
    de, fmse, na = np.array(de), np.array(fmse), np.array(na)
    return dict(n=len(de), atoms=int(na.sum()),
                E_RMSE=1000 * float(np.sqrt(np.mean(de ** 2))),
                E_MAE=1000 * float(np.mean(np.abs(de))),
                E_bias=1000 * float(np.mean(de)),
                F_SB=float(np.sqrt(np.mean(fmse))),
                F_pooled=float(np.sqrt(np.average(fmse, weights=na))))

raw = FIN / "moe" / "raw_predictions"
mine = {s: metrics(raw, s) for s in ("train", "val", "test")}

print("independent recomputation from the exported npz (Local-MoE):")
print("  %-6s %4s %8s %9s %9s %10s %9s %9s" % ("split", "N", "atoms",
      "E_RMSE", "E_MAE", "E_bias", "F_SB", "F_pooled"))
for s, m in mine.items():
    print("  %-6s %4d %8d %9.4f %9.4f %10.4f %9.5f %9.5f"
          % (s, m["n"], m["atoms"], m["E_RMSE"], m["E_MAE"], m["E_bias"],
             m["F_SB"], m["F_pooled"]))

ok = True
if [mine[s]["n"] for s in ("train", "val", "test")] != [369, 38, 105]:
    print("  FAIL: configuration counts are not 369/38/105")
    ok = False

# b. agreement with the plot script's own stats CSV
stats = FIN / "figs_moe" / "energy_force_parity_stats.csv"
if stats.is_file():
    worst = 0.0
    for r in csv.DictReader(open(stats)):
        m = mine[r["split"]]
        for col, key in (("E_RMSE_meV_per_atom", "E_RMSE"), ("E_MAE_meV_per_atom", "E_MAE"),
                         ("E_bias_meV_per_atom", "E_bias"), ("F_SB_RMSE_eV_A", "F_SB"),
                         ("F_atom_weighted_RMSE_eV_A", "F_pooled")):
            worst = max(worst, abs(float(r[col]) - m[key]))
    print("\n  max |difference| vs the figure's stats CSV: %.3e" % worst)
    ok = ok and worst < 1e-9
else:
    print("\n  FAIL: no stats CSV from the figure"); ok = False

# c. cross-path check against the pilot's own training-time evaluation
pilot = REPAIR / "pilot" / "c1b" / "final_split_metrics.csv"
print("\n  cross-path check -- export (predict_energy_forces) vs pilot (evaluate_split):")
for r in csv.DictReader(open(pilot)):
    s = r["split"]
    if s not in mine:
        continue
    e_pilot = 1000.0 * float(r["energy_rmse_eV_per_atom"])
    f_pilot = float(r["force_structure_balanced_rmse_eV_A"])
    de = abs(e_pilot - mine[s]["E_RMSE"])
    df = abs(f_pilot - mine[s]["F_SB"])
    flag = "ok" if (de < 5e-3 and df < 5e-5) else "DISAGREES"
    print("    %-6s E_RMSE %8.4f vs %8.4f  (d=%.2e)   F_SB %8.5f vs %8.5f  (d=%.2e)  %s"
          % (s, e_pilot, mine[s]["E_RMSE"], de, f_pilot, mine[s]["F_SB"], df, flag))
    ok = ok and flag == "ok"
print("    test  has no prior value -- it was never evaluated before, by design")

(FIN / "verification.json").write_text(json.dumps(
    dict(local_moe=mine, verified=bool(ok)), indent=2))
print("\nVERIFICATION:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PY
VRC=$?
printf '  "verification": {"exit": %d},\n' "$VRC" >> "$TIM"
[ $VRC -ne 0 ] && { echo "ABORT: verification failed"; printf '}\n' >> "$TIM"; exit $VRC; }

# ---- 6. the frozen artefacts are still frozen ---------------------------------
echo ""
verify_frozen "after" || { echo "ABORT: an artefact CHANGED during evaluation"; exit 1; }

printf '  "done": true\n}\n' >> "$TIM"

echo ""; echo "################ summary ################"
python3 -c "
import json;d=json.load(open('$TIM'))
[print('  %-26s %5ss  exit %s'%(k,v['seconds'],v['exit'])) for k,v in d.items() if isinstance(v,dict) and 'seconds' in v]"
echo ""
echo "Local-MoE per-configuration : $FIN/moe/per_configuration.csv"
echo "Local-MoE raw npz           : $FIN/moe/raw_predictions/{train,val,test}"
echo "common-only raw npz         : $REPAIR/pilot/common/raw_predictions/{train,val,test}"
echo "common-only metrics         : $REPAIR/pilot/common/evaluation_train_val_test/metrics.csv"
echo "per-partition metrics       : $FIN/figs_moe/energy_force_parity_stats.csv"
echo "six-panel figure            : $FIN/figs_moe/energy_force_parity.{pdf,png}"
echo "common vs full (test)       : $FIN/compare/local_moe_vs_common_{per_structure,regions}.csv"
echo "verification                : $FIN/verification.json"
echo ""
echo "FINAL EVALUATION COMPLETE -- static evaluation and parity only."
echo "MD and interface-energy evaluations are NOT part of this run."
