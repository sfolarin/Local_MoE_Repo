#!/bin/bash
# finish_eval.sh -- resume the frozen static evaluation at stage 3.
#
# Stages 1 and 2 completed in the first attempt and are NOT repeated:
#   common_eval_and_export  43 s  exit 0   369/38/105 npz + metrics.csv
#   moe_export              14 s  exit 0   369/38/105 npz + per_configuration.csv (512)
# Re-running them would spend GPU time reproducing byte-identical output from frozen
# checkpoints, and the overwrite guard in final_eval.sh exists precisely to stop a
# partial redo from mixing two runs' predictions into one figure.
#
# Stage 3 uses the PATCHED comparison evaluator, whose only change is that a
# zero-byte or test-less reproduction file is treated as absent rather than crashing
# pandas.  See patch_compare.py for why that is a defect in the original.
#
#   source ~/.pilot_env && bash finish_eval.sh 2>&1 | tee finish_eval.txt
#
# Run inside:  salloc -A $ALLOC -q shared_interactive -C gpu -G 1 -t 0:25:00
set -u
: "${PROJ:?}" ; : "${REPO:?}" ; : "${REPAIR:?}"
export PROJ REPO REPAIR
cd "$REPAIR"

STAGE="$PROJ/data/expanded_holdout40_v1_c1b_stage"
CORR="$REPAIR/torchsoap_corrected_rotinv_v1.py"
CK_COMMON="$REPAIR/pilot/common/best_checkpoint.pt"
CK_ROUTER="$REPAIR/pilot/router/best_torchsoap_routing_gate.pt"
CK_RESID="$REPAIR/pilot/c1b/best_c1b_residual_model.pt"
CMP="$REPAIR/eval_compare_patched.py"

FIN="$REPAIR/final"
TIM="$FIN/finish_eval_timings.json"
echo "{" > "$TIM"

H_COMMON=5b17849752a535739b16cfc8e390e9dd53bbe27c8e50b975235142cf16e346e1
H_ROUTER=9e8e0ca6af4976ef8c74c34e1ded97b2aebeabb51c278037ab3ecab1145e1c66
H_RESID=c7ad2a89c570549309ad8906e477a42adad298b51251da5aa33a09c878ade451
H_DESC=0afd9a1ba7b9a24f8bf8151a7a3bdfc409544ed0920355de0598ffae79d51d21

verify_frozen () {
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
      printf '  %-26s %s  MISMATCH\n' "$(basename $f)" "${got:0:16}"; bad=1
    fi
  done
  return $bad
}

stage () {
  local n="$1"; shift
  echo ""; echo "################ $n ################"
  local t0=$(date +%s)
  "$@"; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '  "%s": {"seconds": %d, "exit": %d},\n' "$n" "$dt" "$rc" >> "$TIM"
  echo "[$n] ${dt}s, exit $rc"
  if [ $rc -ne 0 ]; then
    echo "ABORT: $n failed (exit $rc)."
    printf '  "aborted_at": "%s"\n}\n' "$n" >> "$TIM"
    exit $rc
  fi
  return 0
}

verify_frozen "before" || { echo "ABORT: a frozen artefact is not the authorised one"; exit 1; }

# ---- the stage 1 and 2 outputs this resume depends on must be intact ----------
echo ""; echo "---- inherited outputs ----"
python3 - <<'PY'
import os, sys
from pathlib import Path
R = Path(os.environ["REPAIR"])
want = {"train": 369, "val": 38, "test": 105}
ok = True
for label, root in (("common", R / "pilot/common/raw_predictions"),
                    ("moe",    R / "final/moe/raw_predictions")):
    for s, n in want.items():
        got = len(list((root / s).glob("*.npz"))) if (root / s).is_dir() else 0
        flag = "ok" if got == n else "WRONG"
        print("  %-7s %-5s %4d npz (want %4d)  %s" % (label, s, got, n, flag))
        ok = ok and got == n
m = R / "pilot/common/evaluation_train_val_test/metrics.csv"
pc = R / "final/moe/per_configuration.csv"
for p, lo in ((m, 2), (pc, 513)):
    n = len(p.read_text().splitlines()) if p.is_file() else 0
    flag = "ok" if n >= lo else "WRONG"
    print("  %-40s %5d lines  %s" % (p.name, n, flag))
    ok = ok and n >= lo
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "ABORT: inherited outputs are not intact; rerun final_eval.sh from scratch"; exit 1; }

if [ ! -f "$CMP" ]; then
  echo "ABORT: $CMP not found.  Run patch_compare.py first."
  exit 1
fi
echo ""
echo "patched evaluator: $(sha256sum "$CMP" | cut -c1-16)  $CMP"

for d in "$FIN/compare" "$FIN/figs_moe" "$FIN/figs_common"; do
  if [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ]; then
    echo "ABORT: $d exists and is not empty.  Move it aside or delete it."
    exit 1
  fi
done
mkdir -p "$FIN/compare" "$FIN/figs_moe" "$FIN/figs_common"

# ---- 3. common-only vs full Local-MoE on identical test configurations --------
stage common_vs_full \
  python3 "$CMP" \
    --c1b-script "$REPO/scripts/train_c1b_holdout40.py" \
    --c1b-checkpoint "$CK_RESID" \
    --common-checkpoint "$CK_COMMON" \
    --routing-checkpoint "$CK_ROUTER" \
    --torchsoap-module "$CORR" \
    --stage-root "$STAGE" \
    --common-eval-dir "$REPAIR/pilot/common/evaluation_train_val_test" \
    --device cuda:0 --expect-n-test 105 \
    --out-prefix "$FIN/compare/local_moe_vs_common"

# ---- 4. six-panel parity figures, published configuration ---------------------
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
echo ""; echo "################ verification ################"
python3 - <<'PY'
import csv, json, os
from pathlib import Path
import numpy as np

R = Path(os.environ["REPAIR"])
FIN = R / "final"

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

SPL = ("train", "val", "test")
moe = {s: metrics(FIN / "moe" / "raw_predictions", s) for s in SPL}
com = {s: metrics(R / "pilot" / "common" / "raw_predictions", s) for s in SPL}
ok = True

def table(title, d):
    print("\n%s" % title)
    print("  %-6s %4s %8s %9s %9s %10s %9s %9s"
          % ("split", "N", "atoms", "E_RMSE", "E_MAE", "E_bias", "F_SB", "F_pooled"))
    for s in SPL:
        m = d[s]
        print("  %-6s %4d %8d %9.4f %9.4f %10.4f %9.5f %9.5f"
              % (s, m["n"], m["atoms"], m["E_RMSE"], m["E_MAE"], m["E_bias"],
                 m["F_SB"], m["F_pooled"]))

table("Local-MoE  (energy meV/atom, force eV/A)", moe)
table("Common-network only", com)

if [moe[s]["n"] for s in SPL] != [369, 38, 105]:
    print("  FAIL: Local-MoE counts are not 369/38/105"); ok = False
if [com[s]["n"] for s in SPL] != [369, 38, 105]:
    print("  FAIL: common counts are not 369/38/105"); ok = False
for s in SPL:
    if moe[s]["atoms"] != com[s]["atoms"]:
        print("  FAIL: %s atom counts differ -- not identical configurations" % s)
        ok = False

print("\ncommon-only vs full Local-MoE, identical configurations:")
print("  %-6s %22s %22s" % ("split", "E_RMSE  common -> MoE", "F_SB  common -> MoE"))
for s in SPL:
    de = 100.0 * (moe[s]["E_RMSE"] - com[s]["E_RMSE"]) / com[s]["E_RMSE"]
    df = 100.0 * (moe[s]["F_SB"] - com[s]["F_SB"]) / com[s]["F_SB"]
    print("  %-6s   %8.4f -> %8.4f  (%+6.1f%%)   %7.5f -> %7.5f  (%+6.1f%%)"
          % (s, com[s]["E_RMSE"], moe[s]["E_RMSE"], de,
             com[s]["F_SB"], moe[s]["F_SB"], df))

for label, sub, d in (("Local-MoE", "figs_moe", moe), ("common", "figs_common", com)):
    stats = FIN / sub / "energy_force_parity_stats.csv"
    if not stats.is_file():
        print("\n  FAIL: no stats CSV for %s" % label); ok = False; continue
    worst = 0.0
    for r in csv.DictReader(open(stats)):
        m = d[r["split"]]
        for col, key in (("E_RMSE_meV_per_atom", "E_RMSE"), ("E_MAE_meV_per_atom", "E_MAE"),
                         ("E_bias_meV_per_atom", "E_bias"), ("F_SB_RMSE_eV_A", "F_SB"),
                         ("F_atom_weighted_RMSE_eV_A", "F_pooled")):
            worst = max(worst, abs(float(r[col]) - m[key]))
    print("\n  %-10s max |difference| vs its figure's stats CSV: %.3e" % (label, worst))
    ok = ok and worst < 1e-9

pilot = R / "pilot" / "c1b" / "final_split_metrics.csv"
print("\ncross-path check -- export (predict_energy_forces) vs pilot (evaluate_split):")
for r in csv.DictReader(open(pilot)):
    s = r["split"]
    if s not in moe:
        continue
    e_p = 1000.0 * float(r["energy_rmse_eV_per_atom"])
    f_p = float(r["force_structure_balanced_rmse_eV_A"])
    de, df = abs(e_p - moe[s]["E_RMSE"]), abs(f_p - moe[s]["F_SB"])
    flag = "ok" if (de < 5e-3 and df < 5e-5) else "DISAGREES"
    print("  %-6s E_RMSE %8.4f vs %8.4f (d=%.2e)   F_SB %8.5f vs %8.5f (d=%.2e)  %s"
          % (s, e_p, moe[s]["E_RMSE"], de, f_p, moe[s]["F_SB"], df, flag))
    ok = ok and flag == "ok"
print("  test   no prior value -- never evaluated before, by design")

(FIN / "verification.json").write_text(json.dumps(
    dict(local_moe=moe, common_only=com, verified=bool(ok)), indent=2))
print("\nVERIFICATION:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
PY
VRC=$?
printf '  "verification": {"exit": %d},\n' "$VRC" >> "$TIM"
[ $VRC -ne 0 ] && { echo "ABORT: verification failed"; printf '}\n' >> "$TIM"; exit $VRC; }

echo ""
verify_frozen "after" || { echo "ABORT: an artefact CHANGED during evaluation"; exit 1; }

printf '  "done": true\n}\n' >> "$TIM"
echo ""; echo "################ summary ################"
python3 -c "
import json;d=json.load(open('$TIM'))
[print('  %-22s %5ss  exit %s'%(k,v['seconds'],v['exit'])) for k,v in d.items() if isinstance(v,dict) and 'seconds' in v]"
echo ""
echo "per-configuration (MoE) : $FIN/moe/per_configuration.csv"
echo "per-partition metrics   : $FIN/figs_moe/energy_force_parity_stats.csv"
echo "six-panel figure        : $FIN/figs_moe/energy_force_parity.{pdf,png}"
echo "common-only figure      : $FIN/figs_common/energy_force_parity.{pdf,png}"
echo "common vs full (test)   : $FIN/compare/local_moe_vs_common_{per_structure,regions}.csv"
echo "verification            : $FIN/verification.json"
echo ""
echo "COMPLETE -- static evaluation and parity only.  No MD, no interface energy."
