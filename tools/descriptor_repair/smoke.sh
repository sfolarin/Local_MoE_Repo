#!/bin/bash
# smoke.sh -- genuine miniature end-to-end run: corrected feature generation,
# train-only normalization, short common/router/residual training, checkpoint
# reload, assembled-model physics.  Every training stage is instrumented for
# test access.  Per-stage wall times are recorded so the reduced pilot can be
# scheduled from measurement rather than estimate.
#
# Fails fast: any stage that does not succeed aborts the rest, because a later
# stage running on a failed earlier one produces numbers that look real.
#
#   source ~/.pilot_env && bash smoke.sh 2>&1 | tee smoke.txt
set -u
: "${PROJ:?}" ; : "${REPO:?}" ; : "${REPAIR:?}"
cd "$REPAIR"

EPOCHS="${EPOCHS:-2}"
SM="$REPAIR/smoke"
mkdir -p "$SM"
TIM="$SM/smoke_timings.json"
DS=$(python3 -c "import json;print(json.load(open('pf.json'))['dataset_root'])")
FEATC="$PROJ/data/expanded_holdout40_v1_features_rotinv_v1"
SNAPC="$REPAIR/src_snapshot_rotinv_v1"
CORR="$REPAIR/torchsoap_corrected_rotinv_v1.py"
STAGE="$PROJ/data/expanded_holdout40_v1_c1b_stage"

echo "{" > "$TIM"
stage () {            # stage <name> <command...>
  local n="$1"; shift
  echo ""; echo "################ $n ################"
  local t0=$(date +%s)
  "$@"; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '  "%s": {"seconds": %d, "exit": %d},\n' "$n" "$dt" "$rc" >> "$TIM"
  echo "[$n] ${dt}s, exit $rc"
  if [ $rc -ne 0 ] && [ "$rc" != "3" ]; then
    echo "ABORT: $n failed (exit $rc); later stages would run on a bad input"
    printf '  "aborted_at": "%s"\n}\n' "$n" >> "$TIM"
    exit $rc
  fi
  return 0
}

echo "dataset  $DS"
echo "features $FEATC"
echo "snapshot $SNAPC"
echo "epochs   $EPOCHS   (smoke: interface exercise, NOT an accuracy result)"

# ---- 1. corrected feature generation, all 512 records ----------------------
# --allow-test is used HERE ONLY, under the accepted scope exception: a fixed
# geometry-to-feature transform over all records preserves the pipeline's
# membership-integrity checks.  Every file opened is still recorded.
stage features_compute \
  python3 run_adapted.py --adapter adapters/prep_features_adapter.py \
    --manifest "$DS/manifest.csv" --audit "$SM/audit_features.jsonl" --allow-test -- \
    compute --dataset "$DS" --snapshot "$SNAPC" --out "$FEATC" \
    --rank 0 --world 1 --threads 16

stage features_finalize \
  python3 run_adapted.py --adapter adapters/prep_features_adapter.py \
    --manifest "$DS/manifest.csv" --audit "$SM/audit_features.jsonl" --allow-test -- \
    finalize --dataset "$DS" --snapshot "$SNAPC" --out "$FEATC" --threads 16

# ---- 2. train-only normalization, verified -------------------------------
echo ""; echo "################ normalization check ################"
python3 - <<PY
import json, numpy as np, hashlib
from pathlib import Path
F = Path("$FEATC")
s = json.load(open(F / "feature_summary.json"))
print("finalization_status  :", s.get("finalization_status"))
print("training_moments_ready:", s.get("training_moments_ready"))
print("feature_counts       :", s.get("feature_counts"))
ok = s.get("finalization_status") == "PASS" and s.get("training_moments_ready")
with np.load(F / "training_feature_moments.npz", allow_pickle=False) as d:
    ids = d["train_record_ids"]
    print("train_record_ids     :", len(ids), "(must be 369: TRAIN ONLY)")
    print("mean shape           :", d["mean"].shape)
    ok = ok and len(ids) == 369
m = F / "src_snapshot" / "torchsoap_autograd_module_periodic_regnorm.py"
h = hashlib.sha256(m.read_bytes()).hexdigest() if m.exists() else None
print("features src_snapshot descriptor:", h)
ok = ok and h == "0afd9a1ba7b9a24f8bf8151a7a3bdfc409544ed0920355de0598ffae79d51d21"
print("RESULT:", "normalization is train-only and the descriptor is corrected" if ok
      else "FAIL")
raise SystemExit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "ABORT: normalization/descriptor check failed"; exit 1; }

# ---- 3. short common training ---------------------------------------------
# --splits val keeps the test partition closed; its default is ["val","test"].
stage common_train \
  python3 "$PROJ/scripts/train_expanded_moe_common.py" train \
    --features "$FEATC" --out "$SM/common" \
    --energy-weight 10000 --force-weight 1000 --splits val \
    --resume --threads 16 --max-epochs "$EPOCHS"

# ---- 4. distillation dataset from the corrected features ------------------
DISTC="$PROJ/data/expanded_holdout40_v1_distillation_rotinv_v1"
stage distill_dataset \
  python3 "$REPO/scripts/prepare_expanded_torchsoap_distillation_dataset.py" \
    --features "$FEATC" \
    --teachers "$PROJ/data/expanded_holdout40_v1_teachers" \
    --routing-dataset "$PROJ/data/expanded_holdout40_v1_routing" \
    --out "$DISTC"

# ---- 5. short router distillation ----------------------------------------
stage router_train \
  python3 run_adapted.py --adapter adapters/router_distill_adapter.py \
    --manifest "$DISTC/manifest.csv" --audit "$SM/audit_router.jsonl" -- \
    --dataset_root "$DISTC" --output_root "$SM/router" --device cuda \
    --seed 20260811 --torch_threads 16 --max_epochs "$EPOCHS"

# ---- 6. short residual training ------------------------------------------
stage c1b_train \
  python3 run_adapted.py --adapter adapters/c1b_holdout40_adapter.py \
    --manifest "$STAGE/manifest.csv" --audit "$SM/audit_c1b.jsonl" -- \
    --stage_b1_root "$STAGE" \
    --common_checkpoint "$SM/common/best_checkpoint.pt" \
    --routing_checkpoint "$SM/router/best_torchsoap_routing_gate.pt" \
    --torchsoap_module "$CORR" --output_root "$SM/c1b" --device cuda \
    --energy_weight 10000.0 --force_weight 1000.0 --seed 20260811 \
    --max_epochs "$EPOCHS"

# ---- 7. checkpoint reload -------------------------------------------------
echo ""; echo "################ checkpoint reload ################"
python3 - <<PY
import hashlib, torch
from pathlib import Path
ok = True
for lbl, p in (("common",   "$SM/common/best_checkpoint.pt"),
               ("router",   "$SM/router/best_torchsoap_routing_gate.pt"),
               ("residual", "$SM/c1b/best_c1b_residual_model.pt")):
    f = Path(p)
    if not f.is_file():
        print("%-9s MISSING %s" % (lbl, p)); ok = False; continue
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    d = torch.load(f, map_location="cpu", weights_only=False)
    n = sum(1 for k in d if "state_dict" in k)
    print("%-9s sha256 %s  reloaded, %d state dict(s), %d keys" % (lbl, h[:16], n, len(d)))
raise SystemExit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "ABORT: a smoke checkpoint did not reload"; exit 1; }

# ---- 8. assembled-model physics on the SMOKE checkpoints ------------------
stage physics \
  env TRAINER="$REPAIR/adapters/c1b_holdout40_adapter.py" CORR="$CORR" \
      COMMON="$SM/common/best_checkpoint.pt" \
      ROUTER="$SM/router/best_torchsoap_routing_gate.pt" \
      RESIDUAL="$SM/c1b/best_c1b_residual_model.pt" \
      STAGE="$STAGE" DEVICE=cuda NREC=2 NPROBE=3 \
      JSON="$SM/physics_smoke.json" \
  python3 verify_assembled.py

printf '  "done": true\n}\n' >> "$TIM"
echo ""; echo "################ summary ################"
python3 -c "import json;d=json.load(open('$TIM'));[print('  %-20s %5ss  exit %s'%(k,v['seconds'],v['exit'])) for k,v in d.items() if isinstance(v,dict)]"
echo ""
for a in "$SM"/audit_*.jsonl; do
  [ -f "$a" ] || continue
  n=$(python3 -c "
import json,sys
t=sum(1 for l in open('$a') if l.strip() and json.loads(l).get('split')=='test')
print(t)")
  echo "  $(basename $a): test-partition opens = $n"
done
echo ""
echo "SMOKE COMPLETE -- interface exercise only.  $EPOCHS epochs per stage is not"
echo "an accuracy result and must not be reported as one."
