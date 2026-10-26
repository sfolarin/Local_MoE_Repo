#!/bin/bash
# pilot.sh -- bounded pilot on corrected features.  Features and the distillation
# dataset already exist from the smoke run; this trains the three stages to their
# normal stopping criteria and re-checks the physics.
#
# Epoch budgets are the trainers' own defaults.  Measured smoke rates: common
# 22 s/epoch, router 6.5 s/epoch, C1b 19 s/epoch -> about 2.83 GPU-h total.
# --max-hours on the common stage is a clean-stop guard, not the plan.
set -u
: "${PROJ:?}" ; : "${REPO:?}" ; : "${REPAIR:?}"
cd "$REPAIR"
PL="$REPAIR/pilot"; mkdir -p "$PL"
TIM="$PL/pilot_timings.json"
FEATC="$PROJ/data/expanded_holdout40_v1_features_rotinv_v1"
DISTC="$PROJ/data/expanded_holdout40_v1_distillation_rotinv_v1"
CORR="$REPAIR/torchsoap_corrected_rotinv_v1.py"
STAGE="$PROJ/data/expanded_holdout40_v1_c1b_stage"
echo "{" > "$TIM"
stage () {
  local n="$1"; shift
  echo ""; echo "################ $n ################"
  local t0=$(date +%s); "$@"; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '  "%s": {"seconds": %d, "exit": %d},\n' "$n" "$dt" "$rc" >> "$TIM"
  echo "[$n] ${dt}s, exit $rc"
  if [ $rc -ne 0 ] && [ "$rc" != "3" ]; then
    echo "ABORT: $n failed (exit $rc)"
    printf '  "aborted_at": "%s"\n}\n' "$n" >> "$TIM"; exit $rc
  fi
}

stage common_train \
  python3 "$PROJ/scripts/train_expanded_moe_common.py" train \
    --features "$FEATC" --out "$PL/common" \
    --energy-weight 10000 --force-weight 1000 --splits val \
    --resume --threads 16 --max-epochs 200 --max-hours 1.8

stage router_train \
  python3 run_adapted.py --adapter adapters/router_distill_adapter.py \
    --manifest "$DISTC/manifest.csv" --audit "$PL/audit_router.jsonl" -- \
    --dataset_root "$DISTC" --output_root "$PL/router" --device cuda \
    --seed 20260811 --torch_threads 16

stage c1b_train \
  python3 run_adapted.py --adapter adapters/c1b_holdout40_adapter.py \
    --manifest "$STAGE/manifest.csv" --audit "$PL/audit_c1b.jsonl" -- \
    --stage_b1_root "$STAGE" \
    --common_checkpoint "$PL/common/best_checkpoint.pt" \
    --routing_checkpoint "$PL/router/best_torchsoap_routing_gate.pt" \
    --torchsoap_module "$CORR" --output_root "$PL/c1b" --device cuda \
    --energy_weight 10000.0 --force_weight 1000.0 --seed 20260811

stage physics \
  env TRAINER="$REPAIR/adapters/c1b_holdout40_adapter.py" CORR="$CORR" \
      COMMON="$PL/common/best_checkpoint.pt" \
      ROUTER="$PL/router/best_torchsoap_routing_gate.pt" \
      RESIDUAL="$PL/c1b/best_c1b_residual_model.pt" \
      STAGE="$STAGE" DEVICE=cuda NREC=3 NPROBE=3 \
      JSON="$PL/physics_pilot.json" \
  python3 verify_assembled.py

printf '  "done": true\n}\n' >> "$TIM"
echo ""; echo "################ summary ################"
python3 -c "import json;d=json.load(open('$TIM'));[print('  %-16s %6ss exit %s'%(k,v['seconds'],v['exit'])) for k,v in d.items() if isinstance(v,dict)]"
for a in "$PL"/audit_*.jsonl; do [ -f "$a" ] || continue
  echo "  $(basename $a) test opens = $(python3 -c "
import json;print(sum(1 for l in open('$a') if l.strip() and json.loads(l).get('split')=='test'))")"
done
for f in "$PL"/common/best_checkpoint.pt "$PL"/router/best_torchsoap_routing_gate.pt \
         "$PL"/c1b/best_c1b_residual_model.pt; do
  [ -f "$f" ] && echo "  $(sha256sum $f)"
done
