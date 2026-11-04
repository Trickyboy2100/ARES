#!/usr/bin/env bash
set -euo pipefail
# Run at repository root; choose a NEW output root, never overwrite evidence.
: "${RUN_ROOT:?Set a fresh absolute output directory}"
: "${PARENT:?Set extracted success_retention controlled directory}"
PY=/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python
T=research/skill_handoff_lab/tools
run() { bash "$T/handoff_offline.sh" "$PY" "$@"; }
run "$T/collect_intervention_credit.py" --out "$RUN_ROOT/pairs" --count 8
run "$T/collect_intervention_credit.py" --out "$RUN_ROOT/late_pairs" --count 8 --phases 6
run "$T/collect_intervention_credit.py" --out "$RUN_ROOT/rule_pairs" --start 757008 --count 8 --phases 6 --continuation rule
"$PY" - "$RUN_ROOT" <<'PY'
from pathlib import Path
import json,sys
r=Path(sys.argv[1]);o=r/'zero_data';o.mkdir()
for k in ['PAIRS','TRANSITIONS','LABELS']:
 x=[]
 for name in ['pairs','late_pairs']:x+=json.loads((r/name/(k+'.json')).read_text())
 (o/(k+'.json')).write_text(json.dumps(x))
PY
run "$T/eval_budget_feedback_interventions.py" --model "$PARENT/FINAL.pt" --out "$RUN_ROOT/references" --start 757200 --count 24 --fixed-grasp 1 --modes zero_suffix budget_refine_zplus
for mode in sac awac measured_gap; do
 run "$T/train_intervention_credit.py" --out "$RUN_ROOT/$mode" --parent "$PARENT" --pairs "$RUN_ROOT/zero_data" --mode "$mode"
done
for mode in sac measured_gap awac measured_preference; do
 run "$T/train_online_intervention_credit.py" --out "$RUN_ROOT/current_$mode" --parent "$PARENT" --mode "$mode" --episodes 8
done
for mode in awac measured_preference; do
 run "$T/train_intervention_credit.py" --out "$RUN_ROOT/memory_$mode" --parent "$RUN_ROOT/current_measured_preference" --pairs "$RUN_ROOT/current_measured_preference" --mode "$mode" --episodes 0 --updates 1000
done
for mode in sac awac measured_gap measured_preference; do
 run "$T/eval_budget_feedback_interventions.py" --model "$RUN_ROOT/current_$mode/FINAL.pt" --out "$RUN_ROOT/retake_$mode" --fixed-grasp 1 --scenes 757300 757302 --modes full
done
run "$T/analyze_intervention_credit.py" --root "$RUN_ROOT"
run "$T/probe_intervention_credit.py" --root "$RUN_ROOT"
