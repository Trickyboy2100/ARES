#!/usr/bin/env bash
set -euo pipefail
# From repository root. Extract all RAW_*.tar.gz into one evidence directory first.
parent="${1:?Path to extracted PARENT directory}"
run_out="${2:?New output directory (must not exist)}"
skill_workspace="$(cd ../.. && pwd)"
py="$skill_workspace/envs/method-evolution-20261028/bin/python"
runner=(bash research/skill_handoff_lab/tools/handoff_offline.sh "$py")
train=research/skill_handoff_lab/tools/train_success_retention.py
consolidate=research/skill_handoff_lab/tools/consolidate_success_retention.py
mkdir "$run_out"
"${runner[@]}" "$train" --parent "$parent" --out "$run_out/baseline" --baseline
for name in removed persistent controlled persistent_controlled; do
 args=();weight=0
 [[ "$name" != *persistent* ]] || { args+=(--success-bc .2);weight=.2; }
 [[ "$name" != *controlled* ]] || args+=(--exploration controlled)
 "${runner[@]}" "$train" --parent "$parent" --out "$run_out/$name" "${args[@]}"
 "${runner[@]}" "$consolidate" --parent "$run_out/$name" --out "$run_out/${name}_consolidated" --success-bc "$weight"
done
for replay in removed controlled; do
 for anchor in all_success nominal_success; do
  label=all;[[ "$anchor" != nominal_success ]] || label=nominal
  prefix=anchor;[[ "$replay" != controlled ]] || prefix=controlled_anchor
  "${runner[@]}" "$consolidate" --parent "$run_out/$replay" --out "$run_out/${prefix}_${label}" --success-bc .2 --anchor-mode "$anchor"
 done
done
"${runner[@]}" "$consolidate" --parent "$run_out/controlled" --out "$run_out/controlled_anchor_005" --success-bc .05 --anchor-mode nominal_success
"${runner[@]}" research/skill_handoff_lab/tools/eval_budget_feedback_interventions.py --model "$run_out/controlled_consolidated/FINAL.pt" --out "$run_out/strong_rule" --fixed-grasp 1 --modes budget_refine_zplus
"${runner[@]}" research/skill_handoff_lab/tools/eval_budget_feedback_interventions.py --model "$run_out/controlled_consolidated/FINAL.pt" --out "$run_out/verify" --fixed-grasp 1 --scenes 753204 753212 753209 --modes full zero_suffix budget_refine_zplus
"$py" research/skill_handoff_lab/tools/analyze_success_retention.py --root "$run_out" --out "$run_out/SUMMARY.json"
