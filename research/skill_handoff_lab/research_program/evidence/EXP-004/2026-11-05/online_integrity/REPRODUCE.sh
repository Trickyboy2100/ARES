#!/usr/bin/env bash
set -euo pipefail
# Requires the inherited parent checkpoint and pinned local environment recorded
# in ENVIRONMENT.json. This reruns real native physics; it is not pose playback.
research_out="${1:?Pass a NEW absolute output directory}"
research_py=/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python
research_wrap=research/skill_handoff_lab/tools/handoff_offline.sh
research_tools=research/skill_handoff_lab/tools
mkdir "$research_out"
for research_arm in sac success_bc; do
 bash "$research_wrap" "$research_py" "$research_tools/train_option_online.py" \
  --out "$research_out/online_$research_arm" --arm "$research_arm" --episodes 64 \
  --train-start 759000 --dev-start 759200 --dev-count 16 --seed 31601 --updates 80
done
bash "$research_wrap" "$research_py" "$research_tools/probe_remaining_plan.py" --out "$research_out/remaining_plan"
bash "$research_wrap" "$research_py" "$research_tools/probe_remaining_plan.py" --out "$research_out/refresh_only" --refresh-only
bash "$research_wrap" "$research_py" "$research_tools/eval_online_controls.py" --out "$research_out/controls"
# Resume a run with the EXACT original arguments plus --resume. The trainer
# rejects incompatible seeds/config/source, and does not replay a completed job.
# Actual interrupted-run verification scripts and frozen source are in RAW.
