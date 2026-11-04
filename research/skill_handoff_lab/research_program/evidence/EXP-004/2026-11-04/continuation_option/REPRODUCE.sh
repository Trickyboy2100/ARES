#!/usr/bin/env bash
# Historical numeric reproduction: run from implementation commit758ab2c.
# Later empty-success runtime fix has a separate controller ID and validation.
# Run from the ARES-SCI-program repository. Uses new output directories only.
set -euo pipefail
workspace="$(cd ../.. && pwd)"
py="$workspace/envs/method-evolution-20261028/bin/python"
tools=research/skill_handoff_lab/tools
out="$workspace/runs/research_program/EXP-004/continuation_option_reproduction"
isolated=(bash "$tools/handoff_offline.sh" "$py")
"$py" "$tools/check_continuation_option.py"
"${isolated[@]}" "$tools/probe_continuation_option.py" --out "$out/pilot"
"${isolated[@]}" "$tools/probe_continuation_option.py" --out "$out/event_pilot" --event-only
"${isolated[@]}" "$tools/train_continuation_option.py" --out "$out/shared_train" --rounds 8 --dev-count 16
"${isolated[@]}" "$tools/eval_continuation_controls.py" --out "$out/controls"
"${isolated[@]}" "$tools/train_continuation_option.py" --out "$out/joint_train" --resume "$out/shared_train" --compare-joint --rounds 8 --dev-count 16
"${isolated[@]}" "$tools/train_continuation_option.py" --out "$out/smdp_train" --resume "$out/joint_train" --smdp-credit --rounds 8 --dev-count 16
"${isolated[@]}" "$tools/retake_continuation_training.py" --run "$out/shared_train" --out "$out/retakes"
"$py" "$tools/analyze_continuation_option.py" --root "$out"
# Parent checkpoint paths are inherited from previous success_retention and
# intervention_credit evidence. Restore those archives at their documented
# run locations before a clean-machine reproduction. No hardware is used.
