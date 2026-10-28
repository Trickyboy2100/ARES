#!/usr/bin/env bash
# Fresh physical collection + matched training/evaluation. No hardware access.
set -euo pipefail
task_checkout="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
task_workspace="$(cd "$task_checkout/../.." && pwd)"
task_python="$task_workspace/envs/method-evolution-20261028/bin/python"
task_runs="${1:?Usage: handoff_reproduce.sh ABSOLUTE_NEW_RUN_DIRECTORY}"
[[ "$task_runs" = /* ]] || { echo 'Use an absolute run directory.' >&2; exit 2; }
[[ ! -e "$task_runs" ]] || { echo 'Refusing to overwrite existing evidence.' >&2; exit 2; }
mkdir -p "$task_runs"
launcher="$task_checkout/research/skill_handoff_lab/tools/handoff_offline.sh"
bash "$launcher" "$task_python" research/skill_handoff_lab/tools/handoff_vector.py --output "$task_runs/opportunity64" --seed 420000 --cases 64 --grid
bash "$launcher" "$task_python" research/skill_handoff_lab/tools/handoff_learn.py --root "$task_runs" --phase fit
bash "$launcher" "$task_python" research/skill_handoff_lab/tools/handoff_learn.py --root "$task_runs" --phase train
bash "$launcher" "$task_python" research/skill_handoff_lab/tools/handoff_learn.py --root "$task_runs" --phase train --ablate
bash "$launcher" "$task_python" research/skill_handoff_lab/tools/handoff_learn.py --root "$task_runs" --phase eval
bash "$launcher" "$task_python" research/skill_handoff_lab/tools/handoff_analyze.py --root "$task_runs" --output "$task_runs/analysis"
# Native repeatability is conditional on pinned drivers/physics/planner and GPU
# numerics. Report divergence; never replace it with the original outcomes.

bash "$launcher" "$task_python" research/skill_handoff_lab/tools/handoff_performance.py --root "$task_runs" --output "$task_runs/analysis"
