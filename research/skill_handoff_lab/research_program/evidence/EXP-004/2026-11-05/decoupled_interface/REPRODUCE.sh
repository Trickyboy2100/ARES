#!/usr/bin/env bash
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
LAB=$(cd "$REPO/../.." && pwd)
PY="$LAB/envs/method-evolution-20261028/bin/python"
PKG="$REPO/research/skill_handoff_lab/research_program/evidence/EXP-004/2026-11-05/decoupled_interface"
T="$REPO/research/skill_handoff_lab/tools"
case "${1:-verify}" in
 verify)
  "$PY" "$T/verify_decoupled_archive.py" --package "$PKG"
  "$PY" "$REPO/research/skill_handoff_lab/research_program/scripts/program_status.py" check
  ;;
 train64)
  : "${2:?provide a NEW output directory}" "${3:?arm sac or success_bc}" "${4:?frame measured or remaining_target}"
  bash "$T/handoff_offline.sh" "$PY" "$T/train_option_online.py" --out "$2" --arm "$3" --frame "$4" --episodes 64 --train-start 759000 --dev-start 759200 --dev-count 16 --seed 31601 --updates 80
  ;;
 *) printf '%s\n' 'Usage: REPRODUCE.sh verify | train64 NEW_OUTPUT sac|success_bc measured|remaining_target' >&2; exit 2 ;;
esac
# For continuation use the recorded MANIFEST parameters plus --initialize-run.
# For interrupted runs use exact original arguments plus --resume; never overwrite.
# Restore historical parent assets first. No hardware services or SDK calls.
