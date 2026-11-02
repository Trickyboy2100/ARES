#!/usr/bin/env bash
set -euo pipefail
# Run from repository root; supply a NEW output directory.
output="${1:?new absolute output directory required}"
py="${ARES_PYTHON:-/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python}"
tools=research/skill_handoff_lab/tools
bash "$tools/handoff_offline.sh" "$py" "$tools/program_pyramid_contract_probe.py" --out "$output/pilot" --seeds 744000
bash "$tools/handoff_offline.sh" "$py" "$tools/program_pyramid_contract_probe.py" --symmetry-aware --out "$output/symmetry" --seeds 744000 744001
bash "$tools/handoff_offline.sh" "$py" "$tools/program_pyramid_contract_probe.py" --symmetry-aware --feasible-symmetry --out "$output/feasible" --seeds 744000 744001
"$py" "$tools/program_pyramid_contract_analyze.py" --root "$output" --out "$output/analysis"
