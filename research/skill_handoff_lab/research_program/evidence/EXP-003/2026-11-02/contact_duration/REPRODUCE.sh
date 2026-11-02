#!/usr/bin/env bash
# Run from ARES root. Creates NEW outputs; never overwrite archived evidence.
set -euo pipefail
: "${ARES_PYTHON:?Set isolated ManiSkill Python path}"
: "${ARES_OUTPUT:?Set new, empty output root}"
T=research/skill_handoff_lab/tools
mkdir -p "$ARES_OUTPUT"
physics() { bash "$T/handoff_offline.sh" "$ARES_PYTHON" "$@"; }
physics "$T/program_tool_effect_probe.py" --out "$ARES_OUTPUT/tool_effect" --seeds 745000 745001 --masses .75 1.25 --grasps .035 .07 .105 --modes baseline refresh12
physics "$T/program_tool_effect_probe.py" --out "$ARES_OUTPUT/tool_controls" --seeds 745000 745001 --masses .75 1.25 --grasps .035 .07 .105 --modes frozen12 slow3
physics "$T/program_tool_effect_probe.py" --out "$ARES_OUTPUT/train108" --seeds 745010 745011 745012 745013 745014 745015 --masses .75 1.25 --grasps .035 .07 .105 --modes baseline frozen12 slow3
"$ARES_PYTHON" "$T/program_effect_value.py" --data "$ARES_OUTPUT/train108" --out "$ARES_OUTPUT/models"
physics "$T/program_effect_evaluate.py" --model "$ARES_OUTPUT/models/MODEL.pt" --data "$ARES_OUTPUT/train108" --out "$ARES_OUTPUT/evaluation"
physics "$T/program_tool_effect_probe.py" --out "$ARES_OUTPUT/retained100" --seeds 745020 745021 --masses .75 1.25 --grasps .1 --modes baseline frozen12 slow3
physics "$T/program_tool_all_contacts.py" --out "$ARES_OUTPUT/tool_all_contacts" --seeds 745000 --masses .75 --grasps .035 .07 --modes baseline frozen12 slow3 settle9
physics "$T/program_tool_endpoint_probe.py" --out "$ARES_OUTPUT/endpoint_fixed" --seeds 745000 --masses .75 --grasps .035 .07 --modes baseline frozen12 settle9 dense02
physics "$T/program_tool_endpoint_probe.py" --out "$ARES_OUTPUT/endpoint_dense" --seeds 745000 --masses .75 --grasps .035 .07 --modes dense005
"$ARES_PYTHON" "$T/program_effect_report.py" --root "$ARES_OUTPUT" --out "$ARES_OUTPUT/analysis"
physics "$T/program_effect_video.py" --left "$ARES_OUTPUT/tool_all_contacts/745000_m0.75_g0.035_baseline.json.gz" --right "$ARES_OUTPUT/tool_all_contacts/745000_m0.75_g0.035_frozen12.json.gz" --left-label 'Fixed transfer / failed' --right-label 'Frozen-attachment replan / passed' --out "$ARES_OUTPUT/video"
# Pyramid archived-source commands and original-state hashes are inside each archive.
# Do not reproduce the invalid FK logger attempts. No sealed data or hardware used.
