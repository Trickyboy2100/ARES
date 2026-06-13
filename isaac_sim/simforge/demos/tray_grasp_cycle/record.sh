#!/usr/bin/env bash
# record.sh — launch tray_grasp_cycle demo with full log capture and report.
#
# Usage:
#   bash record.sh [--cycles N] [--label "tag"]
#
# Output:
#   isaac_sim/simforge/logs/tray_grasp_cycle/<TIMESTAMP>/
#       run.log        — full Isaac Sim stdout + stderr
#       summary.md     — auto-generated run report
#
# Environment:
#   The demo runs until the Isaac Sim window is closed (or Ctrl+C here).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
LOGS_BASE="$REPO_ROOT/isaac_sim/simforge/logs/tray_grasp_cycle"

LABEL="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)_CST"
RUN_DIR="$LOGS_BASE/${TIMESTAMP}_${LABEL}"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/run.log"
SUMMARY="$RUN_DIR/summary.md"

echo "============================================================"
echo "  Tray Grasp Cycle — Demo Recording"
echo "  Output: $RUN_DIR"
echo "============================================================"

# ── kill existing Isaac Sim ────────────────────────────────────────────────
echo "[record] Killing any existing Isaac Sim…"
pkill -f "isaacsim/kit/kit" 2>/dev/null || true
sleep 2

# ── env ────────────────────────────────────────────────────────────────────
CUDALIB=~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle
export LD_LIBRARY_PATH=$CUDALIB/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}

# ── launch ─────────────────────────────────────────────────────────────────
echo "[record] Starting demo at $(date)…"
START_TS=$(date +%s)

cd "$REPO_ROOT"
~/isaacsim/isaac-sim.sh --exec \
    "isaac_sim/simforge/demos/tray_grasp_cycle/demo.py" \
    2>&1 | tee "$LOG_FILE"

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

# ── parse log ──────────────────────────────────────────────────────────────
CYCLES_DONE=$(grep -c "→ PAUSE" "$LOG_FILE" 2>/dev/null || echo 0)
FORCE_STOPS=$(grep -c "FORCE STOP" "$LOG_FILE" 2>/dev/null || echo 0)
MAX_LIFT=$(grep "LIFT\|HOLD\|LOWER" "$LOG_FILE" | grep -oP "lift=\K[0-9.]+" | sort -n | tail -1)
MAX_FORCE=$(grep -oP "F=\K[0-9.]+" "$LOG_FILE" | sort -n | tail -1)
GRASP_POS=$(grep "tray_grasp_point world pos" "$LOG_FILE" | head -1 | grep -oP "\[.*\]" || echo "unknown")
IK_PRE_ERR=$(grep "IK pre" "$LOG_FILE" | head -1 | grep -oP "pos=\K[0-9.]+mm" || echo "?")
STAGE_READY=$(grep "Stage ready" "$LOG_FILE" | head -1 | grep -oP "\(\K[0-9]+ frames" || echo "?")
SETTLE_T=$(grep "settle 100" "$LOG_FILE" | head -1 | awk '{print $1}' || echo "?")

# ── generate summary ───────────────────────────────────────────────────────
cat > "$SUMMARY" << EOF
# Tray Grasp Cycle — Run Summary

| Field | Value |
|-------|-------|
| Timestamp | $(date -d "@$START_TS" "+%Y-%m-%d %H:%M:%S CST") |
| Label | $LABEL |
| Elapsed | ${ELAPSED}s |
| Cycles completed | $CYCLES_DONE |
| Force-stop events | $FORCE_STOPS |
| Max lift height | ${MAX_LIFT:-?} cm |
| Max friction force | ${MAX_FORCE:-?} N |
| Grasp prim pos | $GRASP_POS |
| IK pre pos_err | $IK_PRE_ERR |
| Stage ready | $STAGE_READY |

## Phase Transitions (cycle 1)
\`\`\`
$(grep "→ " "$LOG_FILE" | head -20 || echo "(none)")
\`\`\`

## IK Solutions
\`\`\`
$(grep "IK " "$LOG_FILE" | grep -v "IK done\|IK:" | head -10 || echo "(none)")
\`\`\`

## Force Stop Events
\`\`\`
$(grep "FORCE STOP" "$LOG_FILE" | head -10 || echo "(none)")
\`\`\`

## Tray Structure
\`\`\`
$(grep "Tray children\|Tray translate\|tray_grasp_point" "$LOG_FILE" | head -10 || echo "(none)")
\`\`\`

## Per-Cycle Peak Force (N) & Lift (cm)
\`\`\`
$(grep "HOLD\|LIFT" "$LOG_FILE" | grep "cy=[0-9]" | awk '{print}' | head -30 || echo "(none)")
\`\`\`

## Log
Full log: \`$(basename "$LOG_FILE")\`
EOF

echo ""
echo "============================================================"
echo "  Run complete."
echo "  Cycles:     $CYCLES_DONE"
echo "  Max lift:   ${MAX_LIFT:-?} cm"
echo "  Max force:  ${MAX_FORCE:-?} N"
echo "  Summary:    $SUMMARY"
echo "============================================================"
