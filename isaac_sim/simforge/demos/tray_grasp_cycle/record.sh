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
ISAAC_NVLIBS=$(find ~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia -type d -name "lib" | tr '\n' ':')
export LD_LIBRARY_PATH="${ISAAC_NVLIBS}${LD_LIBRARY_PATH:-}"

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
CYCLES_DONE=$(grep -c "\[TGC\] → PAUSE" "$LOG_FILE" 2>/dev/null || echo 0)
FORCE_STOPS=$(grep -c "FORCE STOP" "$LOG_FILE" 2>/dev/null || echo 0)
HANDOFFS=$(grep -c "L FixedJoint removed → tray held by R" "$LOG_FILE" 2>/dev/null || echo 0)
MAX_FORCE=$(grep -oP "L FORCE STOP F=\K[0-9.]+" "$LOG_FILE" | sort -n | tail -1)
GRASP_POS=$(grep "Tray settled:" "$LOG_FILE" | head -1 | grep -oP "\[\K[^\]]*" || echo "unknown")
IK_PRE_ERR=$(grep "IK L_pre" "$LOG_FILE" | head -1 | grep -oP "pos_err=\K[0-9.]+mm" || echo "?")
RESET_COUNT=$(grep -c "Reset done → tray free" "$LOG_FILE" 2>/dev/null || echo 0)

# ── generate summary ───────────────────────────────────────────────────────
cat > "$SUMMARY" << EOF
# Tray Grasp Cycle v3 — Run Summary

| Field | Value |
|-------|-------|
| Timestamp | $(date -d "@$START_TS" "+%Y-%m-%d %H:%M:%S CST") |
| Label | $LABEL |
| Elapsed | ${ELAPSED}s |
| Cycles completed | $CYCLES_DONE |
| Successful handoffs | ${HANDOFFS:-0} |
| Force-stop events | $FORCE_STOPS |
| Scene resets | ${RESET_COUNT:-0} |
| Max L-arm force | ${MAX_FORCE:-?} N |
| Tray settled pos | $GRASP_POS |
| IK L_pre pos_err | ${IK_PRE_ERR:-?} |

## Phase Transitions (first 30)
\`\`\`
$(grep "\[TGC\] → " "$LOG_FILE" | head -30 || echo "(none)")
\`\`\`

## IK Solutions
\`\`\`
$(grep "\[TGC\] IK " "$LOG_FILE" | grep -v "planning\|done" | head -15 || echo "(none)")
\`\`\`

## Force Stop Events
\`\`\`
$(grep "FORCE STOP" "$LOG_FILE" | head -10 || echo "(none)")
\`\`\`

## Handoff Events
\`\`\`
$(grep "FixedJoint removed\|Reset carrier\|Reset done\|cuRobo" "$LOG_FILE" | head -15 || echo "(none)")
\`\`\`

## Errors / Warnings
\`\`\`
$(grep -iE "ERROR|WARNING|warn|exception|traceback" "$LOG_FILE" | grep "\[TGC\]" | head -15 || echo "(none)")
\`\`\`

## Log
Full log: \`$(basename "$LOG_FILE")\`
EOF

echo ""
echo "============================================================"
echo "  Run complete."
echo "  Cycles:     $CYCLES_DONE"
echo "  Handoffs:   ${HANDOFFS:-0}"
echo "  Max force:  ${MAX_FORCE:-?} N"
echo "  Summary:    $SUMMARY"
echo "============================================================"
