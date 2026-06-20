#!/usr/bin/env bash
# Tray grasp cycle demo launcher.
# Run from any directory — resolves its own location.
#
# Environment variables:
#   ISAACSIM_ROOT   Isaac Sim installation dir  (default: ~/isaacsim)
#   CUROBO_PYTHON   Python with cuRobo installed (default: ~/miniconda3/bin/python3)
#   SIMFORGE_SCENE  Override scene USD path
#   SIMFORGE_URDF_DIR  Override robot URDF directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMFORGE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # simforge/ repo root
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim}"

pkill -f "isaacsim/kit/kit" 2>/dev/null || true
sleep 2

CUDALIB="$ISAACSIM_ROOT/exts/omni.isaac.ml_archive/pip_prebundle"
export LD_LIBRARY_PATH="$CUDALIB/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

exec "$ISAACSIM_ROOT/isaac-sim.sh" \
    --exec "$SIMFORGE_ROOT/demos/tray_grasp_cycle/demo.py" \
    "$@"
