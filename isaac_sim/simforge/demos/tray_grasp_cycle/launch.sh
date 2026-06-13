#!/bin/bash
# Launch the tray grasp cycle demo.
# Run from anywhere — the script resolves its own location.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

# Kill any existing Isaac Sim instance
pkill -f "isaacsim/kit/kit" 2>/dev/null || true
sleep 2

# nvjitlink required to avoid ld.so assertion failure
CUDALIB=~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle
export LD_LIBRARY_PATH=$CUDALIB/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

cd "$REPO_ROOT"
exec ~/isaacsim/isaac-sim.sh --exec \
  "isaac_sim/simforge/demos/tray_grasp_cycle/demo.py" \
  "$@"
