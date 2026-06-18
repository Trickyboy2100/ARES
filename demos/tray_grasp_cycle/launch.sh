#!/usr/bin/env bash
# Tray grasp cycle demo launcher.
# Run from any directory — resolves its own location.
#
# Environment variables:
#   ISAACSIM_ROOT   Isaac Sim installation dir  (default: ~/isaacsim)
#   CUROBO_PYTHON   Python with cuRobo installed (auto-detected if unset)
#   CUROBO_ROOT     cuRobo source checkout (auto-detected if unset)
#   ROS_ROOT        ROS 2 install root for Isaac ROS bridge libs (default: /opt/ros/jazzy)
#   SIMFORGE_SCENE  Override scene USD path
#   SIMFORGE_URDF_DIR  Override robot URDF directory
#   TGC_LOG_FILE    Override log output path (default: logs/tray_grasp_cycle/<timestamp>.log)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMFORGE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # simforge/ repo root

if [[ -z "${ISAACSIM_ROOT:-}" ]]; then
    for candidate in \
        "$HOME/isaacsim" \
        "/media/$USER/SG/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"; do
        if [[ -x "$candidate/isaac-sim.sh" ]]; then
            ISAACSIM_ROOT="$candidate"
            break
        fi
    done
fi
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$HOME/isaacsim}"

if [[ ! -x "$ISAACSIM_ROOT/isaac-sim.sh" ]]; then
    echo "[launch] Isaac Sim launcher not found: $ISAACSIM_ROOT/isaac-sim.sh" >&2
    echo "[launch] Set ISAACSIM_ROOT=/path/to/isaacsim and rerun." >&2
    exit 1
fi

if [[ -z "${CUROBO_PYTHON:-}" ]]; then
    for candidate in \
        "/media/$USER/SG/conda_envs/curobo/bin/python3" \
        "$HOME/miniconda3/envs/curobo/bin/python3" \
        "$HOME/miniconda3/bin/python3" \
        "$(command -v python3 || true)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            CUROBO_PYTHON="$candidate"
            break
        fi
    done
fi
export CUROBO_PYTHON
export SIMFORGE_URDF_DIR="${SIMFORGE_URDF_DIR:-$SIMFORGE_ROOT/robot}"
export SIMFORGE_SCENE="${SIMFORGE_SCENE:-$SIMFORGE_ROOT/scenes/main.usd}"

if [[ -z "${CUROBO_ROOT:-}" ]]; then
    for candidate in \
        "/media/$USER/SG/curobo" \
        "$HOME/curobo"; do
        if [[ -f "$candidate/pyproject.toml" && -d "$candidate/curobo" ]]; then
            CUROBO_ROOT="$candidate"
            break
        fi
    done
fi

if [[ -n "${CUROBO_ROOT:-}" ]]; then
    export CUROBO_ROOT
fi

if [[ ! -f "$SIMFORGE_SCENE" ]]; then
    echo "[launch] Scene USD not found: $SIMFORGE_SCENE" >&2
    echo "[launch] Set SIMFORGE_SCENE=/path/to/scene.usd and rerun." >&2
    exit 1
fi

if ! "$CUROBO_PYTHON" -c 'import curobo' >/dev/null 2>&1; then
    echo "[launch] cuRobo is not importable with CUROBO_PYTHON=$CUROBO_PYTHON" >&2
    echo "[launch] CUROBO_ROOT=${CUROBO_ROOT:-unset}" >&2
    echo "[launch] Set CUROBO_PYTHON=/path/to/python-with-curobo, or install cuRobo dependencies into this Python." >&2
    exit 1
fi

ROS_ROOT="${ROS_ROOT:-/opt/ros/jazzy}"
if [[ -d "$ROS_ROOT/lib" ]]; then
    export ROS_ROOT
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ROS_ROOT/lib"
fi

pkill -f "isaacsim/kit/kit" 2>/dev/null || true
sleep 2

CUDALIB="$ISAACSIM_ROOT/exts/omni.isaac.ml_archive/pip_prebundle"
export LD_LIBRARY_PATH="$CUDALIB/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONPATH

LOG_DIR="$SIMFORGE_ROOT/logs/tray_grasp_cycle"
mkdir -p "$LOG_DIR"
if [[ -z "${TGC_LOG_FILE:-}" ]]; then
    TGC_LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
fi
export TGC_LOG_FILE
echo "[launch] Logging to $TGC_LOG_FILE"

exec > >(tee -a "$TGC_LOG_FILE") 2>&1

exec "$ISAACSIM_ROOT/isaac-sim.sh" \
    --exec "$SIMFORGE_ROOT/demos/tray_grasp_cycle/demo.py" \
    "$@"
