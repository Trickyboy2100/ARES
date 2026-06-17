#!/usr/bin/env bash
# Open the repository scene in Isaac Sim without running any demo logic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMFORGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
    echo "[open_scene] Isaac Sim launcher not found: $ISAACSIM_ROOT/isaac-sim.sh" >&2
    echo "[open_scene] Set ISAACSIM_ROOT=/path/to/isaacsim and rerun." >&2
    exit 1
fi

export SIMFORGE_SCENE="${SIMFORGE_SCENE:-$SIMFORGE_ROOT/scenes/main.usd}"
if [[ ! -f "$SIMFORGE_SCENE" ]]; then
    echo "[open_scene] Scene USD not found: $SIMFORGE_SCENE" >&2
    exit 1
fi

unset PYTHONPATH

exec "$ISAACSIM_ROOT/isaac-sim.sh" --exec "$SIMFORGE_ROOT/demos/open_scene.py" "$@"
