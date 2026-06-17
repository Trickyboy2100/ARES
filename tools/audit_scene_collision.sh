#!/usr/bin/env bash
# Run the scene collision audit inside Isaac Kit so pxr/USD modules are available.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
    echo "[audit] Isaac Sim launcher not found: $ISAACSIM_ROOT/isaac-sim.sh" >&2
    echo "[audit] Set ISAACSIM_ROOT=/path/to/isaacsim and rerun." >&2
    exit 1
fi

export SIMFORGE_SCENE="${SIMFORGE_SCENE:-$REPO_ROOT/scenes/main.usd}"
export AUDIT_KIT_EXIT=1

ROOT_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fix)
            export AUDIT_FIX=1
            shift
            ;;
        --approximation)
            export AUDIT_APPROXIMATION="${2:?missing value for --approximation}"
            shift 2
            ;;
        --scene)
            export AUDIT_SCENE="${2:?missing value for --scene}"
            export SIMFORGE_SCENE="$AUDIT_SCENE"
            shift 2
            ;;
        --root)
            ROOT_ARGS+=("--root" "${2:?missing value for --root}")
            shift 2
            ;;
        *)
            echo "[audit] Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

unset PYTHONPATH

exec "$ISAACSIM_ROOT/isaac-sim.sh" \
    --no-window \
    --exec "$REPO_ROOT/tools/audit_scene_collision.py" \
    "${ROOT_ARGS[@]}"
