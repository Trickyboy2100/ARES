#!/usr/bin/env bash
# checkpoint.sh — copy the current playground scene into scenes/ and git commit.
# After copying, rewrites all external USD paths to ../assets/ so the scene
# is self-contained for colleagues who clone the repo.
#
# Usage (from repo root):
#   bash scenes/checkpoint.sh "Scene description / commit message"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

SCENE_SRC="${SIMFORGE_SCENE:-${ISAACSIM_ROOT:-$HOME/isaacsim}/playground/2026061100_main.usd}"
SCENE_DST="$SCRIPT_DIR/main.usd"

msg="${1:-chore(scene): checkpoint $(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$SCENE_SRC" ]]; then
    echo "ERROR: source scene not found: $SCENE_SRC"
    echo "  Set SIMFORGE_SCENE or ISAACSIM_ROOT to override."
    exit 1
fi

echo "Copying scene: $SCENE_SRC → $SCENE_DST"
cp "$SCENE_SRC" "$SCENE_DST"

echo "Rewriting USD asset paths to ../assets/ ..."
USD_LIB=$(ls -d ~/isaacsim/extscache/omni.usd.libs-*/ 2>/dev/null | head -1)
if [[ -z "$USD_LIB" ]]; then
    echo "WARNING: omni.usd.libs not found — skipping path rewrite (scene may have broken references)"
else
    LD_LIBRARY_PATH="$USD_LIB/bin:${LD_LIBRARY_PATH:-}" \
        ~/isaacsim/kit/python/bin/python3 "$SCRIPT_DIR/rewrite_paths.py"
fi

echo "Staging changes..."
git -C "$REPO_ROOT" add "$SCENE_DST" "$SCRIPT_DIR/rewrite_paths.py"

echo "Committing: $msg"
git -C "$REPO_ROOT" commit -m "$msg"

echo "Done. Scene checkpointed and committed."
