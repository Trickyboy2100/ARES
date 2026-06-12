#!/usr/bin/env bash
# checkpoint.sh — copy the current playground scene into scenes/ and git commit
#
# Usage:
#   ./checkpoint.sh "Scene description / commit message"
#
# After running, the scene at SCENE_SRC is copied to scenes/main.usd.
# You should then `git commit` (or this script will offer to).

set -euo pipefail

SCENE_SRC="/home/andyee/isaacsim/playground/2026061100_main.usd"
SCENE_DST="$(dirname "$0")/main.usd"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

msg="${1:-chore(scene): checkpoint $(date +%Y%m%d_%H%M%S)}"

echo "Copying scene from: $SCENE_SRC"
echo "             to:    $SCENE_DST"
cp "$SCENE_SRC" "$SCENE_DST"

echo "Staging scene file…"
git -C "$REPO_ROOT" add "$SCENE_DST"

echo "Committing: $msg"
git -C "$REPO_ROOT" commit -m "$msg"

echo "Done. Scene checkpointed and committed."
