#!/usr/bin/env python3
"""Open the task USD in Isaac Sim GUI without running any motion playback.

Usage (--exec mode, recommended):
  CUDALIB=~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle
  LD_LIBRARY_PATH=$CUDALIB/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \\
    ~/isaacsim/isaac-sim.sh --exec isaac_sim/simforge/demos/open_scene.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import config as _cfg
    DEFAULT_SCENE = _cfg.SCENE_USD
except Exception:
    DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026061100_main.usd"


def main():
    import omni.kit.app
    import omni.usd

    # Accept an optional positional scene path passed after --exec <script>
    scene = DEFAULT_SCENE
    for arg in sys.argv[1:]:
        if arg.endswith(".usd") or arg.endswith(".usda"):
            scene = arg
            break

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    print(f"[SCENE] Opening: {scene}", flush=True)
    ctx.open_stage(scene)

    for i in range(300):
        app.update()
        s = ctx.get_stage()
        if s and s.GetPrimAtPath("/World").IsValid():
            print(f"[SCENE] Loaded ({i+1} frames). GUI is open — close window to quit.", flush=True)
            break

    while app.is_running():
        app.update()

    print("[SCENE] Window closed.", flush=True)


if __name__ == "__main__":
    main()
