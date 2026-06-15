"""Minimal scene viewer — opens simforge/scenes/main.usd and holds the viewport.

Designed to run as an Isaac Sim --exec script:
    ~/isaacsim/isaac-sim.sh --exec isaac_sim/simforge/scenes/view_scene.py
"""

import sys
from pathlib import Path

_SCENES = Path(__file__).resolve().parent
SCENE_USD = str(_SCENES / "main.usd")

import omni.kit.app
import omni.usd

app = omni.kit.app.get_app()
ctx = omni.usd.get_context()

print(f"[view_scene] Loading: {SCENE_USD}", flush=True)
ctx.open_stage(SCENE_USD)

# wait up to 300 frames for the stage to be ready
for i in range(300):
    app.update()
    s = ctx.get_stage()
    if s and s.GetPrimAtPath("/World").IsValid():
        print(f"[view_scene] Stage ready ({i+1} frames)", flush=True)
        break
else:
    print("[view_scene] WARNING: /World prim not found after 300 frames", flush=True)

print("[view_scene] Viewport live — close the Isaac Sim window to exit.", flush=True)

# hold open
while app.is_running():
    app.update()
