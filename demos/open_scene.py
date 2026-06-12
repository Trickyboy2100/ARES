#!/usr/bin/env python3
"""Open the task USD in Isaac Sim GUI without running any motion playback."""

from __future__ import annotations

import argparse
import time


try:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import config as _cfg
    DEFAULT_SCENE = _cfg.SCENE_USD
except Exception:
    DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026061100_main.usd"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    return parser.parse_args()


def main():
    args = parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": 1440,
            "height": 900,
            "renderer": "RaytracedLighting",
        }
    )

    import omni.usd

    print(f"[STATIC-GUI] Opening scene: {args.scene}", flush=True)
    omni.usd.get_context().open_stage(args.scene)
    for _ in range(60):
        app.update()
        time.sleep(0.02)

    print("[STATIC-GUI] GUI open. No robot motion script is running. Ctrl-C to quit.", flush=True)
    try:
        while app.is_running():
            app.update()
            time.sleep(0.01)
    finally:
        app.close()


if __name__ == "__main__":
    main()
