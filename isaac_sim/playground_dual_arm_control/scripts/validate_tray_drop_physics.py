#!/usr/bin/env python3
"""Headless PhysX validation for the metallic tray drop scene."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1] / "reports/tray_drop_physics_validation.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--tray-path", default="/World/Tray")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main():
    args = parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    ctx = omni.usd.get_context()
    print(f"[DROP-VALIDATE] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(120):
        app.update()
        time.sleep(0.002)

    stage = ctx.get_stage()
    tray = stage.GetPrimAtPath(args.tray_path)
    if not tray:
        raise RuntimeError(f"Tray not found: {args.tray_path}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.play()

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    samples = []
    steps = int(args.seconds / args.dt)
    for i in range(steps):
        app.update()
        if i % 15 == 0 or i == steps - 1:
            cache.Clear()
            mat = cache.GetLocalToWorldTransform(tray)
            xyz = [float(v) for v in mat.ExtractTranslation()]
            bbox = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), ["default", "render", "proxy"]
            ).ComputeWorldBound(tray).ComputeAlignedBox()
            samples.append(
                {
                    "time_sec": round(i * args.dt, 4),
                    "tray_world_translation_xyz": xyz,
                    "bbox_min_z": float(bbox.GetMin()[2]),
                    "bbox_max_z": float(bbox.GetMax()[2]),
                }
            )
    timeline.stop()

    final = samples[-1]
    report = {
        "schema": "tray_drop_physics_validation/v1",
        "scene": args.scene,
        "seconds": args.seconds,
        "tray_path": args.tray_path,
        "tray_has_rigid_body": tray.HasAPI(UsdPhysics.RigidBodyAPI),
        "tray_has_mass": tray.HasAPI(UsdPhysics.MassAPI),
        "samples": samples,
        "final": final,
        "expected_support_z_min": 0.77,
        "status": "pass" if final["bbox_min_z"] >= 0.74 else "fail",
        "note": "Pass threshold allows small settling/visual-pivot differences; inspect GUI for exact contact.",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    app.close()


if __name__ == "__main__":
    main()
