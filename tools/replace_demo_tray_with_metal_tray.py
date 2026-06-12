#!/usr/bin/env python3
"""Replace the dynamic-demo tray with the real metallic tray mesh."""


# ── SimForge path injection ───────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_HERE = str(_Path(__file__).resolve().parent)
_SIMFORGE = str(_Path(__file__).resolve().parents[1])
_CORE = str(_Path(__file__).resolve().parents[1] / "core")
for _p in (_HERE, _SIMFORGE, _CORE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom


DEFAULT_SOURCE_SCENE = "/home/andyee/isaacsim/playground/scene_final.usd"
DEFAULT_TARGET_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports/replace_demo_tray_with_metal_tray_report.json"
)


def bbox_for(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return None
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy"]
    ).ComputeWorldBound(prim).ComputeAlignedBox()
    return {
        "min": [float(v) for v in bbox.GetMin()],
        "max": [float(v) for v in bbox.GetMax()],
    }


def prim_summary(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return {"exists": False}
    return {
        "exists": True,
        "type": prim.GetTypeName(),
        "active": prim.IsActive(),
        "bbox": bbox_for(stage, path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene", default=DEFAULT_SOURCE_SCENE)
    parser.add_argument("--target-scene", default=DEFAULT_TARGET_SCENE)
    parser.add_argument("--source-path", default="/World/Tray")
    parser.add_argument("--target-path", default="/World/Tray")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    source_scene = Path(args.source_scene)
    target_scene = Path(args.target_scene)
    if not source_scene.exists():
        raise FileNotFoundError(source_scene)
    if not target_scene.exists():
        raise FileNotFoundError(target_scene)

    source_layer = Sdf.Layer.FindOrOpen(str(source_scene))
    target_layer = Sdf.Layer.FindOrOpen(str(target_scene))
    if source_layer is None:
        raise RuntimeError(f"Could not open source layer: {source_scene}")
    if target_layer is None:
        raise RuntimeError(f"Could not open target layer: {target_scene}")

    source_stage = Usd.Stage.Open(str(source_scene))
    if not source_stage.GetPrimAtPath(args.source_path):
        raise RuntimeError(f"Source tray not found: {args.source_path}")

    target_stage_before = Usd.Stage.Open(str(target_scene))
    before = {
        path: prim_summary(target_stage_before, path)
        for path in (
            "/World/Tray",
            "/World/Tray/Mesh",
            "/World/Tray_01",
            "/World/DemoTray",
            "/World/MAT_DemoTray",
        )
    }

    # Remove the demo tray and stale tray variants from the task scene, then copy
    # the real metallic tray from the older industrial scene.
    target_stage_edit = Usd.Stage.Open(str(target_scene))
    for path in ("/World/DemoTray", "/World/MAT_DemoTray", "/World/Tray_01", args.target_path):
        target_stage_edit.RemovePrim(path)
    target_stage_edit.GetRootLayer().Save()

    target_layer = Sdf.Layer.FindOrOpen(str(target_scene))
    if target_layer is None:
        raise RuntimeError(f"Could not reopen target layer: {target_scene}")

    ok = Sdf.CopySpec(
        source_layer,
        Sdf.Path(args.source_path),
        target_layer,
        Sdf.Path(args.target_path),
    )
    if not ok:
        raise RuntimeError(f"Failed to copy {args.source_path} to {args.target_path}")

    target_layer.Save()

    target_stage_after = Usd.Stage.Open(str(target_scene))
    after = {
        path: prim_summary(target_stage_after, path)
        for path in (
            "/World/Tray",
            "/World/Tray/Mesh",
            "/World/Tray/Mesh/Material",
            "/World/Tray/Mesh/Material/PBR",
            "/World/Tray_01",
            "/World/DemoTray",
            "/World/MAT_DemoTray",
            "/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim/RgbCamera",
            "/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim/RgbCamera",
        )
    }

    report = {
        "schema": "replace_demo_tray_with_metal_tray/v1",
        "source_scene": str(source_scene),
        "source_path": args.source_path,
        "target_scene": str(target_scene),
        "target_path": args.target_path,
        "removed_paths": ["/World/DemoTray", "/World/MAT_DemoTray", "/World/Tray_01"],
        "before": before,
        "after": after,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
