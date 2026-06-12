#!/usr/bin/env python3
"""Configure tray collision proxies and grasp frames for the two thin ears."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1] / "reports/tray_ear_collision_config_report.json"
)


TRAY_PATH = "/World/Tray"
TRAY_MESH_PATH = "/World/Tray/Mesh"
PROXY_ROOT = "/World/Tray/CollisionProxies"
GRASP_ROOT = "/World/Tray/GraspFrames"

# Mesh coordinates recovered from the real metallic tray. In the current tray
# pose, mesh local X maps to world Y, mesh local Y maps to world Z, and mesh
# local Z maps to world X. Therefore X min/max are the two grasp ears, and
# clamp_axis_local_mesh_y means top/bottom pad closure in world Z.
MESH_TRANSLATE = Gf.Vec3d(-0.06625744214471063, -0.31873307012154717, -0.08529542542626736)
LOCAL_REGIONS = {
    "Body": {
        "mesh_min": (0.0380, 0.0000, 0.0000),
        "mesh_max": (0.1415, 0.0300, 0.0995),
        "role": "tray main body; not a preferred grasp target",
    },
    "YMinusEar": {
        "mesh_min": (0.0000, 0.0000, 0.0000),
        "mesh_max": (0.0380, 0.0300, 0.0995),
        "role": "negative-world-Y grasp ear; first arm can clamp top/bottom here",
    },
    "YPlusEar": {
        "mesh_min": (0.1415, 0.0000, 0.0000),
        "mesh_max": (0.1795, 0.0300, 0.0995),
        "role": "positive-world-Y grasp ear; handoff arm can clamp top/bottom here",
    },
}


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


def disable_mesh_collision(stage: Usd.Stage):
    mesh = stage.GetPrimAtPath(TRAY_MESH_PATH)
    if not mesh:
        raise RuntimeError(f"Missing tray mesh: {TRAY_MESH_PATH}")
    if mesh.HasAPI(UsdPhysics.CollisionAPI):
        mesh.RemoveAPI(UsdPhysics.CollisionAPI)
    if mesh.HasAPI(UsdPhysics.MeshCollisionAPI):
        mesh.RemoveAPI(UsdPhysics.MeshCollisionAPI)


def define_cube_proxy(stage: Usd.Stage, name: str, mesh_min, mesh_max):
    min_v = Gf.Vec3d(*mesh_min)
    max_v = Gf.Vec3d(*mesh_max)
    center = MESH_TRANSLATE + (min_v + max_v) * 0.5
    size = max_v - min_v
    path = f"{PROXY_ROOT}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(center)
    xf.AddScaleOp().Set(size)
    apply_once(UsdPhysics.CollisionAPI, cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    return {
        "path": path,
        "local_center_xyz": [float(v) for v in center],
        "local_size_xyz": [float(v) for v in size],
        "mesh_min": [float(v) for v in min_v],
        "mesh_max": [float(v) for v in max_v],
    }


def define_grasp_frame(stage: Usd.Stage, name: str, mesh_min, mesh_max):
    min_v = Gf.Vec3d(*mesh_min)
    max_v = Gf.Vec3d(*mesh_max)
    center = MESH_TRANSLATE + (min_v + max_v) * 0.5
    frame = UsdGeom.Xform.Define(stage, f"{GRASP_ROOT}/{name}")
    xf = UsdGeom.Xformable(frame.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(center)
    return {
        "path": str(frame.GetPath()),
        "local_center_xyz": [float(v) for v in center],
        "semantic_axes": {
            "ear_span_axis": "tray mesh local X -> current world Y",
            "pad_clamp_axis": "tray mesh local Y -> current world Z; one pad above and one below the thin ear",
            "pad_width_axis": "tray mesh local Z -> current world X",
        },
    }


def bbox_for(stage: Usd.Stage, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return None
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"]
    ).ComputeWorldBound(prim).ComputeAlignedBox()
    return {
        "min": [float(v) for v in bbox.GetMin()],
        "max": [float(v) for v in bbox.GetMax()],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise RuntimeError(f"Could not open scene: {args.scene}")

    # Remove any older proxy/frame definitions before writing the current model.
    for path in (PROXY_ROOT, GRASP_ROOT):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)

    tray = stage.GetPrimAtPath(TRAY_PATH)
    if not tray:
        raise RuntimeError(f"Missing tray: {TRAY_PATH}")
    apply_once(UsdPhysics.RigidBodyAPI, tray).CreateRigidBodyEnabledAttr().Set(True)
    apply_once(UsdPhysics.MassAPI, tray).CreateMassAttr().Set(0.5)
    disable_mesh_collision(stage)

    UsdGeom.Xform.Define(stage, PROXY_ROOT)
    UsdGeom.Xform.Define(stage, GRASP_ROOT)
    proxies = {}
    grasp_frames = {}
    for name, spec in LOCAL_REGIONS.items():
        proxies[name] = define_cube_proxy(stage, name, spec["mesh_min"], spec["mesh_max"])
        if name != "Body":
            grasp_frames[name] = define_grasp_frame(stage, name, spec["mesh_min"], spec["mesh_max"])

    stage.GetRootLayer().Save()

    report = {
        "schema": "tray_ear_collision_config/v1",
        "scene": args.scene,
        "understanding": {
            "task": "The two arms should not exchange on the same ear. One arm grasps one Y-end ear; after handoff, the other arm grasps the opposite Y-end ear.",
            "grasp": "Each gripper uses its two pads to clamp one thin ear from above and below, so the pad closure axis is world Z in the current tray pose.",
        },
        "tray": {
            "path": TRAY_PATH,
            "visual_mesh_collision_enabled": False,
            "compound_collision_proxies": proxies,
            "grasp_frames": grasp_frames,
            "bbox": bbox_for(stage, TRAY_PATH),
        },
        "static_scene_collision_kept": {
            "table": "/World/Table/* cube colliders",
            "loading_equip": "/World/LoadingEquip/Mesh mesh collider",
            "dryer": "/World/Dryer/Mesh mesh collider",
        },
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
