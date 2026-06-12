#!/usr/bin/env python3
"""Configure the task scene for a PhysX tray drop test.

Keeps one real metallic tray, removes duplicate/demo trays, and adds collision
to the tray plus the table, loading equipment, and dryer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1] / "reports/tray_drop_physics_setup_report.json"
)

STATIC_ROOTS = ("/World/Table", "/World/LoadingEquip", "/World/Dryer")
REMOVE_PATHS = ("/World/Tray_01", "/World/DemoTray", "/World/MAT_DemoTray")


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


def set_mesh_approximation(prim, approximation: str):
    mesh_api = apply_once(UsdPhysics.MeshCollisionAPI, prim)
    mesh_api.CreateApproximationAttr().Set(approximation)


def ensure_collision(prim, approximation: str | None = None):
    apply_once(UsdPhysics.CollisionAPI, prim)
    if approximation and prim.IsA(UsdGeom.Mesh):
        set_mesh_approximation(prim, approximation)


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


def collect_collision_targets(stage: Usd.Stage, root_path: str, mesh_approximation: str):
    root = stage.GetPrimAtPath(root_path)
    if not root:
        return []
    touched = []
    for prim in Usd.PrimRange(root):
        if not prim.IsActive():
            continue
        if prim.IsA(UsdGeom.Mesh):
            ensure_collision(prim, mesh_approximation)
            touched.append(str(prim.GetPath()))
        elif prim.IsA(UsdGeom.Cube):
            ensure_collision(prim)
            touched.append(str(prim.GetPath()))
    return touched


def reset_xform_to_drop_pose(stage: Usd.Stage, tray_path: str, drop_xyz):
    tray = stage.GetPrimAtPath(tray_path)
    if not tray:
        raise RuntimeError(f"Missing tray: {tray_path}")
    xf = UsdGeom.Xformable(tray)
    ops = xf.GetOrderedXformOps()
    translate_op = None
    for op in ops:
        if op.GetOpName() == "xformOp:translate":
            translate_op = op
            break
    if translate_op is None:
        xf.ClearXformOpOrder()
        translate_op = xf.AddTranslateOp()
        xf.AddRotateZOp().Set(90.0)
        xf.AddRotateYOp().Set(0.0)
        xf.AddRotateXOp().Set(90.0)
        xf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    translate_op.Set(Gf.Vec3d(*drop_xyz))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--tray-path", default="/World/Tray")
    parser.add_argument(
        "--tray-drop-xyz",
        type=float,
        nargs=3,
        default=[0.1133052415, 0.2869565601, 1.4440678416],
        help="World translation op for /World/Tray before pressing Play.",
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise RuntimeError(f"Could not open scene: {args.scene}")

    # Keep only the real metallic tray that planning will use next.
    for path in REMOVE_PATHS:
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)

    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    reset_xform_to_drop_pose(stage, args.tray_path, args.tray_drop_xyz)
    tray = stage.GetPrimAtPath(args.tray_path)
    if not tray:
        raise RuntimeError(f"Tray not found after cleanup: {args.tray_path}")
    rigid = apply_once(UsdPhysics.RigidBodyAPI, tray)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    mass_api = apply_once(UsdPhysics.MassAPI, tray)
    mass_api.CreateMassAttr().Set(0.5)

    tray_colliders = collect_collision_targets(stage, args.tray_path, "convexHull")
    static_colliders = {
        root: collect_collision_targets(stage, root, "none") for root in STATIC_ROOTS
    }

    stage.GetRootLayer().Save()

    report = {
        "schema": "tray_drop_physics_setup/v1",
        "scene": args.scene,
        "physics_scene": "/World/PhysicsScene",
        "removed_paths": [p for p in REMOVE_PATHS],
        "tray": {
            "path": args.tray_path,
            "drop_translate_xyz": args.tray_drop_xyz,
            "rigid_body": tray.HasAPI(UsdPhysics.RigidBodyAPI),
            "mass_kg": 0.5,
            "colliders": tray_colliders,
            "bbox": bbox_for(stage, args.tray_path),
        },
        "static_colliders": static_colliders,
        "static_bboxes": {root: bbox_for(stage, root) for root in STATIC_ROOTS},
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
