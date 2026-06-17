#!/usr/bin/env python3
"""Add static collision to mesh prims under /World/Dryer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = os.environ.get("SIMFORGE_SCENE") or str(REPO_ROOT / "scenes" / "main.usd")


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


def remove_dynamic_body(prim):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    if prim.HasAPI(UsdPhysics.MassAPI):
        prim.RemoveAPI(UsdPhysics.MassAPI)


def bbox_for(stage: Usd.Stage, root):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(root).ComputeAlignedBox()
    mn = Gf.Vec3d(box.GetMin())
    mx = Gf.Vec3d(box.GetMax())
    dims = mx - mn
    if max(float(dims[0]), float(dims[1]), float(dims[2])) <= 1e-6:
        raise RuntimeError(f"Could not compute a non-empty bbox for {root.GetPath()}")
    return (mn + mx) * 0.5, dims, mn, mx


def define_collision_cube(stage: Usd.Stage, path: str, center: Gf.Vec3d, dims: Gf.Vec3d):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    cube.CreatePurposeAttr().Set(UsdGeom.Tokens.proxy)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.SetResetXformStack(True)
    xf.AddTranslateOp().Set(center)
    xf.AddScaleOp().Set(dims)
    apply_once(UsdPhysics.CollisionAPI, cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    return {
        "path": path,
        "center_world_xyz": [float(center[0]), float(center[1]), float(center[2])],
        "dims_world_xyz": [float(dims[0]), float(dims[1]), float(dims[2])],
    }


def define_box_proxy(stage: Usd.Stage, root, name: str = "CollisionProxy"):
    center, dims, mn, mx = bbox_for(stage, root)
    path = f"{root.GetPath()}/{name}"
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    cube_info = define_collision_cube(stage, path, center, Gf.Vec3d(float(dims[0]), float(dims[1]), float(dims[2])))
    return {
        **cube_info,
        "bbox_min_world_xyz": [float(mn[0]), float(mn[1]), float(mn[2])],
        "bbox_max_world_xyz": [float(mx[0]), float(mx[1]), float(mx[2])],
    }


def define_hollow_shell(stage: Usd.Stage, root, name: str, open_axis: str, open_side: str, thickness: float):
    center, dims, mn, mx = bbox_for(stage, root)
    root_path = str(root.GetPath())
    shell_root = f"{root_path}/{name}"
    for stale in (f"{root_path}/CollisionProxy", shell_root):
        if stage.GetPrimAtPath(stale):
            stage.RemovePrim(stale)

    UsdGeom.Xform.Define(stage, shell_root).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
    t = max(float(thickness), min(dx, dy, dz) * 0.04)
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    minx, miny, minz = float(mn[0]), float(mn[1]), float(mn[2])
    maxx, maxy, maxz = float(mx[0]), float(mx[1]), float(mx[2])

    pieces = []

    def add(label, c, d):
        pieces.append(define_collision_cube(stage, f"{shell_root}/{label}", Gf.Vec3d(*c), Gf.Vec3d(*d)))

    if open_axis == "y":
        add("LeftWall_XMin", (minx + t / 2, cy, cz), (t, dy, dz))
        add("RightWall_XMax", (maxx - t / 2, cy, cz), (t, dy, dz))
        add("Bottom_ZMin", (cx, cy, minz + t / 2), (dx, dy, t))
        add("Top_ZMax", (cx, cy, maxz - t / 2), (dx, dy, t))
        back_y = miny + t / 2 if open_side == "plus" else maxy - t / 2
        add("BackWall_YMin" if open_side == "plus" else "BackWall_YMax", (cx, back_y, cz), (dx, t, dz))
    elif open_axis == "x":
        add("SideWall_YMin", (cx, miny + t / 2, cz), (dx, t, dz))
        add("SideWall_YMax", (cx, maxy - t / 2, cz), (dx, t, dz))
        add("Bottom_ZMin", (cx, cy, minz + t / 2), (dx, dy, t))
        add("Top_ZMax", (cx, cy, maxz - t / 2), (dx, dy, t))
        back_x = minx + t / 2 if open_side == "plus" else maxx - t / 2
        add("BackWall_XMin" if open_side == "plus" else "BackWall_XMax", (back_x, cy, cz), (t, dy, dz))
    else:
        raise ValueError("--open-axis must be x or y")

    return {
        "root": shell_root,
        "open_axis": open_axis,
        "open_side": open_side,
        "thickness_m": t,
        "bbox_min_world_xyz": [minx, miny, minz],
        "bbox_max_world_xyz": [maxx, maxy, maxz],
        "pieces": pieces,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--root", default="/World/Dryer")
    parser.add_argument("--approximation", default="convexHull")
    parser.add_argument("--proxy-mode", choices=["hollow", "box", "none"], default="none")
    parser.add_argument("--proxy-name", default="CollisionShell")
    parser.add_argument("--open-axis", choices=["x", "y"], default="y")
    parser.add_argument("--open-side", choices=["plus", "minus"], default="plus")
    parser.add_argument("--thickness", type=float, default=0.04)
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise RuntimeError(f"Could not open scene: {args.scene}")

    root = stage.GetPrimAtPath(args.root)
    if not root or not root.IsValid():
        raise RuntimeError(f"Missing dryer root: {args.root}")

    touched = []
    for prim in Usd.PrimRange(root):
        if not prim.IsActive():
            continue
        remove_dynamic_body(prim)
        if prim.IsA(UsdGeom.Mesh):
            apply_once(UsdPhysics.CollisionAPI, prim).CreateCollisionEnabledAttr().Set(True)
            apply_once(UsdPhysics.MeshCollisionAPI, prim).CreateApproximationAttr().Set(args.approximation)
            touched.append(str(prim.GetPath()))

    proxy = None
    if not touched and args.proxy_mode == "box":
        proxy = define_box_proxy(stage, root, args.proxy_name)
    elif not touched and args.proxy_mode == "hollow":
        proxy = define_hollow_shell(stage, root, args.proxy_name, args.open_axis, args.open_side, args.thickness)

    stage.GetRootLayer().Save()
    print(json.dumps({
        "scene": args.scene,
        "root": args.root,
        "approximation": args.approximation,
        "mesh_colliders": touched,
        "mesh_count": len(touched),
        "proxy_mode": args.proxy_mode,
        "proxy": proxy,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
