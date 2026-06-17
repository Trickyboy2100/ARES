#!/usr/bin/env python3
"""Audit and optionally add PhysX collision APIs for scene prims."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = os.environ.get("SIMFORGE_SCENE") or str(REPO_ROOT / "scenes" / "main.usd")
DEFAULT_STATIC_ROOTS = ["/World/Table", "/World/LoadingEquip", "/World/Dryer"]


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


def remove_dynamic_body(prim):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    if prim.HasAPI(UsdPhysics.MassAPI):
        prim.RemoveAPI(UsdPhysics.MassAPI)


def audit_root(stage, root_path: str, fix: bool, approximation: str):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return {"path": root_path, "status": "missing"}

    mesh_paths = []
    cube_paths = []
    collision_paths = []
    rigid_paths = []
    fixed_paths = []
    prims = [prim for prim in Usd.PrimRange(root) if prim.IsActive()]
    for prim in prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_paths.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            mesh_paths.append(str(prim.GetPath()))
            if fix:
                remove_dynamic_body(prim)
                apply_once(UsdPhysics.CollisionAPI, prim).CreateCollisionEnabledAttr().Set(True)
                if approximation != "none":
                    apply_once(UsdPhysics.MeshCollisionAPI, prim).CreateApproximationAttr().Set(approximation)
                fixed_paths.append(str(prim.GetPath()))
        elif prim.IsA(UsdGeom.Cube):
            cube_paths.append(str(prim.GetPath()))
            if fix:
                remove_dynamic_body(prim)
                apply_once(UsdPhysics.CollisionAPI, prim).CreateCollisionEnabledAttr().Set(True)
                fixed_paths.append(str(prim.GetPath()))

    if fix:
        collision_paths = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
        rigid_paths = [str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]

    return {
        "path": root_path,
        "status": "ok",
        "mesh_count": len(mesh_paths),
        "cube_count": len(cube_paths),
        "collision_api_count": len(collision_paths),
        "rigid_body_count": len(rigid_paths),
        "missing_collision_meshes": [p for p in mesh_paths if p not in set(collision_paths)],
        "fixed_count": len(fixed_paths),
        "fixed_paths": fixed_paths,
        "rigid_body_paths": rigid_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=os.environ.get("AUDIT_SCENE", DEFAULT_SCENE))
    parser.add_argument("--root", action="append", dest="roots", default=[])
    parser.add_argument("--fix", action="store_true", default=env_flag("AUDIT_FIX"))
    parser.add_argument(
        "--approximation",
        default=os.environ.get("AUDIT_APPROXIMATION", "none"),
        help="Static mesh approximation, usually none or convexDecomposition",
    )
    parser.add_argument(
        "--kit-exit",
        action="store_true",
        default=env_flag("AUDIT_KIT_EXIT"),
        help="Quit Isaac Kit after running under isaac-sim.sh --exec",
    )
    args, _unknown = parser.parse_known_args()

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise RuntimeError(f"Could not open scene: {args.scene}")

    roots = args.roots or DEFAULT_STATIC_ROOTS
    report = {
        "scene": args.scene,
        "fix": args.fix,
        "approximation": args.approximation,
        "roots": [audit_root(stage, root, args.fix, args.approximation) for root in roots],
    }

    if args.fix:
        stage.GetRootLayer().Save()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.kit_exit:
        try:
            import omni.kit.app

            omni.kit.app.get_app().post_quit()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    main()
