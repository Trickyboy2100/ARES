#!/usr/bin/env python3
"""USD scene utility helpers for Isaac Sim simulation scripts.

Covers:
  - Bounding-box queries (bbox_payload, bbox_center)
  - World-frame transform lookups (xform_world_xyz)
  - USD ↔ NumPy matrix conversion (gf_matrix_from_column_transform)
  - Physics material creation and assignment
  - Hidden kinematic carrier management (ensure_hidden_carrier, set_carrier)
  - FixedJoint creation for grasp locking (create_grasp_lock)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ── Bounding box ──────────────────────────────────────────────────────────────

def bbox_payload(stage, path: str) -> dict:
    """Return {'min': [x,y,z], 'max': [x,y,z]} world-axis-aligned bbox of a prim."""
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    box = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        .ComputeWorldBound(prim)
        .ComputeAlignedBox()
    )
    return {"min": [float(v) for v in box.GetMin()], "max": [float(v) for v in box.GetMax()]}


def bbox_center(b: dict) -> np.ndarray:
    """Return centre of a bbox_payload dict as a (3,) float64 array."""
    return np.array([(b["min"][i] + b["max"][i]) * 0.5 for i in range(3)], dtype=float)


# ── World-frame transform ─────────────────────────────────────────────────────

def xform_world_xyz(stage, path: str) -> np.ndarray:
    """Return world-frame translation (x,y,z) of a prim as (3,) float64."""
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Missing prim: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation(), dtype=float)


# ── Matrix conversion ─────────────────────────────────────────────────────────

def gf_matrix_from_column_transform(T) -> object:
    """Convert a (4,4) column-major NumPy array to a pxr.Gf.Matrix4d."""
    from pxr import Gf
    return Gf.Matrix4d(*sum(np.asarray(T, dtype=float).T.tolist(), []))


def apply_once(api_cls, prim):
    """Apply a USD API schema to a prim if not already applied. Returns the API."""
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


# ── Physics materials ─────────────────────────────────────────────────────────

def ensure_physics_material(
    stage, mat_path: str, static: float, dynamic: float, restitution: float = 0.0
):
    """Create a UsdPhysics.MaterialAPI at mat_path if it doesn't already exist."""
    from pxr import UsdPhysics, UsdShade
    if stage.GetPrimAtPath(mat_path) and stage.GetPrimAtPath(mat_path).IsValid():
        return UsdShade.Material(stage.GetPrimAtPath(mat_path))
    mat = UsdShade.Material.Define(stage, mat_path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(static)
    pm.CreateDynamicFrictionAttr().Set(dynamic)
    pm.CreateRestitutionAttr().Set(restitution)
    return mat


def apply_friction_to_collision_prims(
    stage, root_path: str, mat_path: str, static: float, dynamic: float
):
    """Apply a physics friction material to every CollisionAPI prim under root_path."""
    from pxr import Usd, UsdPhysics, UsdShade
    mat = ensure_physics_material(stage, mat_path, static, dynamic)
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )


# ── Hidden kinematic carrier ──────────────────────────────────────────────────

def ensure_hidden_carrier(stage, probe_root: str) -> tuple:
    """Create (or recreate) a hidden kinematic Cube at probe_root/Carrier.

    Returns (carrier_prim, translate_op).  The caller moves the carrier by
    calling carrier_op.Set(Gf.Vec3d(x, y, z)) each simulation frame.

    The carrier acts as the physics anchor for the grasp FixedJoint.
    """
    from pxr import UsdGeom, UsdPhysics

    if stage.GetPrimAtPath(probe_root):
        stage.RemovePrim(probe_root)
    UsdGeom.Xform.Define(stage, probe_root)
    cube = UsdGeom.Cube.Define(stage, f"{probe_root}/Carrier")
    prim = cube.GetPrim()
    cube.CreateSizeAttr(0.01)
    cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rb = apply_once(UsdPhysics.RigidBodyAPI, prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    rb.CreateKinematicEnabledAttr().Set(True)
    apply_once(UsdPhysics.MassAPI, prim).CreateMassAttr().Set(0.05)
    return prim, op


def set_carrier(op, xyz: np.ndarray):
    """Move the kinematic carrier to a world position each frame."""
    from pxr import Gf
    op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


# ── Grasp lock (FixedJoint) ───────────────────────────────────────────────────

def create_grasp_lock(stage, tray_path: str, carrier_path: str, joint_path: str) -> str:
    """Create a FixedJoint between carrier and tray at joint_path.

    MUST only be called after physical contact is confirmed (force_stop_step
    is not None) — per CLAUDE.md design constraint (no fake grasp).
    Returns the joint prim path string.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    if stage.GetPrimAtPath(joint_path):
        return joint_path

    tray    = stage.GetPrimAtPath(tray_path)
    carrier = stage.GetPrimAtPath(carrier_path)
    cache   = UsdGeom.XformCache(Usd.TimeCode.Default())
    tray_w    = cache.GetLocalToWorldTransform(tray)
    carrier_w = cache.GetLocalToWorldTransform(carrier)
    anchor  = carrier_w.ExtractTranslation()
    local0  = carrier_w.GetInverse().Transform(anchor)
    local1  = tray_w.GetInverse().Transform(anchor)

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(carrier_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tray_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local1]))
    tray_rot_inv = tray_w.ExtractRotationQuat().GetInverse().GetNormalized()
    tri = tray_rot_inv.GetImaginary()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(float(tray_rot_inv.GetReal()),
                 float(tri[0]), float(tri[1]), float(tri[2]))
    )
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint_path
