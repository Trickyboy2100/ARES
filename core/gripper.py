#!/usr/bin/env python3
"""EG2-4C2 gripper kinematics, physics drive, and contact setup.

EG2-4C2 physics joint structure (USD: EG2_4C2_left.usd)
---------------------------------------------------------
  root_joint  (PhysicsFixedJoint, ArticulationRootAPI)
      body0 = []  → anchored to parent prim (gripper_flange) xform
      body1 = gripper_base
  left_gripper_joint  (PhysicsRevoluteJoint + PhysicsDriveAPI:angular)
      body0 = gripper_base,  body1 = left_outer_link
      drive: stiffness=3, damping=1, maxForce=10 N·m, target=0°
      axis=Y, lowerLimit=0°, upperLimit=46.98°
  5 × mimic joints  (physxMimicJoint:rotY:gearing=-1.0)
      left_pad_joint, left_inner_joint, right_inner_joint,
      right_outer_joint, right_pad_joint → follow master automatically

Pad separation formula
-----------------------
  x_pad(a) = -0.04 + 0.0223·cos(a) - 0.035591·sin(a)   [metres]
  sep(a)   = 2 · |x_pad(a)|

  a = 0      → sep = 35.4 mm  (fully closed — physical minimum)
  a = 0.1945 → sep = 50.0 mm  (default open)
  a = 0.6241 → sep = 85.4 mm  (full open)

Physics drive API
-----------------
  clear_gripper_xform_overrides(stage, gripper_root)
      Remove locally-authored xform ops on finger link prims so PhysX
      can write joint outputs.  Must be called once before physics starts
      (or after settle, before Phase 1).

  set_gripper_drive_deg(stage, gripper_root, target_deg) → bool
      Set master joint targetPosition [°]. Mimic joints follow.
      Gripper closes/opens toward target; stops at contact when
      drive force = maxForce (10 N·m).

  read_gripper_joint_deg(stage, gripper_root) → float | None
      Read actual joint position [°] during simulation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

GRIPPER_OPEN_ANGLE_RAD  = 0.1945   # default open  → 50 mm gap
GRIPPER_CLOSE_ANGLE_RAD = 0.0      # fully closed  → 35.4 mm gap

GRIPPER_JOINT_SUFFIX = "joints/gripper_joint"  # base USD; left/right variant USDs rename it
CLOSE_PHYSICS_FRAMES = 60   # frames to wait for physics close (~1 s at 60 fps)

# Kinematic chain for each finger link: list of (translation, rotation_axis) joints.
# T_link = ∏ (translate(xyz) @ axis_angle(axis, joint_angle))
GRIPPER_LINK_JOINT_CHAINS: Dict[str, List[Tuple]] = {
    "left_outer_link":  [((-0.04, -0.009, 0.079), (0.0, -1.0, 0.0))],
    "left_inner_link":  [((-0.03, -0.009, 0.081), (0.0, -1.0, 0.0))],
    "right_inner_link": [((0.03,  -0.009, 0.081), (0.0,  1.0, 0.0))],
    "right_outer_link": [((0.04,  -0.009, 0.079), (0.0,  1.0, 0.0))],
    "left_pad": [
        ((-0.04, -0.009, 0.079),    (0.0, -1.0, 0.0)),
        ((0.0223, 0.003, 0.035591), (0.0,  1.0, 0.0)),
    ],
    "right_pad": [
        ((0.04,   -0.009, 0.079),    (0.0,  1.0, 0.0)),
        ((-0.0223, 0.003, 0.035591), (0.0, -1.0, 0.0)),
    ],
}

# Default contact-box parameters for tray-ear grasping (pad-prim local frame).
# Units: metres.  cx/cy/cz = centre offset; hx/hy/hz = half-extents.
#   pad local X = EG2 X = world -Z  (jaw direction)
#   pad local Y = EG2 Y = world +X
#   pad local Z = EG2 Z = world -Y  (finger / approach direction)
PAD_CONTACT_DEFAULT = {
    "left_pad":  {"cx": +0.0071, "cy": +0.006, "cz": 0.020,
                  "hx":  0.014,  "hy":  0.009, "hz": 0.008},
    "right_pad": {"cx": -0.0071, "cy": +0.006, "cz": 0.020,
                  "hx":  0.014,  "hy":  0.009, "hz": 0.008},
}

PAD_FRICTION_STATIC  = 1.5
PAD_FRICTION_DYNAMIC = 1.0

# ── Kinematic helpers ──────────────────────────────────────────────────────────

def _translate_matrix(xyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _axis_angle_matrix(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(4)
    x, y, z = axis / norm
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1.0 - c
    R = np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def gripper_link_transform(link_name: str, joint_angle_rad: float) -> np.ndarray:
    """Return 4×4 local transform of a gripper link at the given joint angle."""
    T = np.eye(4)
    for xyz, axis in GRIPPER_LINK_JOINT_CHAINS[link_name]:
        T = T @ _translate_matrix(xyz) @ _axis_angle_matrix(axis, joint_angle_rad)
    return T


def pad_separation_m(joint_angle_rad: float) -> float:
    """Return pad centre-to-centre separation (metres) for a given joint angle."""
    x = -0.04 + 0.0223 * math.cos(joint_angle_rad) - 0.035591 * math.sin(joint_angle_rad)
    return 2.0 * abs(x)


def angle_for_pad_separation_m(gap_m: float) -> float:
    """Return joint angle (rad) for a given pad separation (metres, [35.4–85.4 mm])."""
    from scipy.optimize import brentq
    min_gap, max_gap = 0.0354, 0.0854
    if gap_m < min_gap:
        raise ValueError(f"gap {gap_m*1000:.1f} mm < minimum {min_gap*1000:.1f} mm")
    if gap_m > max_gap:
        raise ValueError(f"gap {gap_m*1000:.1f} mm > maximum {max_gap*1000:.1f} mm")
    half = gap_m / 2.0
    f = lambda a: (-0.04 + 0.0223 * np.cos(a) - 0.035591 * np.sin(a)) + half
    return float(brentq(f, 0.0, 0.624))


# ── Xform (kinematic) control ─────────────────────────────────────────────────

def set_gripper_xform(gripper_ops: dict, closed_fraction: float) -> float:
    """Animate gripper via xform ops: 0.0 = open (50 mm), 1.0 = closed (35.4 mm).

    gripper_ops: {link_name: UsdGeom.XformOp} built by setup_gripper_xform_ops().
    Returns the actual joint angle used.
    """
    from pxr import Gf
    joint_angle = GRIPPER_OPEN_ANGLE_RAD * (1.0 - float(closed_fraction))
    for link_name, op in gripper_ops.items():
        T = gripper_link_transform(link_name, joint_angle)
        op.Set(Gf.Matrix4d(*sum(np.asarray(T, dtype=float).T.tolist(), [])))
    return joint_angle


def setup_gripper_xform_ops(stage, gripper_root: str, op_suffix: str = "gripper_fk") -> dict:
    """Create xform ops on all finger link prims. Returns {link_name: op} dict."""
    from pxr import UsdGeom
    ops = {}
    for link_name in GRIPPER_LINK_JOINT_CHAINS:
        prim = stage.GetPrimAtPath(f"{gripper_root}/{link_name}")
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            continue
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        ops[link_name] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, op_suffix)
    return ops


# ── Physics drive control ─────────────────────────────────────────────────────

def clear_gripper_xform_overrides(stage, gripper_root: str) -> int:
    """Remove session-layer xform overrides so PhysX can write joint outputs."""
    from pxr import UsdGeom
    cleared = 0
    for link_name in GRIPPER_LINK_JOINT_CHAINS:
        prim = stage.GetPrimAtPath(f"{gripper_root}/{link_name}")
        if not prim or not prim.IsValid():
            continue
        order_attr = prim.GetAttribute(UsdGeom.Tokens.xformOpOrder)
        if order_attr and order_attr.IsAuthored():
            order_attr.Clear()
        for suffix in ("gripper_fk", "ear_grasp_gripper_fk", "tray_demo_gripper_fk",
                       "force_demo_gripper_fk"):
            prop_name = f"xformOp:transform:{suffix}"
            if prim.HasAttribute(prop_name) and prim.GetAttribute(prop_name).IsAuthored():
                prim.RemoveProperty(prop_name)
        cleared += 1
    return cleared


def activate_gripper_joints(stage, gripper_root: str) -> int:
    """Set active=True on all EG2 joint prims (scene USD defaults them to False for xform-FK).

    Must be called before timeline.play().
    """
    joint_names = [
        "root_joint",
        "joints/gripper_joint",
        "joints/left_pad_joint",
        "joints/left_inner_joint",
        "joints/right_inner_joint",
        "joints/right_outer_joint",
        "joints/right_pad_joint",
    ]
    activated = 0
    for name in joint_names:
        prim = stage.GetPrimAtPath(f"{gripper_root}/{name}")
        if not prim or not prim.IsValid():
            print(f"[GRIPPER-ACT] WARNING: {name} not found at {gripper_root}", flush=True)
            continue
        prim.SetActive(True)
        activated += 1
    return activated


def set_gripper_drive_deg(stage, gripper_root: str, target_deg: float) -> bool:
    """Set EG2 master joint drive targetPosition [degrees]. Returns True on success.

    target_deg = 0.0   → fully closed (35.4 mm gap)
    target_deg = 11.14 → default open  (50 mm gap, GRIPPER_OPEN_ANGLE_RAD in deg)
    target_deg = 46.98 → full open    (85.4 mm gap)

    The drive caps force at maxForce=10 N·m; gripper naturally stops when
    contact resistance equals the drive force.  Mimic joints on all other
    finger links follow the master automatically via physxMimicJoint.
    """
    joint_prim = stage.GetPrimAtPath(f"{gripper_root}/{GRIPPER_JOINT_SUFFIX}")
    if not joint_prim or not joint_prim.IsValid():
        print(f"[GRIPPER-DRIVE] WARNING: joint not found at "
              f"{gripper_root}/{GRIPPER_JOINT_SUFFIX}", flush=True)
        return False
    attr = joint_prim.GetAttribute("drive:angular:physics:targetPosition")
    if not attr:
        print("[GRIPPER-DRIVE] WARNING: drive targetPosition attribute missing", flush=True)
        return False
    attr.Set(float(target_deg))
    return True


def read_gripper_joint_deg(stage, gripper_root: str) -> Optional[float]:
    """Read current physics joint position [degrees]. None if not available.

    Available after physics simulation has started and PhysX has written
    state back to USD.  The value stops changing when contact force equals
    the drive force (force-limited stop).
    """
    joint_prim = stage.GetPrimAtPath(f"{gripper_root}/{GRIPPER_JOINT_SUFFIX}")
    if not joint_prim or not joint_prim.IsValid():
        return None
    attr = joint_prim.GetAttribute("state:angular:physics:position")
    if not attr:
        return None
    val = attr.Get()
    return float(val) if val is not None else None


# ── Contact box setup ─────────────────────────────────────────────────────────

def setup_pad_contact_boxes(
    stage,
    gripper_root: str,
    pad_contact: Optional[dict] = None,
    friction_static: float = PAD_FRICTION_STATIC,
    friction_dynamic: float = PAD_FRICTION_DYNAMIC,
):
    """Add kinematic RigidBody + CollisionBox to left_pad and right_pad prims.

    pad_contact: dict with 'left_pad' and 'right_pad' entries, each having
      cx, cy, cz (centre offset in pad-local frame) and hx, hy, hz (half-extents).
      Defaults to PAD_CONTACT_DEFAULT.
    """
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    if pad_contact is None:
        pad_contact = PAD_CONTACT_DEFAULT

    mat_path = f"{gripper_root}/PadFrictionMaterial"
    if not stage.GetPrimAtPath(mat_path):
        mat = UsdShade.Material.Define(stage, mat_path)
        pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        pm.CreateStaticFrictionAttr().Set(friction_static)
        pm.CreateDynamicFrictionAttr().Set(friction_dynamic)
        pm.CreateRestitutionAttr().Set(0.0)
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))

    for pad_name, g in pad_contact.items():
        pad_path = f"{gripper_root}/{pad_name}"
        pad_prim = stage.GetPrimAtPath(pad_path)
        if not pad_prim or not pad_prim.IsValid():
            print(f"[PAD] WARNING: {pad_path} not found — skipping", flush=True)
            continue

        rb = UsdPhysics.RigidBodyAPI.Apply(pad_prim)
        rb.CreateRigidBodyEnabledAttr().Set(True)
        rb.CreateKinematicEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(pad_prim).CreateMassAttr().Set(0.02)

        box_path = f"{pad_path}/PadContactBox"
        if stage.GetPrimAtPath(box_path):
            stage.RemovePrim(box_path)
        box = UsdGeom.Cube.Define(stage, box_path)
        box.CreateSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(box.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(g["cx"], g["cy"], g["cz"]))
        xf.AddScaleOp().Set(Gf.Vec3f(g["hx"] * 2, g["hy"] * 2, g["hz"] * 2))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        print(f"[PAD] Contact box added: {pad_path}", flush=True)
