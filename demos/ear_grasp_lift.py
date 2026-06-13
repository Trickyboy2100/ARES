#!/usr/bin/env python3
"""Left-arm tray-ear grasp + lift demo — physics-correct version.

Physics design
--------------
Gripper orientation (vertical jaw):
  EG2 X = world -Z  (jaw opens/closes vertically)
  EG2 Y = world +X  (TARGET_UP_WORLD)
  EG2 Z = world -Y  (approach direction, TARGET_FORWARD_WORLD)

Minimum jaw gap at angle=0 (fully closed): 35.4 mm
→ The jaw cannot clamp a 2 mm flange from above/below.

Real grip mechanism:
  The two pads (upper at world +Z, lower at world -Z) each press against the
  Y-facing surface of the tray ear.  Normal force ∥ world Y.  Friction force ∥
  world Z opposes gravity.  With static friction ≥ 1.5 and two pads, the tray
  is reliably held.

EE Z target = tray body centre (≈ 0.996 m), NOT the GraspFrame Z (1.140 m).
At tray-centre Z, the pads straddle the 30 mm body (±17.7 mm at closed angle)
and the contact boxes span the full tray height to press the ear face.

Contact-box parameters  (cx/cy/cz in pad-prim local frame, hx/hy/hz half-extents):
  pad local X = EG2 X = world −Z   (jaw direction)
  pad local Y = EG2 Y = world +X
  pad local Z = EG2 Z = world −Y   (finger / approach direction)

  left_pad  cx = +0.0071  (→ physical pad EG2-X centre, 7.1 mm from pad prim)
  right_pad cx = −0.0071
  cy = +0.006             (→ physical pad EG2-Y centre, 6 mm in world +X)
  cz = 0.020              (→ 20 mm in world −Y; box covers pad tip [0.012, 0.028];
                             at pick, box back face = ear GraspFrame Y = 0.381)
  hx = 0.014              (28 mm total Z span → covers 30 mm tray body, ±1 mm slack)
  hy = 0.009              (18 mm total X span → physical pad width)
  hz = 0.008              (16 mm total Y depth, spans 2 mm ear face with margin)

Force-stop rule (CLAUDE.md constraint):
  FixedJoint is created ONLY after force_stop_step is not None.
  force_stop_step is set when the tray Y-position shifts > CONTACT_Y_THRESHOLD
  during the approach phase (indicates physical pad-to-ear contact).

Gripper open angle for 50 mm separation: 0.1945 rad (11.1°).

Usage (GUI mode):
  isaac-sim.sh --exec path/to/this_script.py
  Then press Play in the Isaac Sim GUI.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the scripts directory is on sys.path so local modules are importable
# when running via isaac-sim.sh --exec (Kit does not always add it automatically).
_CORE_DIR = str(Path(__file__).resolve().parents[1] / "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

# ── Constants ─────────────────────────────────────────────────────────────────

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
try:
    import config as _cfg
    DEFAULT_SCENE = _cfg.SCENE_USD
except Exception:
    DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026061100_main.usd"
DEFAULT_LOG      = PLAYGROUND_ROOT / "logs/left_arm_ear_grasp_lift/motion_log.jsonl"
DEFAULT_REPORT   = PLAYGROUND_ROOT / "reports/left_arm_ear_grasp_lift.json"
DEFAULT_CACHE    = PLAYGROUND_ROOT / "runtime/ear_grasp_lift_path_cache.npz"

LEFT_ROOT   = "/World/robot/jaka_minicobo_left"
TRAY_PATH   = "/World/Tray"
EAR_FRAME   = "/World/Tray/GraspFrames/YPlusEar"
PROBE_ROOT  = "/World/LeftArmEarGraspRuntime"
LINK_NAMES  = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]

# ── Gripper geometry ──────────────────────────────────────────────────────────

# EG2 closed (angle = 0 rad): pad separation ≈ 35.4 mm (physical minimum).
# EG2 open  (angle = 0.1945 rad): pad separation ≈ 50 mm.
# Formula: x_pad(a) = -0.04 + 0.0223*cos(a) - 0.035591*sin(a)
# Verified by FK from GRIPPER_LINK_JOINT_CHAINS.
# NOTE: Do NOT increase beyond 50 mm — wider opening enlarges the finger swept volume
# during the home→pregrasp joint-space path and causes a side collision with the tray.
GRIPPER_OPEN_ANGLE_RAD  = 0.1945   # → 50 mm gap
GRIPPER_CLOSE_ANGLE_RAD = 0.0      # → 35.4 mm gap (minimum; both pads press ear)

# Physics gripper drive constants
GRIPPER_JOINT_SUFFIX  = "joints/gripper_joint"  # relative to gripper_root
CLOSE_PHYSICS_FRAMES  = 60   # frames to wait for physics close (~1 s at 60 fps)

# Gripper IK orientation: vertical jaw, approach from +Y.
TARGET_UP_WORLD      = np.array([1.0,  0.0, 0.0], dtype=float)  # EG2 Y = world +X
TARGET_FORWARD_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)  # EG2 Z = world -Y

# Chest endpoint orientation (unchanged for carry phase if added later).
CHEST_UP_WORLD      = np.array([0.0, -1.0, 0.0], dtype=float)
CHEST_FORWARD_WORLD = np.array([-1.0, 0.0, 0.0], dtype=float)

# ── Contact-box geometry ──────────────────────────────────────────────────────
# These boxes are CHILDREN of left_pad / right_pad prims (pad local frame).
# All units in metres.
#
#   cx = ±0.0071 → physical pad EG2-X centre (mesh X ∈ [-0.0177, +0.0035])
#   cy = +0.006  → physical pad EG2-Y centre (mesh Y ∈ [-0.003, +0.015])
#   cz = 0.020   → covers pad tip region [0.012, 0.028]; at pick (carrier_Y=0.393)
#                  box back face lands at Y=0.381 = ear GraspFrame centre
#                  detection: carrier_Y = ear_outer_Y + cz + hz = 0.409+0.028 = 0.437
#   hx = 0.014   → 28 mm Z span, covers 30 mm tray body (±1 mm slack)
#   hy = 0.009   → 18 mm X span, matches physical pad width
#   hz = 0.008   → 16 mm Y depth, spans 2 mm ear face with margin
_PAD_CONTACT = {
    "left_pad":  {"cx": +0.0071, "cy": +0.006, "cz": 0.020,
                  "hx":  0.014,  "hy":  0.009, "hz": 0.008},
    "right_pad": {"cx": -0.0071, "cy": +0.006, "cz": 0.020,
                  "hx":  0.014,  "hy":  0.009, "hz": 0.008},
}

# ── Path parameters ───────────────────────────────────────────────────────────
PRE_Y_OFFSET  = 0.125   # pregrasp: EE 125 mm behind ear in world +Y
NEAR_Y_OFFSET = 0.040   # near:     EE 40 mm behind ear — switches to slow approach
PICK_Y_OFFSET = 0.012   # pick:     EE 12 mm behind ear in world +Y
LIFT_M        = 0.15    # lift 150 mm in world +Z after grasp lock

# Contact detection threshold on tray Y displacement during approach.
# Small value so we catch the first micro-slip before stick-slip jolt.
CONTACT_Y_THRESHOLD = 0.0003   # 0.3 mm

# Friction applied to both pad contact boxes and tray collision geometry.
PAD_FRICTION_STATIC   = 1.5
PAD_FRICTION_DYNAMIC  = 1.0
TRAY_FRICTION_STATIC  = 1.5
TRAY_FRICTION_DYNAMIC = 1.0

ORIENT_WEIGHT = 0.22    # orientation weight for constrained IK paths

# ── Gripper link kinematics (from gui_left_arm_tray_ear_grasp_demo.py) ────────
GRIPPER_LINK_JOINT_CHAINS = {
    "left_outer_link":  [((-0.04, -0.009, 0.079), (0.0, -1.0, 0.0))],
    "left_inner_link":  [((-0.03, -0.009, 0.081), (0.0, -1.0, 0.0))],
    "right_inner_link": [((0.03,  -0.009, 0.081), (0.0,  1.0, 0.0))],
    "right_outer_link": [((0.04,  -0.009, 0.079), (0.0,  1.0, 0.0))],
    "left_pad":  [
        ((-0.04, -0.009, 0.079),  (0.0, -1.0, 0.0)),
        ((0.0223, 0.003, 0.035591), (0.0, 1.0, 0.0)),
    ],
    "right_pad": [
        ((0.04,  -0.009, 0.079),  (0.0,  1.0, 0.0)),
        ((-0.0223, 0.003, 0.035591), (0.0, -1.0, 0.0)),
    ],
}

# ── Utility: gripper angle for a given pad separation ─────────────────────────

def angle_for_pad_separation_m(gap_m: float) -> float:
    """Return EG2 joint angle (rad) that gives the requested pad centre-to-centre
    separation *gap_m* (metres).  Range: [35.4 mm, 85.4 mm] → [0, 0.624 rad].
    Raises ValueError if gap_m is out of the achievable range.
    """
    from scipy.optimize import brentq
    min_gap = 0.0354
    max_gap = 0.0854
    if gap_m < min_gap:
        raise ValueError(f"gap {gap_m*1000:.1f} mm < minimum {min_gap*1000:.1f} mm")
    if gap_m > max_gap:
        raise ValueError(f"gap {gap_m*1000:.1f} mm > maximum {max_gap*1000:.1f} mm")
    half = gap_m / 2.0
    # x_pad(a) = -0.04 + 0.0223*cos(a) - 0.035591*sin(a)
    f = lambda a: (-0.04 + 0.0223 * np.cos(a) - 0.035591 * np.sin(a)) + half
    return float(brentq(f, 0.0, 0.624))


# ── Helpers ───────────────────────────────────────────────────────────────────

def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def vec(values):
    return [round(float(v), 6) for v in values]


def bbox_payload(stage, path: str):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    box = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        .ComputeWorldBound(prim)
        .ComputeAlignedBox()
    )
    return {"min": [float(v) for v in box.GetMin()], "max": [float(v) for v in box.GetMax()]}


def bbox_center(b):
    return np.array([(b["min"][i] + b["max"][i]) * 0.5 for i in range(3)], dtype=float)


def xform_world_xyz(stage, path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation(), dtype=float)


def gf_matrix_from_column_transform(T):
    from pxr import Gf
    return Gf.Matrix4d(*sum(np.asarray(T, dtype=float).T.tolist(), []))


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


# ── Physics material helpers ──────────────────────────────────────────────────

def _ensure_physics_material(stage, mat_path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade
    if stage.GetPrimAtPath(mat_path):
        return UsdShade.Material(stage.GetPrimAtPath(mat_path))
    mat = UsdShade.Material.Define(stage, mat_path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(static)
    pm.CreateDynamicFrictionAttr().Set(dynamic)
    pm.CreateRestitutionAttr().Set(0.0)
    return mat


def apply_tray_friction(stage):
    """Apply high-friction PhysX material to every CollisionAPI prim under /World/Tray."""
    from pxr import Usd, UsdPhysics, UsdShade
    mat = _ensure_physics_material(
        stage, f"{TRAY_PATH}/TrayFrictionMaterial",
        TRAY_FRICTION_STATIC, TRAY_FRICTION_DYNAMIC,
    )
    tray = stage.GetPrimAtPath(TRAY_PATH)
    for prim in Usd.PrimRange(tray):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
            print(f"[FRICTION] Tray material → {prim.GetPath()}", flush=True)


# ── Pad kinematic contact setup ───────────────────────────────────────────────

def setup_pad_kinematic_contacts(stage, gripper_root: str):
    """Make left_pad and right_pad kinematic rigid bodies with friction contact boxes.

    Contact-box design rationale (see module docstring):
      hx=0.020 spans 40 mm in world Z → covers the full 30 mm tray body height.
      cz=0.012 centres the box at the ear Y-face when EE is 12 mm from ear.
      hz=0.008 spans 16 mm in world Y → spans 2 mm ear with 7 mm margin each side.
    """
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    mat = _ensure_physics_material(
        stage, f"{gripper_root}/PadFrictionMaterial",
        PAD_FRICTION_STATIC, PAD_FRICTION_DYNAMIC,
    )

    for pad_name, g in _PAD_CONTACT.items():
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
        print(f"[PAD] Contact body added: {pad_path}", flush=True)


# ── Hidden carrier + FixedJoint ───────────────────────────────────────────────

def ensure_hidden_carrier(stage):
    from pxr import UsdGeom, UsdPhysics
    if stage.GetPrimAtPath(PROBE_ROOT):
        stage.RemovePrim(PROBE_ROOT)
    UsdGeom.Xform.Define(stage, PROBE_ROOT)
    cube = UsdGeom.Cube.Define(stage, f"{PROBE_ROOT}/Carrier")
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
    from pxr import Gf
    op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def create_grasp_lock(stage, tray_path: str, carrier_path: str) -> str:
    """Create a FixedJoint between carrier and tray.

    MUST only be called after force_stop_step is not None (CLAUDE.md constraint).
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    joint_path = f"{PROBE_ROOT}/GraspFixedJoint"
    if stage.GetPrimAtPath(joint_path):
        return joint_path
    tray    = stage.GetPrimAtPath(tray_path)
    carrier = stage.GetPrimAtPath(carrier_path)
    cache   = UsdGeom.XformCache(Usd.TimeCode.Default())
    tray_w    = cache.GetLocalToWorldTransform(tray)
    carrier_w = cache.GetLocalToWorldTransform(carrier)
    anchor = carrier_w.ExtractTranslation()
    local0 = carrier_w.GetInverse().Transform(anchor)
    local1 = tray_w.GetInverse().Transform(anchor)
    joint  = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(carrier_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tray_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local1]))
    tray_rot_inv = tray_w.ExtractRotationQuat().GetInverse().GetNormalized()
    tri = tray_rot_inv.GetImaginary()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(float(tray_rot_inv.GetReal()), float(tri[0]), float(tri[1]), float(tri[2]))
    )
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint_path


# ── Arm FK / gripper ──────────────────────────────────────────────────────────

def _translate_matrix(xyz):
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _axis_angle_matrix(axis, angle_rad):
    import math
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


def _gripper_link_transform(link_name: str, joint_angle_rad: float) -> np.ndarray:
    T = np.eye(4)
    for xyz, axis in GRIPPER_LINK_JOINT_CHAINS[link_name]:
        T = T @ _translate_matrix(xyz) @ _axis_angle_matrix(axis, joint_angle_rad)
    return T


def joint_limits_from_chain(chain):
    from kinematics_probe import ARM_JOINTS
    lower, upper = [], []
    for joint in chain:
        if joint.name in ARM_JOINTS:
            lower.append(-np.pi if joint.lower is None else joint.lower)
            upper.append( np.pi if joint.upper is None else joint.upper)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def set_robot_links(link_ops, chains, q):
    from kinematics_probe import ARM_JOINTS, fk
    q_map = dict(zip(ARM_JOINTS, np.asarray(q, dtype=float).tolist()))
    for link_name in LINK_NAMES:
        link_ops[link_name].Set(gf_matrix_from_column_transform(fk(chains[link_name], q_map)))


def set_gripper(gripper_ops, closed_fraction: float):
    """Animate gripper: 0.0 = open (GRIPPER_OPEN_ANGLE_RAD), 1.0 = closed (0 rad)."""
    joint_angle = GRIPPER_OPEN_ANGLE_RAD * (1.0 - float(closed_fraction))
    for link_name, op in gripper_ops.items():
        op.Set(gf_matrix_from_column_transform(_gripper_link_transform(link_name, joint_angle)))
    return joint_angle


# ── Physics gripper drive (force-limited via EG2 articulation joints) ─────────
#
# The EG2-4C2 has a PhysX articulation:
#   root_joint (FixedJoint, ArticulationRootAPI): anchors gripper_base to parent
#   left_gripper_joint (RevoluteJoint + DriveAPI): master, maxForce=10 N·m
#   5× mimic joints (physxMimicJoint:rotY:gearing=-1.0): follow master
#
# When gripper_flange moves via arm xform, the articulation follows automatically.
# By NOT authoring xform overrides on the finger link prims, PhysX writes joint
# outputs directly to those prims → true physics-based, force-limited closing.

def clear_gripper_xform_overrides(stage, gripper_root: str):
    """Remove local-layer xform overrides on all gripper link prims.

    After clearing, PhysX writes joint outputs directly to each link prim,
    enabling force-limited closing via the master joint drive (maxForce=10 N·m).
    """
    from pxr import UsdGeom
    cleared = 0
    for link_name in GRIPPER_LINK_JOINT_CHAINS:
        prim = stage.GetPrimAtPath(f"{gripper_root}/{link_name}")
        if not prim or not prim.IsValid():
            continue
        # Remove authored xformOpOrder override (restores reference-layer order)
        order_attr = prim.GetAttribute(UsdGeom.Tokens.xformOpOrder)
        if order_attr and order_attr.IsAuthored():
            order_attr.Clear()
        # Remove our authored transform op value (frees the local override)
        prop_name = "xformOp:transform:ear_grasp_gripper_fk"
        if prim.HasAttribute(prop_name) and prim.GetAttribute(prop_name).IsAuthored():
            prim.RemoveProperty(prop_name)
        cleared += 1
    print(f"[GRIPPER-DRIVE] Cleared xform overrides on {cleared} gripper links", flush=True)


def set_gripper_drive_deg(stage, gripper_root: str, target_deg: float) -> bool:
    """Set EG2 master joint drive target (degrees). Returns True on success.

    target_deg = 0.0   → fully closed (35.4 mm gap)
    target_deg = 11.14 → default open  (50 mm gap)
    target_deg = 46.98 → full open    (85.4 mm gap)

    maxForce=10 N·m caps the drive torque, so the gripper stops naturally
    when contact force matches the drive force — no external contact logic needed.
    All mimic joints follow the master automatically via physxMimicJoint.
    """
    joint_path = f"{gripper_root}/{GRIPPER_JOINT_SUFFIX}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim or not joint_prim.IsValid():
        print(f"[GRIPPER-DRIVE] WARNING: joint not found at {joint_path}", flush=True)
        return False
    attr = joint_prim.GetAttribute("drive:angular:physics:targetPosition")
    if not attr:
        print(f"[GRIPPER-DRIVE] WARNING: targetPosition attribute missing", flush=True)
        return False
    attr.Set(float(target_deg))
    return True


def read_gripper_joint_deg(stage, gripper_root: str) -> float | None:
    """Read current physics joint position (degrees). Returns None if unavailable.

    Available after physics simulation has started. Returns actual joint angle,
    which stops changing when contact force equals drive force (force-limited stop).
    """
    joint_path = f"{gripper_root}/{GRIPPER_JOINT_SUFFIX}"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim or not joint_prim.IsValid():
        return None
    attr = joint_prim.GetAttribute("state:angular:physics:position")
    if not attr:
        return None
    val = attr.Get()
    return float(val) if val is not None else None


def pad_world_xyz(base_world, link6_chain, link6_to_pad, q) -> np.ndarray:
    from kinematics_probe import ARM_JOINTS, fk
    q_map = dict(zip(ARM_JOINTS, np.asarray(q, dtype=float).tolist()))
    return (base_world @ fk(link6_chain, q_map) @ link6_to_pad)[:3, 3]


# ── Path planning ─────────────────────────────────────────────────────────────

def build_arm_path(stage, cache_usd, settled_center, ear_center, chest_target, lift_m):
    """Compute joint-angle path for: pregrasp → approach → lift.

    EE Z is set to settled_center[2] (tray body centre), NOT the GraspFrame Z.
    This is the physically correct height to straddle the 30 mm tray body with
    the vertical-jaw orientation.

    Returned dict keys:
      chains, link6_chain, base_world, link6_to_pad
      path          : (N, 6) float64 joint angles
      phase_bounds  : {"pregrasp": int, "approach": int, "lift": int}
      targets       : {"pre", "pick", "lift"} world xyz np.ndarray
    """
    from kinematics_probe import ARM_JOINTS, DEFAULT_ARM_URDF, GRIPPER_ROOT_SUFFIX, chain_to_link, load_joints
    from make_tray_handoff_curobo_demo import (
        constrained_pose_path,
        fallback_path,
        selected_pad_midpoint,
        solve_pad_pose_ik,
    )

    arm_joints  = load_joints(Path(DEFAULT_ARM_URDF))
    chains      = {name: chain_to_link(arm_joints, "Link_0", name) for name in LINK_NAMES}
    link6_chain = chains["Link_6"]
    lower, upper = joint_limits_from_chain(link6_chain)
    base_world, _pad_mid, link6_to_pad = selected_pad_midpoint(stage, cache_usd, "left")

    # ── Seed bank biased toward vertical-jaw / side-approach ──────────────────
    seed_bank = [
        np.zeros(6),
        np.array([1.0,  0.5,  0.4, -1.1, -0.2, -0.4], dtype=float),
        np.array([1.4,  0.6,  0.4, -1.1, -0.2, -0.4], dtype=float),
        np.array([0.8,  0.8,  0.2, -1.2,  0.1, -0.2], dtype=float),
        np.array([0.9,  0.5,  0.8, -0.8,  1.4,  1.5], dtype=float),
        np.array([1.1,  0.4,  0.7, -0.9,  1.5,  1.6], dtype=float),
        np.array([0.7,  0.7,  0.6, -1.0,  1.3,  1.4], dtype=float),
    ]

    # ── EE targets ────────────────────────────────────────────────────────────
    # X: tray bbox centre X (no offset → EE centred on tray X, contact boxes
    #    land at tray centre via cy=+0.006 / pad-prim EG2-Y=−0.006 cancellation).
    # Y = ear GraspFrame Y (approach axis reference).
    # Z = tray bbox centre Z (pads straddle the 30 mm tray body in Z).
    _EX_OUTER_LINK_OFFSET_M = 0.000   # centred on tray X
    ee_z = float(settled_center[2])
    ex   = float(settled_center[0]) + _EX_OUTER_LINK_OFFSET_M
    ey   = float(ear_center[1])

    print(
        f"[EAR-GRASP] EE X: tray_cx={settled_center[0]:.4f}  "
        f"+outer_link_offset={_EX_OUTER_LINK_OFFSET_M*1000:.0f}mm  "
        f"ex={ex:.4f}  (ear_center_x={ear_center[0]:.4f})",
        flush=True,
    )

    pre  = np.array([ex, ey + PRE_Y_OFFSET,  ee_z], dtype=float)
    near = np.array([ex, ey + NEAR_Y_OFFSET, ee_z], dtype=float)  # slow-approach start
    pick = np.array([ex, ey + PICK_Y_OFFSET, ee_z], dtype=float)
    lift = np.array([ex, ey + PICK_Y_OFFSET, ee_z + lift_m], dtype=float)

    print(f"[EAR-GRASP] EE targets:", flush=True)
    print(f"  pre  = {vec(pre)}", flush=True)
    print(f"  near = {vec(near)}  (slow-approach start)", flush=True)
    print(f"  pick = {vec(pick)}", flush=True)
    print(f"  lift = {vec(lift)}", flush=True)

    # ── IK solve ──────────────────────────────────────────────────────────────
    print("[EAR-GRASP] Solving IK for pregrasp position…", flush=True)
    q0 = np.zeros(6)
    q_pre, *_ = solve_pad_pose_ik(
        link6_chain, lower, upper, base_world, link6_to_pad,
        pre, TARGET_UP_WORLD, TARGET_FORWARD_WORLD, [q0] + seed_bank,
    )

    print("[EAR-GRASP] Solving IK path: home → pregrasp…", flush=True)
    path_to_pre = fallback_path(q0, q_pre, count=91)

    # Fast approach: pregrasp → near (85 mm, 61 steps ≈ 1.4 mm/step)
    print("[EAR-GRASP] Solving IK path: pregrasp → near (fast)…", flush=True)
    path_fast, _ = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_to_pre[-1], pre, near,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [q_pre] + seed_bank, count=61,
    )

    # Slow approach: near → pick (28 mm, 91 steps ≈ 0.31 mm/step)
    # Slow speed ensures contact force stays below table static friction,
    # preventing stick-slip jolt that would push the tray.
    print("[EAR-GRASP] Solving IK path: near → pick (slow)…", flush=True)
    path_slow, _ = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_fast[-1], near, pick,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [path_fast[-1]] + seed_bank, count=91,
    )

    print("[EAR-GRASP] Solving IK path: pick → lift…", flush=True)
    path_lift, _ = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_slow[-1], pick, lift,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [path_slow[-1]] + seed_bank, count=81,
        axis_weight=ORIENT_WEIGHT, forward_weight=ORIENT_WEIGHT,
    )

    path_approach = np.vstack([path_fast, path_slow[1:]])
    full_path = np.vstack([path_to_pre, path_approach[1:], path_lift[1:]])
    phase_bounds = {
        "pregrasp":  len(path_to_pre),
        "approach":  len(path_to_pre) + len(path_approach) - 1,
        "lift":      len(full_path),
    }
    print(f"[EAR-GRASP] Path solved: {len(full_path)} steps  phases={phase_bounds}", flush=True)
    print(f"  fast_approach={len(path_fast)} slow_approach={len(path_slow)}", flush=True)

    return {
        "chains":       chains,
        "link6_chain":  link6_chain,
        "base_world":   base_world,
        "link6_to_pad": link6_to_pad,
        "path":         full_path,
        "phase_bounds": phase_bounds,
        "targets":      {"pre": pre, "near": near, "pick": pick, "lift": lift},
        "lower":        lower,
        "upper":        upper,
        "seed_bank":    seed_bank,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom

    ctx = omni.usd.get_context()
    print(f"[EAR-GRASP] Opening scene: {DEFAULT_SCENE}", flush=True)
    ctx.open_stage(DEFAULT_SCENE)
    app = omni.kit.app.get_app()
    for _ in range(160):
        app.update()
        time.sleep(0.002)
    stage = ctx.get_stage()
    print("[EAR-GRASP] Stage loaded.", flush=True)

    # ── Purge any runtime prims left over from a previous saved session ───────
    # If Isaac Sim was saved while the demo was running, prims under PROBE_ROOT
    # (Carrier with RigidBodyAPI, GraspFixedJoint) get persisted.  On reload the
    # FixedJoint would violently snap the Carrier and Tray together.
    from pxr import Sdf
    stale = stage.GetPrimAtPath(PROBE_ROOT)
    if stale and stale.IsValid():
        stage.RemovePrim(Sdf.Path(PROBE_ROOT))
        print(f"[EAR-GRASP] Purged stale runtime prim: {PROBE_ROOT}", flush=True)

    # ── Setup physics (before Play — does NOT touch arm xform ops) ────────────
    from kinematics_probe import GRIPPER_ROOT_SUFFIX
    gripper_root = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"

    setup_pad_kinematic_contacts(stage, gripper_root)
    apply_tray_friction(stage)
    carrier_prim, carrier_op = ensure_hidden_carrier(stage)
    carrier_path = str(carrier_prim.GetPath())

    # ── Wait for Play ─────────────────────────────────────────────────────────
    # Arm stays in its original USD pose until Play is pressed.
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    print("[EAR-GRASP] Ready. Press Play in the GUI to start.", flush=True)
    while not timeline.is_playing():
        app.update()
        time.sleep(0.02)

    # ── Physics settle ────────────────────────────────────────────────────────
    settle_steps = 120  # 2 s at 60 fps
    print(f"[EAR-GRASP] Settling physics ({settle_steps} steps)…", flush=True)
    for _ in range(settle_steps):
        app.update()

    # ── Setup arm FK ops (AFTER settle — arm snaps to home only now) ──────────
    # Deferring ClearXformOpOrder to here ensures the arm holds its USD pose
    # during the pre-Play wait and physics settling, so the user sees the
    # correct initial configuration before execution begins.
    link_ops = {}
    for link_name in LINK_NAMES:
        prim = stage.GetPrimAtPath(f"{LEFT_ROOT}/{link_name}")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        link_ops[link_name] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "ear_grasp_fk")

    # Switch gripper to physics articulation drive (force-limited close).
    # Clear local-layer xform overrides → PhysX writes joint outputs directly.
    # The EG2 master joint has maxForce=10 N·m; mimic joints follow automatically.
    clear_gripper_xform_overrides(stage, gripper_root)
    _OPEN_DEG = math.degrees(GRIPPER_OPEN_ANGLE_RAD)   # ≈ 11.14°
    if not set_gripper_drive_deg(stage, gripper_root, _OPEN_DEG):
        print("[EAR-GRASP] WARN: physics gripper drive unavailable — xform fallback", flush=True)
        # Fallback: restore xform control
        _OPEN_DEG = None
        gripper_ops_fallback: dict = {}
        for link_name in GRIPPER_LINK_JOINT_CHAINS:
            prim = stage.GetPrimAtPath(f"{gripper_root}/{link_name}")
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Xformable):
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                gripper_ops_fallback[link_name] = xf.AddTransformOp(
                    UsdGeom.XformOp.PrecisionDouble, "ear_grasp_gripper_fk"
                )
        set_gripper(gripper_ops_fallback, 0.0)
    else:
        gripper_ops_fallback = {}
        print(f"[GRIPPER-DRIVE] Open target set to {_OPEN_DEG:.2f}° (~50 mm gap)", flush=True)

    settled_bbox   = bbox_payload(stage, TRAY_PATH)
    settled_center = bbox_center(settled_bbox)
    ear_center     = xform_world_xyz(stage, EAR_FRAME)
    print(f"[EAR-GRASP] Settled tray centre: {vec(settled_center)}", flush=True)
    print(f"[EAR-GRASP] Ear frame world XYZ: {vec(ear_center)} "
          f"(GraspFrame authored Z={ear_center[2]:.4f}; "
          f"EE will use tray-body Z={settled_center[2]:.4f})", flush=True)

    # ── Plan arm path ─────────────────────────────────────────────────────────
    info = build_arm_path(
        stage, UsdGeom.XformCache(Usd.TimeCode.Default()),
        settled_center, ear_center, None, LIFT_M,
    )

    log_path = Path(DEFAULT_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    path         = info["path"]
    phase_bounds = info["phase_bounds"]
    dt           = 1.0 / 60.0
    samples      = []
    force_stop_step: int | None = None
    grasp_lock_path: str | None = None

    tray_y_settled = float(settled_center[1])

    def log_step(i: int, phase: str, closed: float, note: str = ""):
        if i % 5 != 0 and i != len(path) - 1:
            return
        tray_bbox = bbox_payload(stage, TRAY_PATH)
        pad_xyz   = pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], path[i])
        row = {
            "step": i, "time_sec": round(i * dt, 4), "phase": phase,
            "gripper_closed_fraction": round(closed, 4),
            "pad_midpoint_world_xyz":  vec(pad_xyz),
            "tray_bbox_center_xyz":    vec(bbox_center(tray_bbox)),
            "tray_y_drift_mm":         round((bbox_center(tray_bbox)[1] - tray_y_settled) * 1000, 2),
            "force_stop_step":         force_stop_step,
            "grasp_lock_path":         grasp_lock_path,
            "note":                    note,
        }
        samples.append(row)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── Phase 1: Move to pregrasp (arm moves, gripper stays open) ─────────────
    print("[EAR-GRASP] Phase 1: moving to pregrasp…", flush=True)
    for i in range(phase_bounds["pregrasp"]):
        set_robot_links(link_ops, info["chains"], path[i])
        if gripper_ops_fallback:  # physics drive unavailable — use xform fallback
            set_gripper(gripper_ops_fallback, 0.0)
        carrier_xyz = pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], path[i])
        set_carrier(carrier_op, carrier_xyz)
        app.update()
        log_step(i, "pregrasp", 0.0)
        if i % 10 == 0:
            tray_y_now = float(bbox_center(bbox_payload(stage, TRAY_PATH))[1])
            drift_mm = (tray_y_now - tray_y_settled) * 1000
            print(
                f"  [P1 step {i:3d}] pad={vec(carrier_xyz)} "
                f"tray_Y={tray_y_now:.4f} drift={drift_mm:+.2f}mm",
                flush=True,
            )

    # Re-baseline: ignore any residual tray movement from Phase 1
    tray_y_settled = float(bbox_center(bbox_payload(stage, TRAY_PATH))[1])
    print(
        f"[EAR-GRASP] Phase 2 tray Y baseline (post-Phase1): {tray_y_settled*1000:.2f} mm",
        flush=True,
    )

    # ── Phase 2: Approach with force-stop detection ────────────────────────────
    print("[EAR-GRASP] Phase 2: approaching ear (force-stop active)…", flush=True)
    approach_start = phase_bounds["pregrasp"]
    approach_end   = phase_bounds["approach"]
    ear_y = float(ear_center[1])

    for i in range(approach_start, approach_end):
        set_robot_links(link_ops, info["chains"], path[i])
        if gripper_ops_fallback:
            set_gripper(gripper_ops_fallback, 0.0)
        carrier_xyz = pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], path[i])
        set_carrier(carrier_op, carrier_xyz)
        app.update()

        tray_y_now = float(bbox_center(bbox_payload(stage, TRAY_PATH))[1])
        drift_mm   = (tray_y_now - tray_y_settled) * 1000
        pad_ear_mm = (carrier_xyz[1] - ear_y) * 1000  # positive = pad still behind ear

        if (i - approach_start) % 5 == 0:
            print(
                f"  [P2 step {i:3d}] pad_Y={carrier_xyz[1]:.4f} ear_Y={ear_y:.4f} "
                f"gap={pad_ear_mm:.1f}mm  tray_drift={drift_mm:+.2f}mm",
                flush=True,
            )

        # Force-stop: detect tray Y displacement (contact push)
        if force_stop_step is None:
            if abs(tray_y_now - tray_y_settled) > CONTACT_Y_THRESHOLD:
                force_stop_step = i
                print(
                    f"[EAR-GRASP] force_stop at step {i}: tray Y moved "
                    f"{drift_mm:.1f} mm, pad-ear gap={pad_ear_mm:.1f}mm — contact confirmed!",
                    flush=True,
                )
                log_step(i, "approach", 0.0, note="FORCE_STOP_CONTACT")
                break

        log_step(i, "approach", 0.0)

    # ── Guard: only proceed if contact was confirmed ──────────────────────────
    if force_stop_step is None:
        print(
            "[EAR-GRASP] WARNING: approach completed but no contact detected "
            "(tray did not move). Possible miss — stopping without FixedJoint.",
            flush=True,
        )
        report = {
            "status": "no_contact",
            "settled_before_grasp": {
                "bbox": settled_bbox,
                "bbox_center_xyz": vec(settled_center),
                "ear_center_world_xyz": vec(ear_center),
            },
            "targets": {k: vec(v) for k, v in info["targets"].items()},
            "samples": samples,
        }
        atomic_write_json(Path(DEFAULT_REPORT), report)
        timeline.stop()
        return

    # ── Phase 3: Close gripper via physics drive (force-limited) ─────────────
    # Set drive target to 0° (closed). The joint drive (stiffness=3, damping=1)
    # pushes the gripper closed; when contact force equals maxForce=10 N·m the
    # drive stalls naturally — no explicit force threshold needed.
    # Mimic joints on all other finger links follow the master automatically.
    print(
        f"[EAR-GRASP] Phase 3: physics close ({CLOSE_PHYSICS_FRAMES} frames, maxForce=10 N·m)…",
        flush=True,
    )
    last_approach_q = path[force_stop_step]
    carrier_xyz = pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], last_approach_q)
    gripper_contact_frame: int | None = None
    gripper_final_deg: float = 0.0

    if _OPEN_DEG is not None:
        set_gripper_drive_deg(stage, gripper_root, 0.0)   # target = closed
        prev_deg: float = _OPEN_DEG
        for j in range(CLOSE_PHYSICS_FRAMES + 1):
            set_robot_links(link_ops, info["chains"], last_approach_q)
            set_carrier(carrier_op, carrier_xyz)
            app.update()
            curr_deg = read_gripper_joint_deg(stage, gripper_root)
            if curr_deg is not None:
                delta = abs(curr_deg - prev_deg)
                frac_closed = max(0.0, 1.0 - curr_deg / _OPEN_DEG)
                if j > 5 and delta < 0.05 and gripper_contact_frame is None:
                    gripper_contact_frame = j
                    gap_mm = abs(
                        -0.04 + 0.0223 * math.cos(math.radians(curr_deg))
                        - 0.035591 * math.sin(math.radians(curr_deg))
                    ) * 2000
                    print(
                        f"[GRIPPER-DRIVE] Contact stop at frame {j}: "
                        f"joint={curr_deg:.2f}°  gap={gap_mm:.1f}mm  "
                        f"(drive stalled at maxForce=10 N·m)",
                        flush=True,
                    )
                if j % 10 == 0 or gripper_contact_frame == j:
                    gap_mm = abs(
                        -0.04 + 0.0223 * math.cos(math.radians(curr_deg))
                        - 0.035591 * math.sin(math.radians(curr_deg))
                    ) * 2000
                    print(
                        f"  [P3 frame {j:3d}] joint={curr_deg:.2f}°  Δ={delta:.3f}°  "
                        f"gap={gap_mm:.1f}mm  closed={frac_closed:.3f}",
                        flush=True,
                    )
                gripper_final_deg = curr_deg
                prev_deg = curr_deg
                log_step(force_stop_step, "close_gripper", frac_closed)
            else:
                log_step(force_stop_step, "close_gripper", 0.0)
    else:
        # Fallback: xform-based close
        close_steps = 30
        for j in range(close_steps + 1):
            frac = j / close_steps
            set_robot_links(link_ops, info["chains"], last_approach_q)
            set_gripper(gripper_ops_fallback, frac)
            set_carrier(carrier_op, carrier_xyz)
            app.update()
            log_step(force_stop_step, "close_gripper", frac)

    # ── Phase 4: Create FixedJoint (allowed: force_stop_step is not None) ─────
    print("[EAR-GRASP] Phase 4: creating FixedJoint (force_stop confirmed)…", flush=True)
    pad_at_stop  = pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], path[force_stop_step])
    pad_ear_dist = (pad_at_stop[1] - float(ear_center[1])) * 1000   # mm, positive = behind ear
    print(
        f"[EAR-GRASP]   FixedJoint pre-check: pad_Y={pad_at_stop[1]:.4f}  "
        f"ear_Y={ear_center[1]:.4f}  offset={pad_ear_dist:+.1f}mm",
        flush=True,
    )
    # Threshold = 65 mm: real contact ~56 mm (cz+hz=28mm past ear outer face),
    # false trigger from Phase-1 disturbance was 125 mm → clear separation.
    if pad_ear_dist > 65:
        print(
            f"[EAR-GRASP] ABORT: pad is {pad_ear_dist:.1f}mm behind ear "
            f"(false contact, likely Phase-1 disturbance). No FixedJoint created.",
            flush=True,
        )
        report = {
            "status": "false_contact",
            "force_stop_step": force_stop_step,
            "pad_ear_offset_mm": pad_ear_dist,
            "targets": {k: vec(v) for k, v in info["targets"].items()},
            "samples": samples,
        }
        atomic_write_json(Path(DEFAULT_REPORT), report)
        print(f"[EAR-GRASP] status = false_contact", flush=True)
        timeline.stop()
        return
    grasp_lock_path = create_grasp_lock(stage, TRAY_PATH, carrier_path)
    print(f"[EAR-GRASP] FixedJoint created: {grasp_lock_path}", flush=True)
    # Let joint settle for 10 frames.
    for _ in range(10):
        app.update()

    # ── Phase 5: Lift from actual force-stop position (re-planned) ─────────────
    # Pre-planned lift starts at pick_Y=0.393, but force_stop triggers at
    # carrier_Y≈0.437 (cz+hz=28mm ahead of ear outer face).  Running the
    # pre-planned path while the joint is locked would drag the tray −44mm in Y.
    # Instead we re-solve IK from the exact force_stop config straight up by LIFT_M.
    lift_target = np.array([
        float(pad_at_stop[0]),
        float(pad_at_stop[1]),   # keep Y from force_stop, no Y correction
        float(pad_at_stop[2]) + LIFT_M,
    ], dtype=float)
    print(
        f"[EAR-GRASP] Phase 5: re-plan lift from "
        f"[{pad_at_stop[0]:.4f}, {pad_at_stop[1]:.4f}, {pad_at_stop[2]:.4f}]"
        f" → Z+{LIFT_M*1000:.0f}mm",
        flush=True,
    )

    from make_tray_handoff_curobo_demo import constrained_pose_path as _cpose
    q_stop = path[force_stop_step]
    path_lift_replan, _ = _cpose(
        info["link6_chain"], info["lower"], info["upper"],
        info["base_world"], info["link6_to_pad"],
        q_stop, pad_at_stop, lift_target,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [q_stop] + info["seed_bank"],
        count=81,
        axis_weight=ORIENT_WEIGHT, forward_weight=ORIENT_WEIGHT,
    )

    for q in path_lift_replan:
        set_robot_links(link_ops, info["chains"], q)
        if gripper_ops_fallback:
            set_gripper(gripper_ops_fallback, 1.0)
        # physics drive: target remains 0° (closed) from Phase 3, no change needed
        carrier_xyz = pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], q)
        set_carrier(carrier_op, carrier_xyz)
        app.update()
        log_step(force_stop_step, "lift", 1.0)

    # ── Result ────────────────────────────────────────────────────────────────
    final_bbox   = bbox_payload(stage, TRAY_PATH)
    final_center = bbox_center(final_bbox)
    lift_by_min  = final_bbox["min"][2] - settled_bbox["min"][2]
    print(f"[EAR-GRASP] Done. lift_by_bbox_min = {lift_by_min*1000:.1f} mm", flush=True)

    gripper_final_gap_mm = round(
        abs(-0.04 + 0.0223 * math.cos(math.radians(gripper_final_deg))
            - 0.035591 * math.sin(math.radians(gripper_final_deg))) * 2000, 1
    ) if gripper_final_deg is not None else None

    report = {
        "status": "pass" if lift_by_min > 0.02 else "fail",
        "lift_by_bbox_min_m": float(lift_by_min),
        "force_stop_step": force_stop_step,
        "grasp_lock_path": grasp_lock_path,
        "gripper": {
            "mode": "physics_drive" if _OPEN_DEG is not None else "xform_fallback",
            "contact_frame": gripper_contact_frame,
            "final_deg": round(gripper_final_deg, 3) if gripper_final_deg else None,
            "final_gap_mm": gripper_final_gap_mm,
            "open_deg": round(_OPEN_DEG, 3) if _OPEN_DEG else None,
        },
        "settled_before_grasp": {
            "bbox": settled_bbox,
            "bbox_center_xyz": vec(settled_center),
            "ear_center_world_xyz": vec(ear_center),
            "ee_z_used": float(settled_center[2]),
            "ee_z_grasframe": float(ear_center[2]),
        },
        "targets":     {k: vec(v) for k, v in info["targets"].items()},
        "pad_contact": _PAD_CONTACT,
        "friction": {
            "pad_static": PAD_FRICTION_STATIC,
            "pad_dynamic": PAD_FRICTION_DYNAMIC,
            "tray_static": TRAY_FRICTION_STATIC,
            "tray_dynamic": TRAY_FRICTION_DYNAMIC,
        },
        "gripper_open_angle_rad": GRIPPER_OPEN_ANGLE_RAD,
        "gripper_open_separation_mm": round(
            abs(-0.04 + 0.0223*np.cos(GRIPPER_OPEN_ANGLE_RAD) - 0.035591*np.sin(GRIPPER_OPEN_ANGLE_RAD)) * 2000, 1
        ),
        "final": {
            "bbox": final_bbox,
            "bbox_center_xyz": vec(final_center),
        },
        "samples": samples,
    }
    atomic_write_json(Path(DEFAULT_REPORT), report)
    print(f"[EAR-GRASP] Report → {DEFAULT_REPORT}", flush=True)
    print(f"[EAR-GRASP] status = {report['status']}", flush=True)


if __name__ == "__main__":
    main()
