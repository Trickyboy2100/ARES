#!/usr/bin/env python3
"""tray_grasp_cycle v3 — full left→right arm handoff demo.

State machine per cycle:
  TO_PRE_L  → APPROACH_L (Y-linear) → CLOSE_GRIP_L →
  LIFT_L    → CARRY_L   →
  R_TO_NEAR → R_APPROACH → CLOSE_GRIP_R →
  RELEASE_L → RETRACT_L → HOME_L →
  CARRY_DRYER → HOLD_DRYER → RELEASE_R → HOME_R →
  RESET_SCENE → PAUSE → [repeat]

Motion planning:
  All segments:  cuRobo (max_attempts=8, orientation-aware, cached).
                 On cuRobo failure: single-frame jump to goal (no interpolation).
  Lift IK:       solve_pad_pose_ik at contact+LIFT_Z, then cuRobo from contact→lift_end.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ── sys.path ──────────────────────────────────────────────────────────────────
_DEMO_DIR  = Path(__file__).resolve().parent           # demos/tray_grasp_cycle/
_SIMFORGE  = _DEMO_DIR.parents[1]                      # simforge/ (repo root)
_CORE      = _SIMFORGE / "core"
for _p in (str(_SIMFORGE), str(_CORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Scene / URDF paths — computed directly from __file__, env vars override ──
# Do NOT use `import config` here: Isaac Sim may have another `config` module
# on sys.path that shadows simforge/config.py.
import os as _os
DEFAULT_SCENE    = _os.environ.get("SIMFORGE_SCENE") or str(_SIMFORGE / "scenes" / "main.usd")
DEFAULT_ARM_URDF = str(
    Path(_os.environ.get("SIMFORGE_URDF_DIR") or str(_SIMFORGE / "robot"))
    / "jaka_minicobo.urdf"
)

from kinematics import (
    GRIPPER_ROOT_SUFFIX, ARM_JOINTS, chain_to_link, load_joints, fk, get_world_pose,
)
from gripper import (
    GRIPPER_OPEN_ANGLE_RAD,
    gripper_link_transform,
    pad_separation_m,
    setup_gripper_xform_ops,
)
from planning import (
    selected_pad_midpoint,
    solve_pad_pose_ik,
    fallback_path,
    constrained_pose_path,
    pad_world_transform,
    build_curobo_obstacles,
    curobo_tool_pose,
)
from ik_sanity import joint_limits
from scene_utils import (
    gf_matrix_from_column_transform,
    ensure_hidden_carrier,
    set_carrier,
    create_grasp_lock,
)

# ── Robot prim paths ──────────────────────────────────────────────────────────
LEFT_ROOT  = "/World/robot/jaka_minicobo_left"
RIGHT_ROOT = "/World/robot/jaka_minicobo_right"
LEFT_GR    = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
RIGHT_GR   = f"{RIGHT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]
ROBOT_PARENT_PATH = "/World/robot"

TRAY_PATH       = "/World/Tray"
GRASP_PRIM_L    = "/World/Tray/tray_grasp_point"       # left ear
GRASP_PRIM_R    = "/World/Tray/tray_grasp_init_point"  # right ear

CARRIER_ROOT_L  = "/World/_GraspCarrier_L"
JOINT_PATH_L    = "/World/_GraspLock_L"
CARRIER_ROOT_R  = "/World/_GraspCarrier_R"
JOINT_PATH_R    = "/World/_GraspLock_R"
RESET_CARRIER   = "/World/_ResetCarrier"
RESET_JOINT     = "/World/_ResetLock"

DRYER_PATH      = "/World/Dryer"

TRAY_GRASP_INIT_L = np.array([0.0721, 0.3942, 1.0116])  # fallback left ear

# Gripper pads that have PhysicsMeshCollisionAPI — kinematic, physically contact tray
RIGHT_PAD_L_PATH = f"{LEFT_GR}/right_pad"
RIGHT_PAD_R_PATH = f"{RIGHT_GR}/right_pad"

# ── Robot shift (toward right arm = -X in world coords) ───────────────────────
ROBOT_SHIFT_X = -0.15   # m — move entire robot 15cm in right-arm direction

# ── EG2 gripper axis convention (see gripper.py) ─────────────────────────────
# pad local X (col 0) = JAW direction  — fingers open/close along this axis
# pad local Y (col 1) = lateral axis
# pad local Z (col 2) = APPROACH direction (finger tips point this way)
#
# To clamp the tray ear VERTICALLY (top/bottom contact) and allow yaw freedom:
#   TARGET_JAW  = world -Z  → pad X points down; jaw closes from top & bottom
#   forward_weight = 0      → yaw unconstrained
#
# Placeholders for Y/forward used only when forward_weight > 0:
TARGET_UP_WORLD_L      = np.array([ 1.0,  0.0,  0.0])   # pad Y (not constrained for grasp)
TARGET_FORWARD_WORLD_L = np.array([ 0.0, -1.0,  0.0])   # approach from +Y (for reference)
TARGET_JAW_WORLD_L     = np.array([ 0.0,  0.0, -1.0])   # jaw = world -Z (vertical clamp)

TARGET_UP_WORLD_R      = np.array([ 1.0,  0.0,  0.0])
TARGET_FORWARD_WORLD_R = np.array([ 0.0, -1.0,  0.0])
TARGET_JAW_WORLD_R     = np.array([ 0.0,  0.0, -1.0])

# ── Handoff orientations: grippers point TOWARD EACH OTHER along X axis ──────
# Jaw still vertical (top/bottom clamp), forward constrained toward other arm.
TARGET_UP_WORLD_L_HANDOFF      = np.array([ 1.0,  0.0,  0.0])   # not used (axis_weight=0)
TARGET_FORWARD_WORLD_L_HANDOFF = np.array([-1.0,  0.0,  0.0])   # pad Z = world -X
TARGET_JAW_WORLD_L_HANDOFF     = np.array([ 0.0,  0.0, -1.0])   # jaw vertical
TARGET_UP_WORLD_R_HANDOFF      = np.array([ 1.0,  0.0,  0.0])
TARGET_FORWARD_WORLD_R_HANDOFF = np.array([ 1.0,  0.0,  0.0])   # pad Z = world +X
TARGET_JAW_WORLD_R_HANDOFF     = np.array([ 0.0,  0.0, -1.0])

# ── Approach geometry — left arm (matching original verified constants) ────────
PRE_Y_OFFSET    =  0.125
PICK_Y_OFFSET   = -0.008
GRASP_Z_OFFSET  = -0.007
APPROACH_STEPS  =  80     # IK waypoints for the Y-linear approach path
LIFT_Z         =  0.300

# ── Approach geometry — right arm at HANDOFF (approaches from -X side) ───────
# Right arm comes from the -X direction and approaches the right ear (at -X of handoff center)
R_HANDOFF_PRE_X_OFFSET  = -0.125  # pre: 125mm in -X from ear
R_HANDOFF_NEAR_X_OFFSET = -0.040  # near: 40mm in -X from ear

# ── Analytical contact model ───────────────────────────────────────────────────
PAD_FACE_DEPTH_M  = 0.028
K_ARM_Y           = 300.0
K_EAR_Y           = 3000.0
MU_STATIC         = 1.5
GRIP_FORCE_STOP_N = 3.0


# ── IK orientation weights ────────────────────────────────────────────────────
ORIENT_WEIGHT_FREE   = 0.07
ORIENT_WEIGHT_GRASP  = 0.12
ORIENT_WEIGHT_STRONG = 0.30   # approach path: enforce jaw+forward simultaneously

# ── Fixed handoff center (relative to robot geometry; computed at startup) ─────
# 120cm above floor (world Z=1.20) and 30cm in front of robot center axis (-Y from base)
HANDOFF_Z_ABS      = 1.20   # absolute world Z (120cm from floor)
HANDOFF_Y_OFFSET   = -0.30  # m offset from arm base center Y (toward work area)
HANDOFF_EAR_HALF   = 0.0775 # L arm pad offset from handoff center (+X) — original behavior
# Tray rotates ~90° about Z during transport: 18cm Y (at rest) becomes world X at handoff.
# From log: tray mesh center X at handoff = L_pad_X - 0.101 (L pad is 10.1cm in +X from tray center).
# R arm targets tray -X face: handoff_pad_L[X] - TRAY_L_PAD_TO_MESH_CENTER_X - TRAY_HALF_X_HANDOFF
TRAY_HALF_X_HANDOFF         = 0.090  # half of 18cm (tray Y at rest = tray X at handoff)
TRAY_L_PAD_TO_MESH_CENTER_X = 0.101  # observed: L pad is 10.1cm in +X from tray mesh center at handoff
# Legacy alias kept for bbox-verification log
TRAY_HALF_X = TRAY_HALF_X_HANDOFF

# Dryer delivery: arm delivers tray pointing in dryer direction; capped to reachable distance
DRYER_REACH_LIMIT  = 0.55   # m max pad displacement from arm base
DRYER_HOLD_OFFSET  = 0.10   # m extra clearance away from dryer front

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_FRAMES         = 120
MOTION_FRAMES         = 90
FINE_FRAMES           = 120
CLOSE_PHYSICS_FRAMES  = 80
HOLD_FRAMES           = 60
PAUSE_FRAMES          = 90
CARRY_FRAMES          = 90
DRYER_HOLD_FRAMES     = 90
RESET_FRAMES          = 90

APPROACH_OPEN_RAD = 0.50          # gripper opening during approach (not max)
HOME_OPEN_RAD     = 0.6241        # full open → 85.4 mm gap (return-to-zero position)

# ── UI ────────────────────────────────────────────────────────────────────────
HIST_LEN    = 350
UI_EVERY    = 3
FORCE_MAX_N = 20.0
LIFT_MAX_M  = LIFT_Z + 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _rotation_to_euler_deg(R: np.ndarray) -> np.ndarray:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        ry = math.degrees(math.atan2(-R[2, 0], sy))
        rz = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    else:
        rx = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
        ry = math.degrees(math.atan2(-R[2, 0], sy))
        rz = 0.0
    return np.array([rx, ry, rz])


def _solve_ear_contact(drive_y: float, contact_y: float):
    if drive_y >= contact_y:
        return drive_y, 0.0, 0.0
    press_depth = contact_y - drive_y
    actual_y    = drive_y + K_EAR_Y * press_depth / (K_ARM_Y + K_EAR_Y)
    press_mm    = (contact_y - actual_y) * 1000.0
    F_N         = K_ARM_Y * (actual_y - drive_y)
    return actual_y, press_mm, 2.0 * MU_STATIC * F_N


# ─────────────────────────────────────────────────────────────────────────────
# Scene helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_robot_shift(stage, shift_x: float):
    """Shift /World/robot translate by shift_x (in meters)."""
    from pxr import Gf, UsdGeom
    prim = stage.GetPrimAtPath(ROBOT_PARENT_PATH)
    if not prim or not prim.IsValid():
        print(f"[TGC] WARNING: {ROBOT_PARENT_PATH} not found — robot shift skipped", flush=True)
        return
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        name = op.GetName()
        if "translate" in name and "unitsResolve" not in name and "world" not in name.lower():
            v = op.Get()
            op.Set(Gf.Vec3d(float(v[0]) + shift_x, float(v[1]), float(v[2])))
            print(f"[TGC] Robot shifted {shift_x:+.3f}m X → {float(v[0])+shift_x:.3f}", flush=True)
            return
    print("[TGC] WARNING: no suitable translate op on robot parent — shift skipped", flush=True)


def _get_tray_translate(stage) -> np.ndarray:
    v = stage.GetPrimAtPath(TRAY_PATH).GetAttribute("xformOp:translate").Get()
    return np.array([float(v[0]), float(v[1]), float(v[2])])


def _teleport_tray(stage, xyz: np.ndarray):
    """Set tray translate directly (call after making it kinematic)."""
    from pxr import Gf
    stage.GetPrimAtPath(TRAY_PATH).GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    )


def _set_tray_kinematic(stage, kinematic: bool):
    """Toggle tray between kinematic (for teleport) and dynamic (for physics)."""
    from pxr import UsdPhysics, Gf
    prim = stage.GetPrimAtPath(TRAY_PATH)
    rb_api = UsdPhysics.RigidBodyAPI(prim)
    if not rb_api:
        return
    rb_api.GetKinematicEnabledAttr().Set(kinematic)
    if not kinematic:
        # Zero out accumulated velocity so it doesn't fly off on re-enable
        for attr_name, zero in (("physics:velocity", Gf.Vec3f(0, 0, 0)),
                                 ("physics:angularVelocity", Gf.Vec3f(0, 0, 0))):
            attr = prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Set(zero)


def _get_tray_world_T(stage) -> np.ndarray:
    """Read tray 4×4 world transform via XformCache (handles Euler ops AND orient ops)."""
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(TRAY_PATH)
    # Fresh cache each call so PhysX writebacks are always visible.
    m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    # USD GfMatrix4d is row-major (row-vector convention, translation in row 3).
    # Transpose into column-vector convention (translation in column 3).
    return np.array([[float(m[i][j]) for j in range(4)] for i in range(4)]).T


def _set_tray_world_transform(stage, T: np.ndarray):
    """Set tray translate + orient from 4×4 world transform (call while kinematic)."""
    from pxr import Gf
    from scipy.spatial.transform import Rotation
    pos = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # [x, y, z, w]
    prim = stage.GetPrimAtPath(TRAY_PATH)
    prim.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    o_attr = prim.GetAttribute("xformOp:orient")
    if o_attr and o_attr.IsValid():
        o_attr.Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))



def _get_dryer_world_pos(stage) -> np.ndarray | None:
    """Read dryer world position from USD (uses :world op if present)."""
    prim = stage.GetPrimAtPath(DRYER_PATH)
    if not prim or not prim.IsValid():
        return None
    from pxr import UsdGeom, Usd
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    T = cache.GetLocalToWorldTransform(prim)
    # pxr row-major: translation in row 3
    return np.array([float(T[3][0]), float(T[3][1]), float(T[3][2])])


def _remove_prims(stage, *paths):
    for p in paths:
        if stage.GetPrimAtPath(p):
            stage.RemovePrim(p)


def _create_grasp_joint(stage, pad_path: str, joint_path: str):
    """FixedJoint from kinematic pad → dynamic tray, locking current relative 6-DOF pose.

    Call ONLY while the tray is kinematic so the initial constraint is exactly
    satisfied and PhysX applies zero impulse when tray goes dynamic.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Usd
    from scipy.spatial.transform import Rotation

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    T_pad  = get_world_pose(stage, cache, pad_path)
    T_tray = _get_tray_world_T(stage)

    # Express pad's world pose in tray's local frame
    T_local1 = np.linalg.inv(T_tray) @ T_pad
    pos1 = T_local1[:3, 3]
    q1   = Rotation.from_matrix(T_local1[:3, :3]).as_quat()  # [x,y,z,w]

    _remove_prims(stage, joint_path)  # clear any stale prim
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(pad_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(TRAY_PATH)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2])))
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)

    print(
        f"[TGC] FixedJoint {joint_path}:"
        f" pad={np.round(T_pad[:3,3],3)} tray={np.round(T_tray[:3,3],3)}"
        f" local1_pos={np.round(pos1,4)}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Arm FK ops
# ─────────────────────────────────────────────────────────────────────────────

def _setup_arm_ops(stage, root, suffix):
    from pxr import UsdGeom
    ops = {}
    for link in LINK_NAMES:
        prim = stage.GetPrimAtPath(f"{root}/{link}")
        if prim and prim.IsValid():
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            ops[link] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    return ops


def _set_arm_q(ops, chains, q: np.ndarray):
    q_map = dict(zip(ARM_JOINTS, q.tolist()))
    for link, op in ops.items():
        op.Set(gf_matrix_from_column_transform(fk(chains[link], q_map)))


def _set_gripper_xform(gr_ops, angle_rad: float):
    for name, op in gr_ops.items():
        op.Set(gf_matrix_from_column_transform(gripper_link_transform(name, angle_rad)))


# ─────────────────────────────────────────────────────────────────────────────
# IK helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ik_fn(arm_chain, lower, upper, base_world, link6_to_pad,
                up_world, fwd_world, seeds,
                jaw_world=None, constrain_forward=False):
    """Build an IK closure.

    When jaw_world is given:
      - jaw_weight = orient_weight  (keep pad X = jaw_world)
      - axis_weight = 0             (pad Y unconstrained)
      - forward_weight = orient_weight if constrain_forward else 0
    When jaw_world is None (legacy):
      - axis_weight = forward_weight = orient_weight
      - jaw_weight = 0
    """
    def _ik(label, target, ref=None, orient_weight=ORIENT_WEIGHT_FREE):
        seed_list = ([] if ref is None else [ref]) + seeds
        if jaw_world is not None:
            q, pe, ue, fe, je, ok, msg = solve_pad_pose_ik(
                arm_chain, lower, upper, base_world, link6_to_pad,
                target, up_world, fwd_world,
                seed_list,
                axis_weight=0.0,
                forward_weight=orient_weight if constrain_forward else 0.0,
                continuity_weight=0.01 if ref is not None else 0.0,
                target_jaw_world=jaw_world,
                jaw_weight=orient_weight,
            )
            print(
                f"[TGC]   IK {label} {np.round(target, 4)}: "
                f"pos={pe*1000:.1f}mm  jaw={je:.1f}°  fw={fe:.1f}°",
                flush=True,
            )
        else:
            q, pe, ue, fe, je, ok, msg = solve_pad_pose_ik(
                arm_chain, lower, upper, base_world, link6_to_pad,
                target, up_world, fwd_world,
                seed_list,
                axis_weight=orient_weight, forward_weight=orient_weight,
                continuity_weight=0.01 if ref is not None else 0.0,
            )
            print(
                f"[TGC]   IK {label} {np.round(target, 4)}: "
                f"pos={pe*1000:.1f}mm  up={ue:.1f}°  fw={fe:.1f}°",
                flush=True,
            )
        return q
    return _ik


def _linear_y_path(ik_fn, q_start: np.ndarray,
                   start_xyz: np.ndarray, end_xyz: np.ndarray,
                   steps: int,
                   orient_weight: float = ORIENT_WEIGHT_FREE) -> np.ndarray:
    """Cartesian-linear path strictly along world Y. X and Z are held constant.

    At each waypoint the IK is solved using the previous solution as warm-start
    seed, so the resulting joint trajectory is smooth and monotone in Y.
    orient_weight is forwarded to ik_fn so jaw + forward constraints can be
    enforced at every step (use ORIENT_WEIGHT_STRONG for full alignment).
    """
    path = [q_start.copy()]
    prev = q_start.copy()
    for i in range(1, steps):
        alpha = i / (steps - 1)
        y = start_xyz[1] + alpha * (end_xyz[1] - start_xyz[1])
        target = np.array([start_xyz[0], y, start_xyz[2]])
        q = ik_fn("ly", target, ref=prev, orient_weight=orient_weight)
        path.append(q)
        prev = q
    return np.array(path)


# ─────────────────────────────────────────────────────────────────────────────
# cuRobo planning via subprocess (avoids Isaac Sim warp 1.8.2 conflict)
# ─────────────────────────────────────────────────────────────────────────────

_plan_cache: dict = {}
_WORKER_PY    = str(Path(__file__).parent / "_curobo_worker.py")
_MINICONDA_PY = _os.environ.get("CUROBO_PYTHON", "/home/andyee/miniconda3/bin/python3")


def _q_key(q: np.ndarray):
    return tuple(round(float(x), 3) for x in q)


def _cu_batch(jobs: list) -> dict:
    """Run a list of {q_start, q_goal, label, max_attempts} jobs via subprocess.
    Returns {label: np.ndarray path}.
    """
    import subprocess, json as _json
    payload = _json.dumps({"jobs": [
        {"label": j["label"],
         "q_start": [float(x) for x in j["q_start"]],
         "q_goal":  [float(x) for x in j["q_goal"]],
         "max_attempts": j.get("max_attempts", 8)}
        for j in jobs
    ]})
    try:
        proc = subprocess.run(
            [_MINICONDA_PY, _WORKER_PY],
            input=payload, capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            print(f"[TGC] cuRobo worker exited {proc.returncode}:\n{proc.stderr[-600:]}", flush=True)
            return {}
        if proc.stderr:
            print(proc.stderr, flush=True, end="")
        data = _json.loads(proc.stdout)
        out = {}
        for r in data["results"]:
            lbl = r["label"]
            raw = r.get("path")
            out[lbl] = np.array(raw) if raw is not None else None
        return out
    except Exception as _e:
        print(f"[TGC] cuRobo batch error: {_e}", flush=True)
        return {}


def _cu_plan(planner, q_start: np.ndarray, q_goal: np.ndarray,
             steps: int, label: str = "") -> np.ndarray:
    """Plan q_start→q_goal via cuRobo subprocess (max_attempts=8).
    Caches by (q_start, q_goal). On failure: single-frame jump (no interpolation).
    planner arg kept for API compatibility but unused (subprocess owns the planner).
    """
    key = (_q_key(q_start), _q_key(q_goal))
    if key in _plan_cache:
        return _plan_cache[key]

    results = _cu_batch([{"label": label, "q_start": q_start, "q_goal": q_goal}])
    path = results.get(label)

    if path is None or len(path) == 0:
        print(f"[TGC] ⚠ {label}: cuRobo unavailable — single-step jump", flush=True)
        path = np.array([q_goal])
    else:
        print(f"[TGC] cuRobo {label}: {len(path)} steps", flush=True)

    _plan_cache[key] = path
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Handoff geometry
# ─────────────────────────────────────────────────────────────────────────────

def _compute_handoff_center(l_base_world: np.ndarray, r_base_world: np.ndarray) -> np.ndarray:
    """Hardcoded handoff center: 120cm above floor, 30cm in front of arm base midpoint."""
    cx = (float(l_base_world[0, 3]) + float(r_base_world[0, 3])) / 2.0
    cy = (float(l_base_world[1, 3]) + float(r_base_world[1, 3])) / 2.0
    return np.array([cx, cy + HANDOFF_Y_OFFSET, HANDOFF_Z_ABS])


def _compute_dryer_approach(r_base_world: np.ndarray, dryer_pos: np.ndarray | None) -> np.ndarray:
    """Position reachable by right arm that points toward dryer."""
    rb = r_base_world[:3, 3]
    if dryer_pos is not None:
        direction = dryer_pos - rb
        dist      = float(np.linalg.norm(direction))
        if dist > 1e-3:
            direction = direction / dist
        target = rb + direction * min(dist - DRYER_HOLD_OFFSET, DRYER_REACH_LIMIT)
        print(f"[TGC] Dryer pos: {np.round(dryer_pos, 3)}  approach: {np.round(target, 3)}", flush=True)
        return target
    # fallback: a fixed offset in front-left of right arm
    return rb + np.array([-0.30, -0.40, -0.10])


# ─────────────────────────────────────────────────────────────────────────────
# UI — compact dual-arm handoff monitor
# ─────────────────────────────────────────────────────────────────────────────

class HandoffMonitorUI:
    """Dual-arm monitor for the full handoff cycle."""

    PHASE_COLORS = {
        "SETTLE":       0xFF888888,
        "PLAN":         0xFFFFAA00,
        "TO_PRE_L":     0xFF00AAFF,
        "APPROACH_L":   0xFF44DDFF,
        "CLOSE_GRIP_L": 0xFFFFFF44,
        "LIFT_L":       0xFF44FF44,
        "CARRY_L":      0xFF22BB44,
        "R_TO_NEAR":    0xFFFF8844,
        "R_APPROACH":   0xFFFFAA66,
        "CLOSE_GRIP_R": 0xFFFFDD44,
        "RELEASE_L":    0xFFFF6666,
        "RETRACT_L":    0xFF0088FF,
        "HOME_L":       0xFF006688,
        "CARRY_DRYER":  0xFFCC44FF,
        "HOLD_DRYER":   0xFF8844CC,
        "RELEASE_R":    0xFFFF9966,
        "HOME_R":       0xFF884400,
        "RESET_SCENE":  0xFF444488,
        "PAUSE":        0xFF444444,
    }

    def __init__(self):
        import omni.ui as ui
        self._ui   = ui
        self._tick = 0
        self._hist_fL = [0.0] * HIST_LEN
        self._hist_fR = [0.0] * HIST_LEN
        self._s = dict(
            phase="SETTLE", cycle=0,
            l_gr_deg=math.degrees(APPROACH_OPEN_RAD),
            r_gr_deg=math.degrees(APPROACH_OPEN_RAD),
            l_friction=0.0, r_friction=0.0,
            l_press_mm=0.0, r_press_mm=0.0,
            lift_m=0.0,
            l_pad_xyz=np.zeros(3),
            r_pad_xyz=np.zeros(3),
            l_contact=False, r_contact=False,
            tray_xyz=np.zeros(3), tray_ref=np.zeros(3),
        )
        self._win = ui.Window(
            "Handoff Monitor", width=620, height=680,
            flags=ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE,
        )
        self._rebuild()

    def push(self, **kwargs):
        self._s.update(kwargs)
        self._hist_fL = (self._hist_fL + [float(self._s.get("l_friction", 0))])[-HIST_LEN:]
        self._hist_fR = (self._hist_fR + [float(self._s.get("r_friction", 0))])[-HIST_LEN:]
        self._tick += 1
        if self._tick % UI_EVERY == 0:
            self._rebuild()

    def _bar(self, val, max_val, color=0xFF44FF88, h=7):
        ui = self._ui
        pct = min(100.0, max(0.0, val / max(1e-9, max_val)) * 100.0)
        with ui.ZStack(height=h):
            ui.Rectangle(style={"background_color": 0xFF1A1A1A, "border_radius": 2})
            if pct > 0:
                with ui.HStack():
                    ui.Rectangle(width=ui.Percent(pct),
                                 style={"background_color": color, "border_radius": 2})
                    ui.Spacer()

    def _arm_row(self, label, color, pad_xyz, gr_deg, friction, press_mm, contact):
        ui = self._ui
        gap_mm = pad_separation_m(math.radians(gr_deg)) * 1000
        px, py, pz = float(pad_xyz[0]), float(pad_xyz[1]), float(pad_xyz[2])
        with ui.HStack(height=16):
            ui.Label(f"{label}  pad [{px:+.3f} {py:+.3f} {pz:+.3f}]",
                     style={"font_size": 10, "color": color})
            ui.Spacer()
            if contact:
                ui.Label(f"F={friction:.2f}N  p={press_mm:.1f}mm",
                         style={"font_size": 9, "color": 0xFFFFDD44})
            else:
                ui.Label(f"gr={gr_deg:.1f}°  gap={gap_mm:.1f}mm",
                         style={"font_size": 9, "color": 0xFF888888})
        if friction > 0.05:
            self._bar(friction, FORCE_MAX_N,
                      color=0xFFFF6644 if friction > GRIP_FORCE_STOP_N else 0xFF44AAFF)

    def _rebuild(self):
        ui = self._ui
        s  = self._s
        phase_col = self.PHASE_COLORS.get(s["phase"], 0xFFAAAAAA)
        tray_delta = np.linalg.norm(s["tray_xyz"] - s["tray_ref"]) * 1000

        self._win.frame.clear()
        with self._win.frame:
            with ui.VStack(spacing=2):
                # Header
                with ui.HStack(height=26):
                    ui.Label("DUAL-ARM HANDOFF", style={"font_size": 14, "color": 0xFFFFFFFF})
                    ui.Spacer()
                    ui.Label(f"● {s['phase']}  #{s['cycle']}",
                             style={"font_size": 13, "color": phase_col})
                ui.Separator(height=2)

                # Tray row
                with ui.HStack(height=16):
                    tx, ty, tz = float(s["tray_xyz"][0]), float(s["tray_xyz"][1]), float(s["tray_xyz"][2])
                    ui.Label(f"TRAY [{tx:.3f} {ty:.3f} {tz:.3f}]",
                             style={"font_size": 10, "color": 0xFFCCCCCC})
                    ui.Spacer()
                    col = 0xFF44FF44 if tray_delta < 8 else (0xFFFFDD44 if tray_delta < 40 else 0xFFFF4444)
                    ui.Label(f"Δ={tray_delta:.1f}mm", style={"font_size": 9, "color": col})
                ui.Separator(height=2)

                # Arms
                self._arm_row("L", 0xFF88DDFF,
                              s["l_pad_xyz"], s["l_gr_deg"],
                              s["l_friction"], s["l_press_mm"], s["l_contact"])
                self._arm_row("R", 0xFFFFAA44,
                              s["r_pad_xyz"], s["r_gr_deg"],
                              s["r_friction"], s["r_press_mm"], s["r_contact"])
                ui.Separator(height=2)

                # Lift bar
                with ui.HStack(height=16):
                    ui.Label(f"Lift  {s['lift_m']*100:.1f} / {LIFT_Z*100:.0f} cm",
                             style={"font_size": 12})
                self._bar(s["lift_m"], LIFT_MAX_M, color=0xFF44FF88)
                ui.Separator(height=2)

                # Force history
                ui.Label("Left friction (N) history",
                         style={"font_size": 9, "color": 0xFF666666}, height=12)
                ui.Plot(ui.Type.LINE, 0.0, FORCE_MAX_N, *self._hist_fL, height=55,
                        style={"color": 0xFF22AAFF, "background_color": 0xFF050510})
                ui.Label("Right friction (N) history",
                         style={"font_size": 9, "color": 0xFF666666}, height=12)
                ui.Plot(ui.Type.LINE, 0.0, FORCE_MAX_N, *self._hist_fR, height=55,
                        style={"color": 0xFFFF8844, "background_color": 0xFF100505})
                ui.Separator(height=2)

                ui.Label(
                    f"K_arm={K_ARM_Y:.0f}  K_ear={K_EAR_Y:.0f}  μ={MU_STATIC}  "
                    f"stop@{GRIP_FORCE_STOP_N}N  depth={PAD_FACE_DEPTH_M*1000:.0f}mm",
                    style={"font_size": 8, "color": 0xFF444444}, height=13,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Main run loop
# ─────────────────────────────────────────────────────────────────────────────

def _run(app, stage):
    import omni.timeline
    from pxr import Usd, UsdGeom

    # ── Remove stale prims from previous runs BEFORE physics starts ───────────
    # This must happen before timeline.play() so PhysX never simulates these.
    # The old _GraspLock_L joint causes the tray to float above the equipment.
    _remove_prims(stage, JOINT_PATH_L, CARRIER_ROOT_L, JOINT_PATH_R, CARRIER_ROOT_R,
                  RESET_JOINT, RESET_CARRIER)
    try:
        import omni.usd as _ousd
        _ousd.get_context().save_stage()
        print("[TGC] Cleaned stage saved to disk.", flush=True)
    except Exception as _e:
        print(f"[TGC] Stage save skipped: {_e}", flush=True)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.set_end_time(99999.0)
    timeline.set_looping(False)

    # ── URDF / kinematic chains ───────────────────────────────────────────────
    print("[TGC] Loading URDF…", flush=True)
    arm_jts   = load_joints(Path(DEFAULT_ARM_URDF))
    chains    = {n: chain_to_link(arm_jts, "Link_0", n) for n in LINK_NAMES}
    arm_chain = chains["Link_6"]
    lower, upper = joint_limits(arm_chain)
    q_zero = np.zeros(6)
    print("[TGC] URDF loaded.", flush=True)

    # ── Apply robot shift before setting up ops ───────────────────────────────
    _apply_robot_shift(stage, ROBOT_SHIFT_X)

    # ── Arm xform ops (both arms, both grippers) ─────────────────────────────
    l_ops    = _setup_arm_ops(stage, LEFT_ROOT,  "tgc_arm")
    r_ops    = _setup_arm_ops(stage, RIGHT_ROOT, "tgc_arm")
    l_gr_ops = setup_gripper_xform_ops(stage, LEFT_GR,  "tgc_gr")
    r_gr_ops = setup_gripper_xform_ops(stage, RIGHT_GR, "tgc_gr")

    _set_arm_q(l_ops, chains, q_zero)
    _set_arm_q(r_ops, chains, q_zero)
    _set_gripper_xform(l_gr_ops, HOME_OPEN_RAD)
    _set_gripper_xform(r_gr_ops, HOME_OPEN_RAD)
    app.update()

    # ── Read arm bases after shift ────────────────────────────────────────────
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    l_base_world, _l_pad_home, l_link6_to_pad = selected_pad_midpoint(stage, cache, "left")
    r_base_world, _r_pad_home, r_link6_to_pad = selected_pad_midpoint(stage, cache, "right")
    print(f"[TGC] L base: {np.round(l_base_world[:3, 3], 4)}", flush=True)
    print(f"[TGC] R base: {np.round(r_base_world[:3, 3], 4)}", flush=True)

    def _pad_L(q): return pad_world_transform(arm_chain, l_base_world, l_link6_to_pad, q)
    def _pad_R(q): return pad_world_transform(arm_chain, r_base_world, r_link6_to_pad, q)

    # ── Handoff / dryer geometry ──────────────────────────────────────────────
    handoff_center = _compute_handoff_center(l_base_world, r_base_world)
    # L arm: carries tray to handoff_pad_L (original behavior, unchanged).
    # Tray rotates ~90° about Z in transit → 18cm (Y at rest) becomes world X at handoff.
    # Observed from log: tray mesh center X = L_pad_X - TRAY_L_PAD_TO_MESH_CENTER_X (= -0.101).
    # R arm targets tray -X face: tray_center_X - TRAY_HALF_X_HANDOFF (tray left surface).
    handoff_pad_L  = handoff_center + np.array([HANDOFF_EAR_HALF, 0.0, 0.0])
    _tray_center_x_ho = handoff_pad_L[0] - TRAY_L_PAD_TO_MESH_CENTER_X
    handoff_pad_R  = np.array([_tray_center_x_ho - TRAY_HALF_X_HANDOFF,
                               handoff_pad_L[1],
                               handoff_pad_L[2]])
    print(f"[TGC] Handoff center: {np.round(handoff_center, 3)}", flush=True)
    print(f"[TGC]  L pad @handoff: {np.round(handoff_pad_L, 3)}", flush=True)
    print(f"[TGC]  Tray center X (predicted): {_tray_center_x_ho:.3f}  "
          f"(L_pad_X - {TRAY_L_PAD_TO_MESH_CENTER_X})", flush=True)
    print(f"[TGC]  R pad @handoff (tray -X face): {np.round(handoff_pad_R, 3)}", flush=True)
    print(f"[TGC]  L pad → tray center → R pad: {handoff_pad_L[0]:.3f} → "
          f"{_tray_center_x_ho:.3f} → {handoff_pad_R[0]:.3f}", flush=True)

    dryer_pos    = _get_dryer_world_pos(stage)
    dryer_target = _compute_dryer_approach(r_base_world, dryer_pos)
    print(f"[TGC] Dryer approach target: {np.round(dryer_target, 3)}", flush=True)

    # ── IK seed banks ────────────────────────────────────────────────────────
    l_seeds = [
        np.zeros(6),
        np.array([ 0.0,  0.30, -1.00,  0.0,  0.50,  0.0]),
        np.array([ 0.3,  0.50, -1.50,  0.0,  1.00,  0.3]),
        np.array([-0.3,  0.50, -1.50,  0.0,  1.00, -0.3]),
        np.array([ 0.0,  1.00, -2.00,  0.0,  1.50,  0.0]),
        np.array([ 0.1,  0.80, -1.80,  0.1,  1.20,  0.1]),
    ]
    r_seeds = [
        np.zeros(6),
        np.array([1.02, -0.74, -0.82, 1.22, -1.23,  0.57]),
        np.array([0.19,  0.66, -0.99, 0.63, -0.19,  0.64]),
        np.array([1.00,  0.46, -0.72, -0.26, -0.61,  0.34]),
        np.array([1.77,  1.38, -0.67, -0.14, -1.49, -1.70]),
        np.array([0.0,   0.30, -1.00,  0.0,   0.50,  0.0]),
    ]

    # Grasp IK: jaw vertical (pad X = world -Z) + forward along -Y (pad Z = world -Y).
    # ORIENT_WEIGHT_STRONG=0.30 enforces both axes throughout the approach path.
    _ik_L = _make_ik_fn(arm_chain, lower, upper, l_base_world, l_link6_to_pad,
                        TARGET_UP_WORLD_L, TARGET_FORWARD_WORLD_L, l_seeds,
                        jaw_world=TARGET_JAW_WORLD_L, constrain_forward=True)
    _ik_R = _make_ik_fn(arm_chain, lower, upper, r_base_world, r_link6_to_pad,
                        TARGET_UP_WORLD_R, TARGET_FORWARD_WORLD_R, r_seeds,
                        jaw_world=TARGET_JAW_WORLD_R, constrain_forward=False)
    # Handoff IK: jaw vertical + forward constrained (gripper points toward other arm)
    _ik_L_ho = _make_ik_fn(arm_chain, lower, upper, l_base_world, l_link6_to_pad,
                            TARGET_UP_WORLD_L_HANDOFF, TARGET_FORWARD_WORLD_L_HANDOFF, l_seeds,
                            jaw_world=TARGET_JAW_WORLD_L_HANDOFF, constrain_forward=True)
    _ik_R_ho = _make_ik_fn(arm_chain, lower, upper, r_base_world, r_link6_to_pad,
                            TARGET_UP_WORLD_R_HANDOFF, TARGET_FORWARD_WORLD_R_HANDOFF, r_seeds,
                            jaw_world=TARGET_JAW_WORLD_R_HANDOFF, constrain_forward=True)

    # ── Create UI BEFORE physics ──────────────────────────────────────────────
    ui_mon = HandoffMonitorUI()
    print("[TGC] UI created.", flush=True)
    app.update()

    # ── Start physics ─────────────────────────────────────────────────────────
    print("[TGC] Starting timeline…", flush=True)
    timeline.play()
    app.update()

    # ── SETTLE ────────────────────────────────────────────────────────────────
    frame = 0
    for frame in range(SETTLE_FRAMES):
        _set_arm_q(l_ops, chains, q_zero)
        _set_arm_q(r_ops, chains, q_zero)
        _set_gripper_xform(l_gr_ops, HOME_OPEN_RAD)
        _set_gripper_xform(r_gr_ops, HOME_OPEN_RAD)
        ui_mon.push(phase="SETTLE", cycle=0,
                    l_gr_deg=math.degrees(HOME_OPEN_RAD),
                    r_gr_deg=math.degrees(HOME_OPEN_RAD),
                    l_pad_xyz=_pad_L(q_zero)[:3, 3],
                    r_pad_xyz=_pad_R(q_zero)[:3, 3])
        app.update()
        if frame % 30 == 0:
            print(f"[TGC] settle {frame}/{SETTLE_FRAMES}", flush=True)
    frame = SETTLE_FRAMES

    # ── Record tray reference pos after settle ────────────────────────────────
    tray_xyz_ref = _get_tray_translate(stage).copy()
    tray_T_ref   = _get_tray_world_T(stage).copy()
    print(f"[TGC] Tray settled: {np.round(tray_xyz_ref, 4)}", flush=True)

    # ── Log tray bounding box so we can verify TRAY_HALF_X ───────────────────
    try:
        from pxr import UsdGeom, Usd as _Usd
        _bbox_cache = UsdGeom.BBoxCache(
            _Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
        _tray_prim = stage.GetPrimAtPath(TRAY_PATH)
        _bbox = _bbox_cache.ComputeWorldBound(_tray_prim)
        _box  = _bbox.GetBox()
        _bmin = np.array(_box.GetMin())
        _bmax = np.array(_box.GetMax())
        _dim  = _bmax - _bmin
        print(f"[TGC] TRAY BBOX  min={np.round(_bmin,3)}  max={np.round(_bmax,3)}", flush=True)
        print(f"[TGC] TRAY DIM   X={_dim[0]:.3f}m  Y={_dim[1]:.3f}m  Z={_dim[2]:.3f}m  "
              f"(half_X={_dim[0]/2:.3f}, half_Y={_dim[1]/2:.3f})", flush=True)
        # At handoff the tray rotates ~90° → Y at rest becomes X at handoff.
        # TRAY_HALF_X_HANDOFF (=0.090) = half of Y_dim (18cm), not X_dim (10cm).
        print(f"[TGC] At rest  half_X={_dim[0]/2:.3f}m  half_Y={_dim[1]/2:.3f}m  "
              f"TRAY_HALF_X_HANDOFF={TRAY_HALF_X_HANDOFF:.3f} (should ≈ half_Y)", flush=True)
    except Exception as _e:
        print(f"[TGC] Tray bbox error: {_e}", flush=True)

    # ── PLAN ──────────────────────────────────────────────────────────────────
    ui_mon.push(phase="PLAN", cycle=0)
    app.update()

    cache.Clear()

    # Read left ear
    ear_xyz_L = None
    gp_T = get_world_pose(stage, cache, GRASP_PRIM_L)
    if gp_T is not None and np.linalg.norm(gp_T[:3, 3]) > 0.05:
        ear_xyz_L = gp_T[:3, 3].copy()
    if ear_xyz_L is None:
        ear_xyz_L = TRAY_GRASP_INIT_L.copy()
    print(f"[TGC] Left ear: {np.round(ear_xyz_L, 4)}", flush=True)

    # Read right ear
    ear_xyz_R = None
    gp_R = get_world_pose(stage, cache, GRASP_PRIM_R)
    if gp_R is not None and np.linalg.norm(gp_R[:3, 3]) > 0.05:
        ear_xyz_R = gp_R[:3, 3].copy()
    if ear_xyz_R is None:
        # Both ears are at the same position on this tray
        ear_xyz_R = ear_xyz_L.copy()
    print(f"[TGC] Right ear: {np.round(ear_xyz_R, 4)}", flush=True)

    contact_y_L = float(ear_xyz_L[1]) + PAD_FACE_DEPTH_M
    # contact_y_R is set during IK planning (based on handoff position, not initial tray)

    # Pad X aligned with tray centre (midpoint of both ears in world X).
    tray_center_x = (float(ear_xyz_L[0]) + float(ear_xyz_R[0])) / 2.0
    gz_L      = float(ear_xyz_L[2]) + GRASP_Z_OFFSET
    pre_xyz_L  = np.array([tray_center_x, ear_xyz_L[1] + PRE_Y_OFFSET,  gz_L])
    pick_xyz_L = np.array([tray_center_x, ear_xyz_L[1] + PICK_Y_OFFSET, gz_L])
    print(
        f"[TGC] Tray centre X={tray_center_x:.4f}  "
        f"pre_Y={pre_xyz_L[1]:.4f}  pick_Y={pick_xyz_L[1]:.4f}",
        flush=True,
    )

    print("[TGC] IK planning (left arm)…", flush=True)
    q_pre_L  = _ik_L("L_pre",  pre_xyz_L,  orient_weight=ORIENT_WEIGHT_STRONG)
    app.update()

    # Linear-Y approach path: merge previous TO_NEAR_L + APPROACH_L into one
    # straight line in world Y (X = tray_center_x, Z = gz_L constant).
    print("[TGC] Building Y-linear approach path…", flush=True)
    path_approach_L = _linear_y_path(
        _ik_L, q_pre_L, pre_xyz_L, pick_xyz_L, APPROACH_STEPS,
        orient_weight=ORIENT_WEIGHT_STRONG,
    )
    q_pick_L = path_approach_L[-1]
    print(
        f"[TGC] Approach path: {len(path_approach_L)} steps  "
        f"end_pad_Y={_pad_L(q_pick_L)[1,3]:.4f}",
        flush=True,
    )
    app.update()
    # L handoff: gripper points toward right arm (-X direction)
    q_handoff_L = _ik_L_ho("L_handoff", handoff_pad_L,
                            ref=q_pick_L, orient_weight=ORIENT_WEIGHT_GRASP)
    print("[TGC] IK planning (right arm)…", flush=True)
    app.update()
    # R arm approaches the right ear at handoff from the -X side (gripper points +X toward L arm).
    # Approach offsets in X instead of Y.
    near_handoff_xyz_R = handoff_pad_R + np.array([R_HANDOFF_NEAR_X_OFFSET, 0.0, 0.0])
    pre_handoff_xyz_R  = handoff_pad_R + np.array([R_HANDOFF_PRE_X_OFFSET,  0.0, 0.0])
    q_handoff_R      = _ik_R_ho("R_handoff",      handoff_pad_R,      orient_weight=ORIENT_WEIGHT_GRASP)
    q_near_handoff_R = _ik_R_ho("R_near_handoff", near_handoff_xyz_R, ref=q_handoff_R, orient_weight=ORIENT_WEIGHT_GRASP)
    q_pre_handoff_R  = _ik_R_ho("R_pre_handoff",  pre_handoff_xyz_R,  ref=q_near_handoff_R)
    q_dryer_R        = _ik_R("R_dryer",           dryer_target,       orient_weight=ORIENT_WEIGHT_FREE)
    contact_y_R      = float(handoff_pad_R[1]) + PAD_FACE_DEPTH_M
    print("[TGC] IK done.", flush=True)
    app.update()

    # ── Precompute all fixed paths via cuRobo subprocess ─────────────────────
    # cuRobo runs in miniconda python3 to avoid Isaac Sim's Warp 1.8.2 conflict.
    print("[TGC] Planning fixed paths with cuRobo (subprocess)…", flush=True)
    ui_mon.push(phase="PLAN", cycle=0)
    app.update()
    _fixed_jobs = [
        {"label": "home→preL",      "q_start": q_zero,           "q_goal": q_pre_L},
        {"label": "retractL",       "q_start": q_handoff_L,      "q_goal": q_zero},
        {"label": "home→preR",      "q_start": q_zero,           "q_goal": q_pre_handoff_R},
        {"label": "pre→nearR",      "q_start": q_pre_handoff_R,  "q_goal": q_near_handoff_R},
        {"label": "approachR",      "q_start": q_near_handoff_R, "q_goal": q_handoff_R},
        {"label": "handoff→dryer",  "q_start": q_handoff_R,      "q_goal": q_dryer_R},
        {"label": "dryer→homeR",    "q_start": q_dryer_R,        "q_goal": q_zero},
    ]
    _fixed_results = _cu_batch(_fixed_jobs)

    def _get_path(label: str, q_goal: np.ndarray) -> np.ndarray:
        p = _fixed_results.get(label)
        if p is None or len(p) == 0:
            print(f"[TGC] ⚠ {label}: cuRobo failed — single-step jump", flush=True)
            return np.array([q_goal])
        key = (_q_key(p[0]), _q_key(q_goal))
        _plan_cache[key] = p
        print(f"[TGC] cuRobo {label}: {len(p)} steps", flush=True)
        return p

    path_home_to_pre_L    = _get_path("home→preL",     q_pre_L)
    # path_approach_L already built as Y-linear Cartesian path above
    path_carry_L          = np.array([q_handoff_L])   # placeholder; rebuilt on contact
    path_retract_L        = _get_path("retractL",      q_zero)
    path_home_to_pre_R    = _get_path("home→preR",     q_pre_handoff_R)
    path_pre_to_near_R    = _get_path("pre→nearR",     q_near_handoff_R)
    path_approach_R       = _get_path("approachR",     q_handoff_R)
    path_handoff_to_dryer = _get_path("handoff→dryer", q_dryer_R)
    path_dryer_home_R     = _get_path("dryer→homeR",   q_zero)
    print("[TGC] Fixed path planning done.", flush=True)
    app.update()

    # ── Grasp state ───────────────────────────────────────────────────────────
    joint_L_active   = False
    joint_R_active   = False
    pad_z0           = None
    q_contact_L      = None
    q_contact_R      = None
    contact_est_L    = False
    contact_est_R    = False
    q_lift_end_L     = q_pick_L.copy()         # placeholder; set on contact
    path_lift_L      = np.array([q_pick_L])    # placeholder; rebuilt on contact

    grip_close = np.linspace(APPROACH_OPEN_RAD, 0.0,           CLOSE_PHYSICS_FRAMES)
    grip_open  = np.linspace(0.0,               HOME_OPEN_RAD, CLOSE_PHYSICS_FRAMES // 2)

    q_L       = q_zero.copy()
    q_R       = q_zero.copy()
    gr_angle_L = HOME_OPEN_RAD
    gr_angle_R = HOME_OPEN_RAD
    path_idx  = 0
    cycle     = 1
    phase     = "TO_PRE_L"

    def _enter(new_phase):
        nonlocal phase, path_idx
        phase    = new_phase
        path_idx = 0
        print(f"[TGC] → {new_phase}  (cycle {cycle}  fr {frame})", flush=True)

    _enter("TO_PRE_L")

    while app.is_running():

        # ── State machine ─────────────────────────────────────────────────────

        if phase == "TO_PRE_L":
            if path_idx < len(path_home_to_pre_L):
                q_L = path_home_to_pre_L[path_idx]; path_idx += 1
            else:
                _enter("APPROACH_L")

        elif phase == "APPROACH_L":
            if path_idx < len(path_approach_L):
                next_q  = path_approach_L[path_idx]
                test_pw = _pad_L(next_q)
                _, _, test_f = _solve_ear_contact(float(test_pw[1, 3]), contact_y_L)
                if test_f >= GRIP_FORCE_STOP_N and path_idx > 0:
                    q_contact_L = q_L.copy()
                    cur_pad     = _pad_L(q_contact_L)[:3, 3]
                    # Solve lift-end IK → plan lift + carry with cuRobo
                    lift_target  = cur_pad.copy(); lift_target[2] += LIFT_Z
                    q_lift_end_L = _ik_L("L_lift", lift_target, ref=q_contact_L,
                                         orient_weight=ORIENT_WEIGHT_GRASP)
                    print(f"[TGC] Lift IK target={np.round(lift_target, 3)}", flush=True)
                    _lc = _cu_batch([
                        {"label": "liftL",  "q_start": q_contact_L, "q_goal": q_lift_end_L},
                        {"label": "carryL", "q_start": q_lift_end_L, "q_goal": q_handoff_L},
                    ])
                    _p = _lc.get("liftL");  path_lift_L  = _p if (_p is not None and len(_p) > 0) else np.array([q_lift_end_L])
                    _p = _lc.get("carryL"); path_carry_L = _p if (_p is not None and len(_p) > 0) else np.array([q_handoff_L])

                    pad_z0        = float(cur_pad[2])
                    contact_est_L = True
                    _set_tray_kinematic(stage, True)  # pin tray while pad closes
                    print(
                        f"[TGC] L FORCE STOP F={test_f:.2f}N idx={path_idx}"
                        f" pad={np.round(cur_pad, 4)}",
                        flush=True,
                    )
                    _enter("CLOSE_GRIP_L")
                else:
                    q_L = next_q
                    path_idx += 1
            else:
                if not contact_est_L:
                    print("[TGC] APPROACH_L exhausted without contact", flush=True)
                _enter("CLOSE_GRIP_L")

        elif phase == "CLOSE_GRIP_L":
            if path_idx < len(grip_close):
                gr_angle_L = float(grip_close[path_idx]); path_idx += 1
            else:
                gr_angle_L = 0.0
                if not joint_L_active:
                    # No-contact fallback: compute paths and pin tray
                    if q_contact_L is None:
                        q_contact_L = q_L.copy()
                        cur_pad     = _pad_L(q_contact_L)[:3, 3]
                        lift_target  = cur_pad.copy(); lift_target[2] += LIFT_Z
                        q_lift_end_L = _ik_L("L_lift_nc", lift_target, ref=q_contact_L,
                                             orient_weight=ORIENT_WEIGHT_GRASP)
                        _lc_nc = _cu_batch([
                            {"label": "liftL_nc",  "q_start": q_contact_L, "q_goal": q_lift_end_L},
                            {"label": "carryL_nc", "q_start": q_lift_end_L, "q_goal": q_handoff_L},
                        ])
                        _p = _lc_nc.get("liftL_nc");  path_lift_L  = _p if (_p is not None and len(_p) > 0) else np.array([q_lift_end_L])
                        _p = _lc_nc.get("carryL_nc"); path_carry_L = _p if (_p is not None and len(_p) > 0) else np.array([q_handoff_L])
                        pad_z0         = float(cur_pad[2])
                        _set_tray_kinematic(stage, True)
                        print("[TGC] L grasp (no-contact fallback)", flush=True)
                    # Create FixedJoint while tray is kinematic → zero impulse at activation
                    _create_grasp_joint(stage, RIGHT_PAD_L_PATH, JOINT_PATH_L)
                    _set_tray_kinematic(stage, False)
                    joint_L_active = True
                _enter("LIFT_L")

        elif phase == "LIFT_L":
            if path_idx < len(path_lift_L):
                q_L = path_lift_L[path_idx]; path_idx += 1
            else:
                _enter("CARRY_L")

        elif phase == "CARRY_L":
            if path_idx < len(path_carry_L):
                q_L = path_carry_L[path_idx]; path_idx += 1
            else:
                # Both arms at handoff position — right arm moves to its side
                _enter("R_TO_NEAR")

        elif phase == "R_TO_NEAR":
            gr_angle_R = HOME_OPEN_RAD  # keep fully open during transit
            total_R = len(path_home_to_pre_R) + len(path_pre_to_near_R)
            if path_idx < len(path_home_to_pre_R):
                q_R = path_home_to_pre_R[path_idx]; path_idx += 1
            elif path_idx < total_R:
                q_R = path_pre_to_near_R[path_idx - len(path_home_to_pre_R)]; path_idx += 1
            else:
                _enter("R_APPROACH")

        elif phase == "R_APPROACH":
            gr_angle_R = HOME_OPEN_RAD  # keep fully open until closed by CLOSE_GRIP_R
            # No force detection: handoff position is known/fixed, just execute path.
            if path_idx < len(path_approach_R):
                q_R = path_approach_R[path_idx]; path_idx += 1
            else:
                _enter("CLOSE_GRIP_R")

        elif phase == "CLOSE_GRIP_R":
            if path_idx < len(grip_close):
                gr_angle_R = float(grip_close[path_idx]); path_idx += 1
            else:
                gr_angle_R = 0.0
                if not joint_R_active:
                    # Log geometry at grasp moment before creating joint
                    from pxr import UsdGeom as _UG, Usd as _U2
                    _gc = _UG.XformCache(_U2.TimeCode.Default())
                    _rpad_T = get_world_pose(stage, _gc, RIGHT_PAD_R_PATH)
                    _tray_T = _get_tray_world_T(stage)
                    _rpad_pos = _rpad_T[:3, 3] if _rpad_T is not None else np.zeros(3)
                    _tray_pos = _tray_T[:3, 3]
                    _offset   = _tray_pos - _rpad_pos
                    print(f"[TGC-GEO] cy={cycle} R pad  : {np.round(_rpad_pos, 4)}", flush=True)
                    print(f"[TGC-GEO] cy={cycle} Tray   : {np.round(_tray_pos, 4)}", flush=True)
                    print(f"[TGC-GEO] cy={cycle} Offset (tray-Rpad): "
                          f"X={_offset[0]:.4f}  Y={_offset[1]:.4f}  Z={_offset[2]:.4f}  "
                          f"dist={np.linalg.norm(_offset):.4f}m", flush=True)
                    print(f"[TGC-GEO] cy={cycle} handoff_pad_R={np.round(handoff_pad_R,4)}"
                          f"  handoff_pad_L={np.round(handoff_pad_L,4)}", flush=True)
                    # Live tray world AABB at handoff moment
                    try:
                        from pxr import UsdGeom as _UG2, Usd as _U3
                        _bbox_c = _UG2.BBoxCache(
                            _U3.TimeCode.Default(),
                            includedPurposes=[_UG2.Tokens.default_])
                        _tp = stage.GetPrimAtPath(TRAY_PATH)
                        _b  = _bbox_c.ComputeWorldBound(_tp).GetBox()
                        _bmin = np.array(_b.GetMin())
                        _bmax = np.array(_b.GetMax())
                        _ctr  = (_bmin + _bmax) / 2
                        _dim  = _bmax - _bmin
                        print(f"[TGC-BBOX] cy={cycle} tray AABB min={np.round(_bmin,3)}  "
                              f"max={np.round(_bmax,3)}", flush=True)
                        print(f"[TGC-BBOX] cy={cycle} center={np.round(_ctr,3)}  "
                              f"dim X={_dim[0]:.3f}  Y={_dim[1]:.3f}  Z={_dim[2]:.3f}", flush=True)
                        print(f"[TGC-BBOX] cy={cycle} R_pad_X={_rpad_pos[0]:.4f}  "
                              f"tray_Xmin={_bmin[0]:.4f}  tray_Xcenter={_ctr[0]:.4f}  "
                              f"err={_rpad_pos[0]-_bmin[0]:.4f}m (+=inside)", flush=True)
                    except Exception as _be:
                        print(f"[TGC-BBOX] bbox error: {_be}", flush=True)
                    # Create FixedJoint (tray still held by joint_L → safe)
                    _create_grasp_joint(stage, RIGHT_PAD_R_PATH, JOINT_PATH_R)
                    joint_R_active = True
                    contact_est_R  = True
                    print("[TGC] R force grasp → FixedJoint_R created", flush=True)
                _enter("RELEASE_L")

        elif phase == "RELEASE_L":
            if path_idx < len(grip_open):
                gr_angle_L = float(grip_open[path_idx]); path_idx += 1
            else:
                gr_angle_L     = APPROACH_OPEN_RAD
                joint_L_active = False
                pad_z0         = None
                contact_est_L  = False
                q_contact_L    = None
                _remove_prims(stage, JOINT_PATH_L)  # release L FixedJoint → R holds tray
                _tray_after = _get_tray_translate(stage)
                print(f"[TGC] L released → tray held by R FixedJoint  "
                      f"tray_pos={np.round(_tray_after, 4)}", flush=True)
                _enter("RETRACT_L")

        elif phase == "RETRACT_L":
            if path_idx < len(path_retract_L):
                q_L = path_retract_L[path_idx]; path_idx += 1
            else:
                q_L = q_zero.copy()
                _enter("CARRY_DRYER")

        elif phase == "CARRY_DRYER":
            if path_idx < len(path_handoff_to_dryer):
                q_R = path_handoff_to_dryer[path_idx]; path_idx += 1
            else:
                _enter("HOLD_DRYER")

        elif phase == "HOLD_DRYER":
            if path_idx < DRYER_HOLD_FRAMES:
                path_idx += 1
            else:
                _enter("RELEASE_R")

        elif phase == "RELEASE_R":
            if path_idx < len(grip_open):
                gr_angle_R = float(grip_open[path_idx]); path_idx += 1
            else:
                gr_angle_R     = APPROACH_OPEN_RAD
                joint_R_active = False
                contact_est_R  = False
                q_contact_R    = None
                _remove_prims(stage, JOINT_PATH_R)  # release R FixedJoint → tray free
                print("[TGC] R released → tray free (dynamic)", flush=True)
                _enter("HOME_R")

        elif phase == "HOME_R":
            if path_idx < len(path_dryer_home_R):
                q_R = path_dryer_home_R[path_idx]; path_idx += 1
            else:
                q_R = q_zero.copy()
                _enter("RESET_SCENE")

        elif phase == "RESET_SCENE":
            if path_idx == 0:
                # Clean up joints; tray is already dynamic (released by R arm) — let it fall
                joint_L_active = False
                joint_R_active = False
                _remove_prims(stage, JOINT_PATH_L, JOINT_PATH_R)
                print(
                    f"[TGC] Reset: tray free-fall from {np.round(_get_tray_translate(stage), 3)}",
                    flush=True,
                )
                path_idx = 1
            elif path_idx < RESET_FRAMES:
                # Tray falls under physics — just wait
                path_idx += 1
            else:
                # Flash: instant teleport to original pose (kinematic→set→dynamic)
                _set_tray_kinematic(stage, True)
                _set_tray_world_transform(stage, tray_T_ref)
                app.update()
                _set_tray_kinematic(stage, False)
                print("[TGC] Flash: new tray at origin", flush=True)
                _enter("PAUSE")

        elif phase == "PAUSE":
            if path_idx < PAUSE_FRAMES:
                path_idx += 1
            else:
                # Start next cycle — re-read ear positions and re-plan
                cycle          += 1
                q_L             = q_zero.copy()
                q_R             = q_zero.copy()
                gr_angle_L      = HOME_OPEN_RAD
                gr_angle_R      = HOME_OPEN_RAD
                contact_est_L   = False
                contact_est_R   = False
                joint_L_active  = False
                joint_R_active  = False
                q_contact_L     = None
                q_contact_R     = None
                _plan_cache.clear()

                cache.Clear()
                gp_T = get_world_pose(stage, cache, GRASP_PRIM_L)
                gp_R = get_world_pose(stage, cache, GRASP_PRIM_R)
                if gp_T is not None and np.linalg.norm(gp_T[:3, 3]) > 0.05:
                    ear_xyz_L = gp_T[:3, 3].copy()
                    if gp_R is not None and np.linalg.norm(gp_R[:3, 3]) > 0.05:
                        ear_xyz_R = gp_R[:3, 3].copy()
                    contact_y_L   = float(ear_xyz_L[1]) + PAD_FACE_DEPTH_M
                    tray_center_x = (float(ear_xyz_L[0]) + float(ear_xyz_R[0])) / 2.0
                    gz_L          = float(ear_xyz_L[2]) + GRASP_Z_OFFSET
                    pre_xyz_L  = np.array([tray_center_x, ear_xyz_L[1] + PRE_Y_OFFSET,  gz_L])
                    pick_xyz_L = np.array([tray_center_x, ear_xyz_L[1] + PICK_Y_OFFSET, gz_L])
                    q_pre_L = _ik_L("L_pre", pre_xyz_L, orient_weight=ORIENT_WEIGHT_STRONG)
                    path_approach_L = _linear_y_path(
                        _ik_L, q_pre_L, pre_xyz_L, pick_xyz_L, APPROACH_STEPS,
                        orient_weight=ORIENT_WEIGHT_STRONG,
                    )
                    _restart = _cu_batch([
                        {"label": "home→preL_c", "q_start": q_zero, "q_goal": q_pre_L},
                    ])
                    _p = _restart.get("home→preL_c")
                    path_home_to_pre_L = _p if (_p is not None and len(_p) > 0) else np.array([q_pre_L])
                    print(f"[TGC] Cycle {cycle}: ear Y={ear_xyz_L[1]:.4f}  ctrX={tray_center_x:.4f}", flush=True)
                _enter("TO_PRE_L")

        # ── Apply FK both arms ────────────────────────────────────────────────
        _set_arm_q(l_ops, chains, q_L)
        _set_arm_q(r_ops, chains, q_R)
        _set_gripper_xform(l_gr_ops, gr_angle_L)
        _set_gripper_xform(r_gr_ops, gr_angle_R)

        # ── Analytical contact forces ─────────────────────────────────────────
        pad_world_L = _pad_L(q_L)
        pad_world_R = _pad_R(q_R)
        drive_y_L   = float(pad_world_L[1, 3])
        drive_y_R   = float(pad_world_R[1, 3])
        _, press_L, fric_L = _solve_ear_contact(drive_y_L, contact_y_L)
        _, press_R, fric_R = _solve_ear_contact(drive_y_R, contact_y_R)

        # ── Tray tracking: FixedJoint does the work; just measure lift ───────
        lift_m = 0.0
        if joint_L_active and pad_z0 is not None:
            lift_m = max(0.0, float(pad_world_L[2, 3]) - pad_z0)

        # ── Tray XYZ for UI ───────────────────────────────────────────────────
        tray_xyz_cur = _get_tray_translate(stage)

        # ── Update UI ─────────────────────────────────────────────────────────
        if frame % 30 == 0:
            print(
                f"[TGC] fr={frame:5d}  {phase:14s}  cy={cycle}"
                f"  L_pad_y={drive_y_L:.4f}  FL={fric_L:.2f}N"
                f"  R_pad_y={drive_y_R:.4f}  FR={fric_R:.2f}N"
                f"  lift={lift_m*100:.1f}cm",
                flush=True,
            )

        ui_mon.push(
            phase=phase, cycle=cycle,
            l_gr_deg=math.degrees(gr_angle_L),
            r_gr_deg=math.degrees(gr_angle_R),
            l_friction=fric_L if contact_est_L else 0.0,
            r_friction=fric_R if contact_est_R else 0.0,
            l_press_mm=press_L if contact_est_L else 0.0,
            r_press_mm=press_R if contact_est_R else 0.0,
            lift_m=lift_m,
            l_pad_xyz=pad_world_L[:3, 3],
            r_pad_xyz=pad_world_R[:3, 3],
            l_contact=contact_est_L,
            r_contact=contact_est_R,
            tray_xyz=tray_xyz_cur,
            tray_ref=tray_xyz_ref,
        )

        app.update()
        frame += 1

    print("[TGC] Stopped.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    print(f"[TGC] Scene: {DEFAULT_SCENE}", flush=True)
    ctx.open_stage(DEFAULT_SCENE)

    for i in range(200):
        app.update()
        s = ctx.get_stage()
        if s and s.GetPrimAtPath(LEFT_GR + "/left_pad").IsValid():
            print(f"[TGC] Stage ready ({i+1} frames)", flush=True)
            break
    else:
        print("[TGC] ERROR: left_pad not found after 200 frames", flush=True)
        return

    _run(app, ctx.get_stage())


if __name__ == "__main__":
    main()
