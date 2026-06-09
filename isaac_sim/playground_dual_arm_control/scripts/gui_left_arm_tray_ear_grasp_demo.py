#!/usr/bin/env python3
"""Left robot arm IK-driven tray ear grasp demo.

Replaces the floating kinematic pad proxies from gui_left_tray_ear_physics_grasp_demo.py
with actual left arm motion. A full IK path is pre-solved before the PhysX
simulation begins. The arm is then FK-played frame by frame while kinematic pad
proxies (positioned from the FK result) provide the physical contact.

Gripper orientation for the ear grasp
--------------------------------------
The EG2-4C2 jaw axis is the pad-frame local X direction. In the normal tray
handoff the jaw opens in world +X (left/right). For the ear clamp the jaw must
open in world +Z (above/below the thin ear). This is achieved by solving IK with:

    target_up   (pad local Y) = [-1, 0, 0]  (world -X)
    target_fwd  (pad local Z) = [ 0,-1, 0]  (world -Y, approach axis)

    → jaw = target_up × target_fwd = [-1,0,0]×[0,-1,0] = [0,0,1]  ✓ (world +Z)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_TRAJECTORY = PLAYGROUND_ROOT / "runtime/tray_handoff_curobo_trajectory.json"
DEFAULT_REPORT = PLAYGROUND_ROOT / "reports/left_arm_ear_grasp_validation.json"
DEFAULT_LOG = PLAYGROUND_ROOT / "logs/left_arm_ear_grasp/motion_log.jsonl"

TRAY_PATH = "/World/Tray"
EAR_FRAME_PATHS = {
    "YPlusEar": "/World/Tray/GraspFrames/YPlusEar",
    "YMinusEar": "/World/Tray/GraspFrames/YMinusEar",
}
ARM_ROOT = "/World/robot/jaka_minicobo_left"
PROBE_ROOT = "/World/LeftGripperPhysicsProbe"
GRIPPER_SUFFIX = "Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]
ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

GRIPPER_LINK_JOINT_CHAINS = {
    "left_outer_link": [((-0.04, -0.009, 0.079), (0.0, -1.0, 0.0))],
    "left_inner_link": [((-0.03, -0.009, 0.081), (0.0, -1.0, 0.0))],
    "right_inner_link": [((0.03, -0.009, 0.081), (0.0, 1.0, 0.0))],
    "right_outer_link": [((0.04, -0.009, 0.079), (0.0, 1.0, 0.0))],
    "left_pad": [
        ((-0.04, -0.009, 0.079), (0.0, -1.0, 0.0)),
        ((0.0223, 0.003, 0.035591), (0.0, 1.0, 0.0)),
    ],
    "right_pad": [
        ((0.04, -0.009, 0.079), (0.0, 1.0, 0.0)),
        ((-0.0223, 0.003, 0.035591), (0.0, -1.0, 0.0)),
    ],
}
GRIPPER_OPEN_ANGLE_RAD = 0.6240523114116221

# Ear-grasp IK orientation targets.
# Jaw axis (local X = local Y × local Z) resolves to world +Z.
EAR_GRASP_PAD_UP_WORLD = np.array([-1.0, 0.0, 0.0])
EAR_GRASP_PAD_FORWARD_WORLD = np.array([0.0, -1.0, 0.0])

# Seeds that span the wrist-rotated configuration space for the ear-grasp pose.
# joint_6 ≈ ±π/2 covers the 90° wrist rotation needed to point the jaw in +Z.
_IK_SEED_BANK = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.3, -0.3, 0.0, 0.8, 1.57]),
    np.array([0.2, 0.5, -0.5, 0.0, 0.8, 1.57]),
    np.array([0.0, 0.3, -0.3, 0.0, 0.8, -1.57]),
    np.array([0.1, 0.4, -0.6, -0.1, 1.0, 1.57]),
]


# ── Parse args ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Left arm IK ear grasp demo")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--tray-path", default=TRAY_PATH)
    parser.add_argument("--ear", choices=sorted(EAR_FRAME_PATHS), default="YPlusEar")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--lift-m", type=float, default=0.07)
    parser.add_argument("--carry-to-chest", action="store_true")
    parser.add_argument(
        "--no-grasp-lock",
        action="store_true",
        help="Disable the fixed grasp joint during carry. For friction-only experiments.",
    )
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--chest-tray-world-xyz", type=float, nargs=3, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    return parser.parse_args()


# ── Math utilities ────────────────────────────────────────────────────────────

def ease(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return [float(ai) * (1.0 - t) + float(bi) * t for ai, bi in zip(a, b)]


def vec3_add(a, b):
    return [float(a[i]) + float(b[i]) for i in range(3)]


def vec3_sub(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def vec3_len(a):
    return math.sqrt(sum(float(v) ** 2 for v in a))


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ── USD / Gf helpers ──────────────────────────────────────────────────────────

def gf_matrix_from_column_transform(T):
    from pxr import Gf
    return Gf.Matrix4d(*sum(np.asarray(T, dtype=float).T.tolist(), []))


def gf_matrix_from_scale_center(center, size):
    """Axis-aligned scaled cube matrix for kinematic pad proxies."""
    from pxr import Gf
    sx, sy, sz = size
    x, y, z = center
    return Gf.Matrix4d(
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, sz, 0.0,
        x, y, z, 1.0,
    )


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


def xform_world_xyz(stage, path: str):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim: {path}")
    mat = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    return [float(v) for v in mat.ExtractTranslation()]


def bbox_payload(stage, path: str):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim: {path}")
    bbox = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        .ComputeWorldBound(prim)
        .ComputeAlignedBox()
    )
    return {"min": [float(v) for v in bbox.GetMin()], "max": [float(v) for v in bbox.GetMax()]}


def bbox_center(b):
    return [(b["min"][i] + b["max"][i]) * 0.5 for i in range(3)]


# ── Phase trajectory (identical to proxy demo) ───────────────────────────────

def phase_pose(t: float, ear_center, lift_m: float, chest_pad_base=None):
    pad_size = [0.12, 0.028, 0.012]
    open_gap = 0.092
    closed_gap = 0.039
    pre_y = 0.125
    contact_y = 0.012

    if t < 0.8:
        phase = "pregrasp_open"
        a = 0.0
    elif t < 2.1:
        phase = "approach_minus_world_y"
        a = ease((t - 0.8) / 1.3)
    elif t < 3.0:
        phase = "close_top_bottom_on_ear"
        a = 1.0
    elif t < 4.4:
        phase = "lift_with_closed_pads"
        a = 1.0
    elif chest_pad_base is not None and t < 7.4:
        phase = "carry_to_chest_handoff"
        a = 1.0
    elif chest_pad_base is not None:
        phase = "hold_at_chest_handoff"
        a = 1.0
    elif t < 5.4:
        phase = "hold_lifted"
        a = 1.0
    else:
        phase = "settled_hold"
        a = 1.0

    y_offset = pre_y * (1.0 - a) + contact_y * a

    if phase == "close_top_bottom_on_ear":
        gap = open_gap * (1.0 - ease((t - 2.1) / 0.9)) + closed_gap * ease((t - 2.1) / 0.9)
        lift = 0.0
    elif phase in {"lift_with_closed_pads", "hold_lifted", "settled_hold"}:
        gap = closed_gap
        lift = lift_m * ease((min(t, 4.4) - 3.0) / 1.4)
    elif phase == "carry_to_chest_handoff":
        gap = closed_gap
        lift = lift_m
    else:
        gap = open_gap
        lift = 0.0

    base = [ear_center[0], ear_center[1] + y_offset, ear_center[2] + lift]
    if phase == "carry_to_chest_handoff":
        base = lerp(base, chest_pad_base, ease((t - 4.4) / 3.0))
    elif phase == "hold_at_chest_handoff":
        base = list(chest_pad_base)

    upper = [base[0], base[1], base[2] + gap * 0.5]
    lower = [base[0], base[1], base[2] - gap * 0.5]
    return phase, upper, lower, pad_size, gap, y_offset, lift


# ── Gripper animation ─────────────────────────────────────────────────────────

def gripper_closed_fraction(phase: str, t: float) -> float:
    if phase == "close_top_bottom_on_ear":
        return ease((t - 2.1) / 0.9)
    if phase in {"lift_with_closed_pads", "carry_to_chest_handoff",
                 "hold_at_chest_handoff", "hold_lifted", "settled_hold"}:
        return 1.0
    return 0.0


def gripper_joint_angle(closed_fraction: float) -> float:
    return GRIPPER_OPEN_ANGLE_RAD * (1.0 - float(closed_fraction))


def _translate_matrix(xyz):
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _axis_angle_matrix(axis, angle_rad):
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


def gripper_link_local_transform(link_name: str, joint_angle_rad: float) -> np.ndarray:
    T = np.eye(4)
    for xyz, axis in GRIPPER_LINK_JOINT_CHAINS[link_name]:
        T = T @ _translate_matrix(xyz) @ _axis_angle_matrix(axis, joint_angle_rad)
    return T


# ── Kinematic pad proxies ─────────────────────────────────────────────────────

def _make_display_material(stage, path: str, color):
    from pxr import Sdf, UsdShade
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color[:3])
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _make_pad_prim(stage, name: str, color):
    from pxr import UsdGeom, UsdPhysics
    path = f"{PROBE_ROOT}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    prim = cube.GetPrim()
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([color[:3]])
    cube.CreateDisplayOpacityAttr([color[3]])
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "arm_ear_grasp")
    apply_once(UsdPhysics.CollisionAPI, prim).CreateCollisionEnabledAttr().Set(True)
    rb = apply_once(UsdPhysics.RigidBodyAPI, prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    rb.CreateKinematicEnabledAttr().Set(True)
    apply_once(UsdPhysics.MassAPI, prim).CreateMassAttr().Set(0.08)
    return prim, op


def _make_carrier_prim(stage):
    from pxr import UsdGeom, UsdPhysics
    path = f"{PROBE_ROOT}/Carrier"
    cube = UsdGeom.Cube.Define(stage, path)
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


def ensure_probe(stage):
    from pxr import UsdGeom, UsdShade
    if stage.GetPrimAtPath(PROBE_ROOT):
        stage.RemovePrim(PROBE_ROOT)
    UsdGeom.Xform.Define(stage, PROBE_ROOT)
    upper_mat = _make_display_material(stage, f"{PROBE_ROOT}/UpperPadMat", (0.95, 0.18, 0.12, 0.82))
    lower_mat = _make_display_material(stage, f"{PROBE_ROOT}/LowerPadMat", (0.10, 0.45, 1.00, 0.82))
    upper_prim, upper_op = _make_pad_prim(stage, "UpperPad", (0.95, 0.18, 0.12, 0.82))
    lower_prim, lower_op = _make_pad_prim(stage, "LowerPad", (0.10, 0.45, 1.00, 0.82))
    carrier_prim, carrier_op = _make_carrier_prim(stage)
    UsdShade.MaterialBindingAPI.Apply(upper_prim).Bind(upper_mat)
    UsdShade.MaterialBindingAPI.Apply(lower_prim).Bind(lower_mat)
    return {
        "upper": {"prim": upper_prim, "op": upper_op},
        "lower": {"prim": lower_prim, "op": lower_op},
        "carrier": {"prim": carrier_prim, "op": carrier_op},
    }


def set_pad_pose(pads, upper_center, lower_center, pad_size):
    from pxr import Gf
    pads["upper"]["op"].Set(gf_matrix_from_scale_center(upper_center, pad_size))
    pads["lower"]["op"].Set(gf_matrix_from_scale_center(lower_center, pad_size))
    carrier = [(float(upper_center[i]) + float(lower_center[i])) * 0.5 for i in range(3)]
    pads["carrier"]["op"].Set(Gf.Vec3d(*carrier))
    return carrier


def create_grasp_lock(stage, tray_path: str, carrier_path: str) -> str:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    joint_path = f"{PROBE_ROOT}/GraspFixedJoint"
    if stage.GetPrimAtPath(joint_path):
        return joint_path
    tray = stage.GetPrimAtPath(tray_path)
    carrier = stage.GetPrimAtPath(carrier_path)
    if not tray or not carrier:
        raise RuntimeError("Cannot create grasp lock: missing tray or carrier")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    tray_world = cache.GetLocalToWorldTransform(tray)
    carrier_world = cache.GetLocalToWorldTransform(carrier)
    anchor = carrier_world.ExtractTranslation()
    local0 = carrier_world.GetInverse().Transform(anchor)
    local1 = tray_world.GetInverse().Transform(anchor)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(carrier_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tray_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local0]))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local1]))
    tray_rot_inv = tray_world.ExtractRotationQuat().GetInverse().GetNormalized()
    tri = tray_rot_inv.GetImaginary()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(float(tray_rot_inv.GetReal()), float(tri[0]), float(tri[1]), float(tri[2]))
    )
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint_path


# ── IK pre-solve ──────────────────────────────────────────────────────────────

def presolve_arm_path(
    chain, lower, upper, base_world, link6_to_pad,
    ear_center, lift_m, chest_pad_base, dt, total_t,
    seed_bank, start_t=0.0, start_q=None,
):
    """Pre-solve IK for every simulation timestep along the phase_pose trajectory.

    Returns an (N, 6) array of joint angles.
    """
    from make_tray_handoff_curobo_demo import solve_pad_pose_ik

    steps = max(1, round((total_t - start_t) / dt))
    path = np.zeros((steps, 6), dtype=float)
    prev_q = np.array(start_q if start_q is not None else seed_bank[0], dtype=float)

    for i in range(steps):
        t = start_t + i * dt
        _, upper_pos, lower_pos, _, gap, _, _ = phase_pose(t, ear_center, lift_m, chest_pad_base)
        pad_mid = np.array([(upper_pos[k] + lower_pos[k]) * 0.5 for k in range(3)])

        # First step: try all seeds to escape bad local minima from the initial pose.
        # Subsequent steps: warm-start from previous solution for continuity.
        seeds = ([prev_q] + seed_bank) if i == 0 else ([prev_q] + seed_bank[:1])
        q, pos_err, up_err, fwd_err, ok, msg = solve_pad_pose_ik(
            chain, lower, upper, base_world, link6_to_pad,
            pad_mid,
            EAR_GRASP_PAD_UP_WORLD,
            EAR_GRASP_PAD_FORWARD_WORLD,
            seeds,
            reference_q=prev_q,
            continuity_weight=0.015 if i > 0 else 0.0,
        )
        path[i] = q
        prev_q = q
        if i % 60 == 0 or i == steps - 1:
            print(
                f"[IK] t={t:.2f}s  pos={pos_err*1000:.1f}mm  "
                f"up={up_err:.1f}°  fwd={fwd_err:.1f}°  ok={ok}",
                flush=True,
            )

    return path


def _sample_path(path: np.ndarray, idx: int) -> np.ndarray:
    return path[min(idx, len(path) - 1)].copy()


# ── Arm FK ops setup ──────────────────────────────────────────────────────────

def setup_link_ops(stage):
    from pxr import UsdGeom
    ops = {}
    for link_name in LINK_NAMES:
        path = f"{ARM_ROOT}/{link_name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            raise RuntimeError(f"Missing xformable arm link: {path}")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        ops[link_name] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "arm_ear_grasp_fk")
    return ops


def setup_gripper_ops(stage):
    from pxr import UsdGeom
    ops = {}
    for link_name in GRIPPER_LINK_JOINT_CHAINS:
        path = f"{ARM_ROOT}/{GRIPPER_SUFFIX}/{link_name}"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            ops[link_name] = xf.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble, "arm_ear_grasp_gripper_fk"
            )
    return ops


def apply_arm_fk(link_ops, chains, q_map):
    from kinematics_probe import fk
    for link_name in LINK_NAMES:
        link_ops[link_name].Set(gf_matrix_from_column_transform(fk(chains[link_name], q_map)))


def apply_gripper_fk(gripper_ops, joint_angle_rad: float):
    for link_name, op in gripper_ops.items():
        op.Set(gf_matrix_from_column_transform(
            gripper_link_local_transform(link_name, joint_angle_rad)
        ))


# ── Chest target ──────────────────────────────────────────────────────────────

def load_chest_target(args):
    if args.chest_tray_world_xyz is not None:
        return [float(v) for v in args.chest_tray_world_xyz], "cli"
    traj = Path(args.trajectory)
    if traj.exists():
        payload = json.loads(traj.read_text(encoding="utf-8"))
        matrix = payload.get("calibration", {}).get("tray_handoff_world_matrix4x4_expected")
        if matrix:
            return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])], str(traj)
    return [0.135, 0.598, 1.032], "default_fallback"


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo(args, app):
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    from kinematics_probe import (
        chain_to_link,
        get_world_pose,
        load_joints,
        relative_pose,
    )
    from ik_sanity import joint_limits

    ctx = omni.usd.get_context()
    print(f"[ARM-GRASP] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(120):
        app.update()
        time.sleep(0.002)

    stage = ctx.get_stage()
    print("[ARM-GRASP] Stage opened.", flush=True)

    tray = stage.GetPrimAtPath(args.tray_path)
    if not tray:
        raise RuntimeError(f"Tray not found: {args.tray_path}")
    if not tray.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Tray is not a rigid body: {args.tray_path}")
    ear_frame_path = EAR_FRAME_PATHS[args.ear]
    if not stage.GetPrimAtPath(ear_frame_path):
        raise RuntimeError(f"Missing grasp frame: {ear_frame_path}")

    # Load URDF kinematics
    arm_urdf_path = Path(
        "/home/andyee/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo.urdf"
    )
    arm_joints_urdf = load_joints(arm_urdf_path)
    chains = {name: chain_to_link(arm_joints_urdf, "Link_0", name) for name in LINK_NAMES}
    arm_chain = chains["Link_6"]
    lower, upper = joint_limits(arm_chain)

    # Calibrate from the current scene pose
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_world = get_world_pose(stage, cache, f"{ARM_ROOT}/Link_0")
    link6_world = get_world_pose(stage, cache, f"{ARM_ROOT}/Link_6")
    left_pad_world = get_world_pose(stage, cache, f"{ARM_ROOT}/{GRIPPER_SUFFIX}/left_pad")
    right_pad_world = get_world_pose(stage, cache, f"{ARM_ROOT}/{GRIPPER_SUFFIX}/right_pad")
    if any(v is None for v in (base_world, link6_world, left_pad_world, right_pad_world)):
        raise RuntimeError("Could not read arm/gripper calibration poses from scene")

    pad_mid_world = np.array(left_pad_world, copy=True)
    pad_mid_world[:3, 3] = (left_pad_world[:3, 3] + right_pad_world[:3, 3]) * 0.5
    link6_to_pad = relative_pose(link6_world, pad_mid_world)

    # Scene fixtures
    pads = ensure_probe(stage)
    link_ops = setup_link_ops(stage)
    gripper_ops = setup_gripper_ops(stage)
    print("[ARM-GRASP] Probe + FK ops ready.", flush=True)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    # Park pads out of the way and put arm at rest during tray settling
    set_pad_pose(pads, [0.0, 2.0, 2.0], [0.0, 2.0, 1.9], [0.02, 0.02, 0.02])
    q_zero_map = dict(zip(ARM_JOINTS, [0.0] * 6))
    apply_arm_fk(link_ops, chains, q_zero_map)
    apply_gripper_fk(gripper_ops, GRIPPER_OPEN_ANGLE_RAD)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.play()
    print("[ARM-GRASP] Settling tray.", flush=True)
    for _ in range(int(args.settle_sec / args.dt)):
        app.update()

    settled_bbox = bbox_payload(stage, args.tray_path)
    settled_center = bbox_center(settled_bbox)
    ear_center = xform_world_xyz(stage, ear_frame_path)
    if abs(ear_center[2] - settled_center[2]) > 0.20:
        ear_center[2] = settled_center[2]

    chest_target, chest_target_source = load_chest_target(args)
    ear_to_tray_offset = vec3_sub(ear_center, settled_center)
    chest_ear_center = vec3_add(chest_target, ear_to_tray_offset)

    # Nominal chest pad base for pre-solve (carries to chest ear center)
    nominal_chest_pad_base = chest_ear_center if args.carry_to_chest else None
    total_t = 9.2 if args.carry_to_chest else 5.8

    print(
        f"[ARM-GRASP] ear_center={[round(v,4) for v in ear_center]}  "
        f"total_t={total_t:.1f}s. Pre-solving IK...",
        flush=True,
    )
    arm_path = presolve_arm_path(
        arm_chain, lower, upper, base_world, link6_to_pad,
        ear_center, args.lift_m, nominal_chest_pad_base, args.dt, total_t,
        _IK_SEED_BANK,
    )
    print(f"[ARM-GRASP] IK pre-solve done: {len(arm_path)} steps.", flush=True)

    # Snap arm to pre-grasp position before loop starts
    q0 = _sample_path(arm_path, 0)
    apply_arm_fk(link_ops, chains, dict(zip(ARM_JOINTS, q0.tolist())))
    apply_gripper_fk(gripper_ops, GRIPPER_OPEN_ANGLE_RAD)
    app.update()

    # Main simulation loop
    steps = round(total_t / args.dt)
    grasp_lock_path = None
    chest_pad_base = None
    samples = []

    for i in range(steps):
        t = i * args.dt

        # At the lift-complete point, refine chest_pad_base from actual tray position
        # and optionally re-solve the carry-phase IK if the tray drifted from nominal.
        if args.carry_to_chest and chest_pad_base is None and t >= 4.4:
            tray_center_now = bbox_center(bbox_payload(stage, args.tray_path))
            _, u_now, l_now, _, _, _, _ = phase_pose(4.4, ear_center, args.lift_m, None)
            carrier_now = [(u_now[k] + l_now[k]) * 0.5 for k in range(3)]
            tray_from_carrier = vec3_sub(tray_center_now, carrier_now)
            chest_pad_base = vec3_sub(chest_target, tray_from_carrier)
            print(
                f"[ARM-GRASP] Runtime chest carrier: {[round(v,4) for v in chest_pad_base]}",
                flush=True,
            )
            # Re-solve carry path if it drifted more than 30mm from the nominal pre-solve
            drift = vec3_len(vec3_sub(chest_pad_base, nominal_chest_pad_base or chest_ear_center))
            if drift > 0.03:
                print(
                    f"[ARM-GRASP] Chest drift {drift*1000:.0f}mm > 30mm; re-solving carry path.",
                    flush=True,
                )
                carry_path = presolve_arm_path(
                    arm_chain, lower, upper, base_world, link6_to_pad,
                    ear_center, args.lift_m, chest_pad_base, args.dt, total_t,
                    _IK_SEED_BANK, start_t=4.4, start_q=_sample_path(arm_path, i),
                )
                split = round(4.4 / args.dt)
                merged = np.concatenate([arm_path[:split], carry_path], axis=0)
                arm_path = merged[:steps] if len(merged) >= steps else np.pad(
                    merged, ((0, steps - len(merged)), (0, 0)), mode="edge"
                )

        phase, upper_pos, lower_pos, pad_size, gap, y_offset, lift = phase_pose(
            t, ear_center, args.lift_m, chest_pad_base
        )

        # Derive pad proxy centers from IK-solved arm FK rather than raw analytical values.
        # T_pad[:3,0] is the jaw direction (world +Z for ear grasp); gap splits along it.
        q = _sample_path(arm_path, i)
        q_map = dict(zip(ARM_JOINTS, q.tolist()))

        from make_tray_handoff_curobo_demo import pad_world_transform
        T_pad = pad_world_transform(arm_chain, base_world, link6_to_pad, q)
        jaw_dir = T_pad[:3, 0]                          # local X → world +Z for ear grasp
        jaw_dir = jaw_dir / max(1e-12, float(np.linalg.norm(jaw_dir)))
        fk_mid = T_pad[:3, 3]
        upper_fk = (fk_mid + jaw_dir * gap * 0.5).tolist()
        lower_fk = (fk_mid - jaw_dir * gap * 0.5).tolist()

        carrier = set_pad_pose(pads, upper_fk, lower_fk, pad_size)

        # FK playback for visual arm
        apply_arm_fk(link_ops, chains, q_map)
        apply_gripper_fk(gripper_ops, gripper_joint_angle(gripper_closed_fraction(phase, t)))

        # Create grasp lock once the close phase has completed
        if (
            args.carry_to_chest
            and not args.no_grasp_lock
            and grasp_lock_path is None
            and t >= 3.05
        ):
            print("[ARM-GRASP] Creating grasp fixed joint.", flush=True)
            grasp_lock_path = create_grasp_lock(
                stage, args.tray_path, str(pads["carrier"]["prim"].GetPath())
            )
            print(f"[ARM-GRASP] Grasp lock: {grasp_lock_path}", flush=True)

        app.update()

        if i % 6 == 0 or i == steps - 1:
            tray_bbox = bbox_payload(stage, args.tray_path)
            tray_xyz = xform_world_xyz(stage, args.tray_path)
            row = {
                "time_sec": round(t, 4),
                "phase": phase,
                "joint_position_rad": np.round(q, 6).tolist(),
                "fk_pad_midpoint_xyz": [round(float(v), 6) for v in fk_mid],
                "fk_jaw_dir_world": [round(float(v), 6) for v in jaw_dir],
                "upper_pad_world_xyz": [round(float(v), 6) for v in upper_fk],
                "lower_pad_world_xyz": [round(float(v), 6) for v in lower_fk],
                "carrier_world_xyz": [round(float(v), 6) for v in carrier],
                "pad_gap_m": round(float(gap), 6),
                "gripper_closed_fraction": round(float(gripper_closed_fraction(phase, t)), 6),
                "chest_tray_target_world_xyz": [round(float(v), 6) for v in chest_target],
                "tray_world_translation_xyz": [round(float(v), 6) for v in tray_xyz],
                "tray_bbox": tray_bbox,
                "tray_bbox_center_xyz": [round(float(v), 6) for v in bbox_center(tray_bbox)],
            }
            samples.append(row)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    timeline.stop()

    final_bbox = bbox_payload(stage, args.tray_path)
    final_center = bbox_center(final_bbox)
    lift_by_bbox_min = final_bbox["min"][2] - settled_bbox["min"][2]
    lift_by_center = final_center[2] - settled_center[2]
    chest_err = vec3_len(vec3_sub(final_center, chest_target)) if args.carry_to_chest else None

    jaw_world = np.cross(EAR_GRASP_PAD_UP_WORLD, EAR_GRASP_PAD_FORWARD_WORLD).tolist()
    report = {
        "schema": "left_arm_ear_grasp_validation/v1",
        "scene": args.scene,
        "tray_path": args.tray_path,
        "ear": args.ear,
        "ear_grasp_frame": ear_frame_path,
        "method": {
            "approach_axis": "world Y: pregrasp at +Y offset, motion toward -Y",
            "clamp_axis": "world Z: gripper jaw aligned with +Z via IK orientation",
            "ik_orientation": {
                "pad_up_world (local Y)": EAR_GRASP_PAD_UP_WORLD.tolist(),
                "pad_fwd_world (local Z)": EAR_GRASP_PAD_FORWARD_WORLD.tolist(),
                "jaw_world (local X = Y x Z)": jaw_world,
            },
            "physics_contact": "kinematic pad proxies positioned from IK-solved FK",
            "visual_arm": "FK playback on Link_1..Link_6 + gripper animation",
        },
        "carry_to_chest": {
            "enabled": bool(args.carry_to_chest),
            "target_source": chest_target_source,
            "desired_tray_center_world_xyz": [round(float(v), 6) for v in chest_target],
            "grasp_lock_enabled": bool(args.carry_to_chest and not args.no_grasp_lock),
            "grasp_lock_path": grasp_lock_path,
        },
        "settled_before_grasp": {
            "bbox": settled_bbox,
            "bbox_center_xyz": [round(float(v), 6) for v in settled_center],
            "ear_center_world_xyz": [round(float(v), 6) for v in ear_center],
        },
        "final": {
            "bbox": final_bbox,
            "bbox_center_xyz": [round(float(v), 6) for v in final_center],
            "lift_by_bbox_min_m": float(lift_by_bbox_min),
            "lift_by_bbox_center_m": float(lift_by_center),
            "chest_center_error_m": float(chest_err) if chest_err is not None else None,
        },
        "thresholds": {
            "min_required_lift_by_bbox_min_m": 0.025,
            "commanded_lift_m": args.lift_m,
            "max_chest_center_error_m": 0.12 if args.carry_to_chest else None,
        },
        "status": (
            "pass"
            if lift_by_bbox_min > 0.025 and (chest_err is None or chest_err < 0.12)
            else "fail"
        ),
        "samples": samples,
        "log_path": str(log_path),
    }
    atomic_write_json(Path(args.report), report)
    print(json.dumps(report["final"], ensure_ascii=False, indent=2), flush=True)
    print(f"[ARM-GRASP] status={report['status']}  report={args.report}", flush=True)

    if args.hold_open:
        timeline.play()
        while app.is_running():
            app.update()
            time.sleep(0.01)
    return report


def main():
    args = parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({
        "headless": args.headless,
        "width": 1440,
        "height": 900,
        "renderer": "RaytracedLighting",
    })
    try:
        run_demo(args, app)
    finally:
        if not args.hold_open:
            app.close()


if __name__ == "__main__":
    main()
