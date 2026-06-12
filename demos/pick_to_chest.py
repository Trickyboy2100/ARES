#!/usr/bin/env python3
"""GUI/validation demo: visible left arm picks the tray ear and carries to chest.

This replaces the earlier visible red/blue pad probes with the robot's own
visual arm and gripper. A hidden kinematic carrier is still used after gripper
closure to represent a secured grasp in PhysX.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
import sys
_CORE_DIR = str(Path(__file__).resolve().parents[1] / "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)


import numpy as np

from gui_tray_handoff_demo import (
    GRIPPER_LINK_JOINT_CHAINS,
    GRIPPER_OPEN_ANGLE_RAD,
    gf_matrix_from_column_transform,
    gripper_link_transform,
    smoothstep,
)
from kinematics_probe import ARM_JOINTS, DEFAULT_ARM_URDF, GRIPPER_ROOT_SUFFIX, chain_to_link, fk, load_joints

# Gripper orientation for vertical-jaw side-approach grasp:
#   EG2 Z-axis (finger direction) = world -Y  (arm approaches tray from world +Y)
#   EG2 X-axis (jaw close direction) = world -Z (jaw opens/closes vertically)
#   => EG2 Y-axis = world +X  (= cross(EG2_Z, EG2_X) by right-hand rule)
TARGET_UP_WORLD = np.array([1.0, 0.0, 0.0], dtype=float)      # EG2 Y = world +X  (pick phase)
TARGET_FORWARD_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)  # EG2 Z = world -Y  (pick phase)

# At the chest endpoint the gripper rotates to face world -X while keeping the
# tray horizontal.  EG2 X = world -Z throughout (jaw vertical = tray flat).
CHEST_UP_WORLD      = np.array([0.0, -1.0, 0.0], dtype=float)   # EG2 Y = world -Y at chest
CHEST_FORWARD_WORLD = np.array([-1.0, 0.0, 0.0], dtype=float)   # EG2 Z = world -X at chest

# Orientation weights used during lift + carry phases.  Higher than the default
# 0.07 so the tray plane is kept parallel to the ground throughout the motion.
_ORIENT_WEIGHT = 0.22

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
try:
    import config as _cfg
    DEFAULT_SCENE = _cfg.SCENE_USD
except Exception:
    DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026061100_main.usd"
DEFAULT_TRAJECTORY = PLAYGROUND_ROOT / "runtime/tray_handoff_curobo_trajectory.json"
DEFAULT_REPORT = PLAYGROUND_ROOT / "reports/left_arm_tray_pick_to_chest_validation.json"
DEFAULT_LOG = PLAYGROUND_ROOT / "logs/left_arm_tray_pick_to_chest/motion_log.jsonl"
DEFAULT_CACHE = PLAYGROUND_ROOT / "runtime/left_arm_pick_to_chest_path_cache.npz"

LEFT_ROOT = "/World/robot/jaka_minicobo_left"
TRAY_PATH = "/World/Tray"
EAR_FRAME = "/World/Tray/GraspFrames/YPlusEar"

# Tray prim local frame: rotateZ=90, rotateY=0, rotateX=90
# Maps: local X → world Y, local Y → world Z, local Z → world X
TRAY_INIT_ROTATE_ZYX_DEG = (90.0, 0.0, 90.0)  # (rotZ, rotY, rotX)
# TRAY_INIT_TRANSLATE is read from the stage at runtime (see main).

# Left-arm carry endpoint: world XYZ is read from this prim in the scene.
TERMINAL_FRAME = "/World/leftarmterminal"

PROBE_ROOT = "/World/LeftArmGraspRuntime"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]

# Collision box geometry for left_pad / right_pad, expressed in each pad prim's
# OWN local frame (the frame whose origin is the pad prim origin after FK).
# Pad mesh local Z extent: [-0.009, +0.032] m  (finger direction).
# cz=0.020 centres the box at 20 mm along the pad face; hz=0.012 covers 8–32 mm.
# cx offsets inward toward jaw centre (inner contact face); hx covers jaw thickness.
_PAD_CONTACT = {
    "left_pad":  {"cx": +0.0035, "cy": 0.0, "cz": 0.020,
                  "hx":  0.0025, "hy": 0.009, "hz": 0.012},
    "right_pad": {"cx": -0.0035, "cy": 0.0, "cz": 0.020,
                  "hx":  0.0025, "hy": 0.009, "hz": 0.012},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--lift-m", type=float, default=0.07)
    parser.add_argument("--chest-tray-world-xyz", type=float, nargs=3, default=None)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--use-cache", action="store_true",
                        help="Load cached joint path if available (default: always replan).")
    # legacy alias kept for compatibility; use --use-cache to enable cache
    parser.add_argument("--no-cache", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _cache_key(settled_center, ear_center, chest_target, lift_m):
    vals = np.concatenate([settled_center, ear_center, chest_target, [lift_m]])
    return ",".join(f"{v:.4f}" for v in vals)


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def vec(values):
    return [round(float(v), 6) for v in values]


def vec3_sub(a, b):
    return np.asarray(a, dtype=float) - np.asarray(b, dtype=float)


def vec3_len(a):
    return float(np.linalg.norm(np.asarray(a, dtype=float)))


def bbox_payload(stage, path: str):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    box = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"]
    ).ComputeWorldBound(prim).ComputeAlignedBox()
    return {"min": [float(v) for v in box.GetMin()], "max": [float(v) for v in box.GetMax()]}


def bbox_center(payload):
    return np.array([(payload["min"][i] + payload["max"][i]) * 0.5 for i in range(3)], dtype=float)


def xform_world_xyz(stage, path: str):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation(), dtype=float)


def load_chest_target(args, stage=None):
    if args.chest_tray_world_xyz is not None:
        return np.array(args.chest_tray_world_xyz, dtype=float), "cli"
    if stage is not None:
        p = stage.GetPrimAtPath(TERMINAL_FRAME)
        if p and p.IsValid():
            xyz = xform_world_xyz(stage, TERMINAL_FRAME)
            print(f"[LEFT-ARM-PICK] Chest target from scene prim {TERMINAL_FRAME}: {xyz}", flush=True)
            return xyz, TERMINAL_FRAME
    path = Path(args.trajectory)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        matrix = payload.get("calibration", {}).get("tray_handoff_world_matrix4x4_expected")
        if matrix:
            return np.array([matrix[0][3], matrix[1][3], matrix[2][3]], dtype=float), str(path)
    return np.array([0.07, 0.55, 1.182], dtype=float), "default_fallback"


def joint_limits(chain):
    lower = []
    upper = []
    for joint in chain:
        if joint.name in ARM_JOINTS:
            lower.append(-np.pi if joint.lower is None else joint.lower)
            upper.append(np.pi if joint.upper is None else joint.upper)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


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
    rigid = apply_once(UsdPhysics.RigidBodyAPI, prim)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    apply_once(UsdPhysics.MassAPI, prim).CreateMassAttr().Set(0.05)
    return prim, op


def set_carrier(op, xyz):
    from pxr import Gf

    op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def create_grasp_lock(stage, tray_path: str, carrier_path: str):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    joint_path = f"{PROBE_ROOT}/GraspFixedJoint"
    if stage.GetPrimAtPath(joint_path):
        return joint_path
    tray = stage.GetPrimAtPath(tray_path)
    carrier = stage.GetPrimAtPath(carrier_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    tray_world = cache.GetLocalToWorldTransform(tray)
    carrier_world = cache.GetLocalToWorldTransform(carrier)
    anchor = carrier_world.ExtractTranslation()
    local0 = carrier_world.GetInverse().Transform(anchor)
    local1 = tray_world.GetInverse().Transform(anchor)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(carrier_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tray_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local0[0]), float(local0[1]), float(local0[2])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(local1[0]), float(local1[1]), float(local1[2])))
    tray_rot_inv = tray_world.ExtractRotationQuat().GetInverse().GetNormalized()
    tray_rot_inv_i = tray_rot_inv.GetImaginary()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(
            float(tray_rot_inv.GetReal()),
            float(tray_rot_inv_i[0]),
            float(tray_rot_inv_i[1]),
            float(tray_rot_inv_i[2]),
        )
    )
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint_path


def _setup_pad_kinematic_contacts(stage, gripper_root):
    """Add RigidBodyAPI(kinematic) + thin box collider to left_pad and right_pad.

    The box child is fixed in each pad prim's own local frame, so it automatically
    tracks the pad as the FK animation drives the gripper open/closed.  At q=0
    (fully closed) the inner faces are 23.4 mm apart in the jaw direction and
    press into the 30 mm-tall tray ear, generating real PhysX contact forces.
    """
    from pxr import UsdGeom, UsdPhysics, Gf, UsdShade

    mat_path = f"{gripper_root}/PadContactMaterial"
    if not stage.GetPrimAtPath(mat_path):
        m = UsdShade.Material.Define(stage, mat_path)
        pm = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        pm.CreateStaticFrictionAttr().Set(0.9)
        pm.CreateDynamicFrictionAttr().Set(0.7)
        pm.CreateRestitutionAttr().Set(0.0)
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))

    for pad_name, g in _PAD_CONTACT.items():
        pad_path = f"{gripper_root}/{pad_name}"
        pad_prim = stage.GetPrimAtPath(pad_path)
        if not pad_prim or not pad_prim.IsValid():
            print(f"[PAD-CONTACT] WARNING: {pad_path} not found", flush=True)
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
        print(f"[PAD-CONTACT] Kinematic contact body added: {pad_path}", flush=True)


def set_robot_links(link_ops, chains, q):
    q_map = dict(zip(ARM_JOINTS, np.asarray(q, dtype=float).tolist()))
    for link_name in LINK_NAMES:
        link_ops[link_name].Set(gf_matrix_from_column_transform(fk(chains[link_name], q_map)))


def set_gripper(gripper_ops, closed_fraction):
    joint_angle = GRIPPER_OPEN_ANGLE_RAD * (1.0 - float(closed_fraction))
    touched = 0
    for link_name, op in gripper_ops.items():
        op.Set(gf_matrix_from_column_transform(gripper_link_transform(link_name, joint_angle)))
        touched += 1
    return joint_angle, touched


def _pad_world_xyz(base_world, link6_chain, link6_to_pad, q):
    """FK → pad midpoint world position for joint vector q (radians)."""
    q_map = dict(zip(ARM_JOINTS, np.asarray(q, dtype=float).tolist()))
    return (base_world @ fk(link6_chain, q_map) @ link6_to_pad)[:3, 3]


def build_left_arm_path(stage, cache_usd, settled_center, ear_center, chest_target,
                        lift_m=0.07, cache_path=None, force_recompute=False):
    """Compute (or load) the full arm joint-angle path for pick-to-chest.

    All IK solving happens here, before any simulation step.  The corrected chest
    pad target is derived analytically from *settled_center* (the tray doesn't
    move between settling and the grasp lock), so no second IK call is needed at
    runtime.

    Cache: if *cache_path* is given and a matching .npz exists, the heavy IK
    calls are skipped entirely and the path is loaded in milliseconds.
    """
    from make_tray_handoff_curobo_demo import (
        constrained_pose_path,
        constrained_pose_ramp_path,
        fallback_path,
        selected_pad_midpoint,
        solve_pad_pose_ik,
    )

    arm_joints = load_joints(Path(DEFAULT_ARM_URDF))
    chains = {name: chain_to_link(arm_joints, "Link_0", name) for name in LINK_NAMES}
    link6_chain = chains["Link_6"]
    lower, upper = joint_limits(link6_chain)
    base_world, _pad_mid, link6_to_pad = selected_pad_midpoint(stage, cache_usd, "left")
    seed_bank = [
        np.zeros(6),
        np.array([1.0,  0.5,  0.4, -1.1, -0.2, -0.4], dtype=float),
        np.array([1.4,  0.6,  0.4, -1.1, -0.2, -0.4], dtype=float),
        np.array([0.8,  0.8,  0.2, -1.2,  0.1, -0.2], dtype=float),
        # seeds biased toward vertical-jaw / side-approach configuration
        np.array([0.9,  0.5,  0.8, -0.8,  1.4,  1.5], dtype=float),
        np.array([1.1,  0.4,  0.7, -0.9,  1.5,  1.6], dtype=float),
        np.array([0.7,  0.7,  0.6, -1.0,  1.3,  1.4], dtype=float),
    ]

    pre  = ear_center + np.array([0.0,  0.125, 0.0],    dtype=float)
    pick = ear_center + np.array([0.0,  0.012, 0.0],    dtype=float)
    lift = pick       + np.array([0.0,  0.0,   lift_m], dtype=float)

    # ── Try cache ─────────────────────────────────────────────────────────────
    key = _cache_key(settled_center, ear_center, chest_target, lift_m)
    if not force_recompute and cache_path is not None and Path(cache_path).exists():
        try:
            data = np.load(cache_path, allow_pickle=False)
            cached_key = bytes(data["cache_key_bytes"]).decode()
            if cached_key == key:
                path = data["path"]
                phase_bounds = json.loads(bytes(data["phase_bounds_json"]).decode())
                # Recompute corrected_chest_pad analytically (fast FK, no IK)
                lock_idx = phase_bounds["approach_and_close"] + 8
                pad_at_lock = _pad_world_xyz(base_world, link6_chain, link6_to_pad, path[lock_idx])
                tray_from_pad = settled_center - pad_at_lock
                corrected_chest_pad = chest_target - tray_from_pad
                print("[LEFT-ARM-PICK] Loaded cached arm path — skipping IK.", flush=True)
                return {
                    "chains": chains, "link6_chain": link6_chain,
                    "base_world": base_world, "link6_to_pad": link6_to_pad,
                    "path": path, "phase_bounds": phase_bounds,
                    "targets": {"pre": pre, "pick": pick, "lift": lift,
                                "corrected_chest_pad": corrected_chest_pad},
                    "ik_reports": {}, "lower": lower, "upper": upper,
                    "seed_bank": seed_bank, "tray_from_pad": tray_from_pad,
                }
        except Exception as exc:
            print(f"[LEFT-ARM-PICK] Cache load failed ({exc}); recomputing.", flush=True)

    # ── Full IK compute ────────────────────────────────────────────────────────
    print("[LEFT-ARM-PICK] Computing arm path via IK (vertical-jaw orientation)…", flush=True)
    q0 = np.zeros(6)
    q_pre, *_ = solve_pad_pose_ik(
        link6_chain, lower, upper, base_world, link6_to_pad, pre,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD, [q0] + seed_bank,
    )
    path_to_pre = fallback_path(q0, q_pre, count=91)
    path_to_pick, report_pick = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_to_pre[-1], pre, pick,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [q_pre] + seed_bank, count=81,
    )
    path_lift, report_lift = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_to_pick[-1], pick, lift,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [path_to_pick[-1]] + seed_bank, count=81,
        axis_weight=_ORIENT_WEIGHT, forward_weight=_ORIENT_WEIGHT,
    )

    # Derive the corrected chest pad target analytically from the settle position.
    # At the moment the grasp lock fires (approach_and_close + 8), the tray is
    # still at settled_center because the kinematic arm applies no PhysX force
    # until after the lock is created.
    partial = np.vstack([path_to_pre, path_to_pick[1:], path_lift[1:]])
    approach_end = len(path_to_pre) + len(path_to_pick) - 1
    lock_idx = approach_end + 8
    pad_at_lock = _pad_world_xyz(base_world, link6_chain, link6_to_pad, partial[lock_idx])
    tray_from_pad = settled_center - pad_at_lock
    corrected_chest_pad = chest_target - tray_from_pad

    # Carry: smoothly rotate the gripper from pick orientation (EG2 Z=-Y, EG2 Y=+X)
    # to chest orientation (EG2 Z=-X, EG2 Y=-Y).  Both up and forward axes stay in
    # the world XY plane throughout, so EG2 X = up×forward = world -Z at every
    # waypoint — the tray plane remains horizontal for the entire motion.
    path_chest, report_chest = constrained_pose_ramp_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_lift[-1], lift, corrected_chest_pad,
        TARGET_FORWARD_WORLD, CHEST_FORWARD_WORLD,
        [path_lift[-1]] + seed_bank, count=151,
        start_up_world=TARGET_UP_WORLD, end_up_world=CHEST_UP_WORLD,
        axis_weight=_ORIENT_WEIGHT, forward_weight=_ORIENT_WEIGHT,
    )
    full_path = np.vstack([partial, path_chest[1:]])
    phase_bounds = {
        "move_to_pregrasp":  len(path_to_pre),
        "approach_and_close": approach_end,
        "lift":              len(path_to_pre) + len(path_to_pick) + len(path_lift) - 2,
        "carry_to_chest":    len(full_path),
    }

    # ── Save cache ─────────────────────────────────────────────────────────────
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            path=full_path,
            phase_bounds_json=np.frombuffer(json.dumps(phase_bounds).encode(), dtype=np.uint8),
            cache_key_bytes=np.frombuffer(key.encode(), dtype=np.uint8),
        )
        print(f"[LEFT-ARM-PICK] Path cached → {cache_path}", flush=True)

    return {
        "chains": chains, "link6_chain": link6_chain,
        "base_world": base_world, "link6_to_pad": link6_to_pad,
        "path": full_path, "phase_bounds": phase_bounds,
        "targets": {"pre": pre, "pick": pick, "lift": lift,
                    "corrected_chest_pad": corrected_chest_pad},
        "ik_reports": {"pick": report_pick, "lift": report_lift, "corrected_chest": report_chest},
        "lower": lower, "upper": upper,
        "seed_bank": seed_bank, "tray_from_pad": tray_from_pad,
    }


def pad_world(info, q):
    return info["base_world"] @ fk(info["link6_chain"], dict(zip(ARM_JOINTS, q.tolist()))) @ info["link6_to_pad"]


def main():
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1440,
            "height": 900,
            "renderer": "RaytracedLighting",
        }
    )

    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom

    ctx = omni.usd.get_context()
    print(f"[LEFT-ARM-PICK] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(160):
        app.update()
        time.sleep(0.002)
    stage = ctx.get_stage()

    link_ops = {}
    for link_name in LINK_NAMES:
        prim = stage.GetPrimAtPath(f"{LEFT_ROOT}/{link_name}")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        link_ops[link_name] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "left_arm_pick_fk")

    gripper_ops = {}
    for link_name in GRIPPER_LINK_JOINT_CHAINS:
        prim = stage.GetPrimAtPath(f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}/{link_name}")
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Xformable):
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            gripper_ops[link_name] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "left_arm_pick_gripper")

    gripper_root = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
    _setup_pad_kinematic_contacts(stage, gripper_root)
    # Initialise gripper to OPEN so kinematic pad bodies start at correct world
    # positions before physics begins (avoids spurious collisions during settle).
    set_gripper(gripper_ops, 0.0)

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)

    print("[LEFT-ARM-PICK] Scene ready. Press Play in the GUI to start.", flush=True)
    while not timeline.is_playing():
        app.update()
        time.sleep(0.02)
    print("[LEFT-ARM-PICK] Play detected — waiting for physics to settle...", flush=True)

    for _ in range(int(args.settle_sec * args.fps)):
        app.update()

    settled_bbox = bbox_payload(stage, TRAY_PATH)
    settled_center = bbox_center(settled_bbox)
    ear_center = xform_world_xyz(stage, EAR_FRAME)
    if abs(float(ear_center[2] - settled_center[2])) > 0.20:
        ear_center[2] = settled_center[2]
    tray_init_translate = xform_world_xyz(stage, TRAY_PATH)
    print(f"[LEFT-ARM-PICK] Tray start position from scene: {np.round(tray_init_translate, 4)}", flush=True)
    chest_target, chest_source = load_chest_target(args, stage=stage)
    # Cache is disabled by default; pass --use-cache to skip IK replanning.
    use_cache = args.use_cache and not args.no_cache
    cache_path = str(DEFAULT_CACHE) if use_cache else None
    print("[LEFT-ARM-PICK] Solving arm path (IK)…", flush=True)
    info = build_left_arm_path(
        stage, UsdGeom.XformCache(Usd.TimeCode.Default()),
        settled_center, ear_center, chest_target,
        lift_m=args.lift_m,
        cache_path=cache_path,
        force_recompute=not use_cache,
    )

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    samples = []
    gripper_closed_logged = False
    path = info["path"]
    dt = 1.0 / args.fps
    i = 0
    while i < len(path):
        q = path[i]
        set_robot_links(link_ops, info["chains"], q)
        phase = "move_to_pregrasp"
        closed = 0.0
        if i >= info["phase_bounds"]["move_to_pregrasp"]:
            phase = "approach_and_close"
            span = max(1, info["phase_bounds"]["approach_and_close"] - info["phase_bounds"]["move_to_pregrasp"])
            closed = smoothstep((i - info["phase_bounds"]["move_to_pregrasp"]) / span)
        if i >= info["phase_bounds"]["approach_and_close"]:
            phase = "lift"
            closed = 1.0
        if i >= info["phase_bounds"]["lift"]:
            phase = "carry_to_chest"
            closed = 1.0
        joint_angle, touched = set_gripper(gripper_ops, closed)
        pad_T = pad_world(info, q)
        if not gripper_closed_logged and i >= info["phase_bounds"]["approach_and_close"] + 8:
            gripper_closed_logged = True
            print(
                f"[LEFT-ARM-PICK] Gripper closed — kinematic pad contact active (no FixedJoint); "
                f"chest target={vec(info['targets']['corrected_chest_pad'])}",
                flush=True,
            )
        app.update()

        if i % 5 == 0 or i == len(path) - 1:
            tray_bbox = bbox_payload(stage, TRAY_PATH)
            row = {
                "time_sec": round(i * dt, 4),
                "sample_index": int(i),
                "phase": phase,
                "joint_position_rad": np.round(q, 6).tolist(),
                "joint_position_deg": np.round(np.degrees(q), 3).tolist(),
                "pad_midpoint_world_xyz": vec(pad_T[:3, 3]),
                "gripper_closed_fraction": round(float(closed), 6),
                "gripper_joint_angle_deg": round(float(np.degrees(joint_angle)), 6),
                "gripper_visual_links_touched": int(touched),
                "tray_bbox": tray_bbox,
                "tray_bbox_center_xyz": vec(bbox_center(tray_bbox)),
                "chest_tray_target_world_xyz": vec(chest_target),
                "grasp_method": "kinematic_pad_contact",
            }
            samples.append(row)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        i += 1

    final_bbox = bbox_payload(stage, TRAY_PATH)
    final_center = bbox_center(final_bbox)
    lift_min = final_bbox["min"][2] - settled_bbox["min"][2]
    chest_error = vec3_len(final_center - chest_target)
    report = {
        "schema": "left_arm_tray_pick_to_chest_demo/v1",
        "scene": args.scene,
        "method": {
            "visible_motion": "left arm Link_1..Link_6 FK plus real gripper linkage open/close",
            "physics_grasp": "Path A: left_pad/right_pad kinematic rigid bodies with box colliders"
                             " physically contact tray ear — no FixedJoint lock",
            "jaw_orientation": "vertical (EG2 X ≈ world -Z); approach from world +Y",
            "cache": "enabled" if use_cache else "disabled (replanned each run)",
        },
        "tray_path": TRAY_PATH,
        "ear_frame": EAR_FRAME,
        "settled_before_grasp": {
            "bbox": settled_bbox,
            "bbox_center_xyz": vec(settled_center),
            "ear_center_world_xyz": vec(ear_center),
        },
        "targets": {
            "pregrasp_pad_world_xyz": vec(info["targets"]["pre"]),
            "pick_pad_world_xyz": vec(info["targets"]["pick"]),
            "lift_pad_world_xyz": vec(info["targets"]["lift"]),
            "chest_tray_world_xyz": vec(chest_target),
            "chest_target_source": chest_source,
            "corrected_chest_pad_world_xyz": None
            if "corrected_chest_pad" not in info["targets"]
            else vec(info["targets"]["corrected_chest_pad"]),
        },
        "final": {
            "bbox": final_bbox,
            "bbox_center_xyz": vec(final_center),
            "lift_by_bbox_min_m": float(lift_min),
            "chest_center_error_m": float(chest_error),
            "last_pad_midpoint_world_xyz": samples[-1]["pad_midpoint_world_xyz"] if samples else None,
        },
        "thresholds": {"min_lift_by_bbox_min_m": 0.02, "max_chest_center_error_m": 0.03},
        "status": "pass" if lift_min > 0.02 and chest_error < 0.03 else "fail",
        "samples": samples,
        "log_path": str(log_path),
    }
    atomic_write_json(Path(args.report), report)
    print(json.dumps(report["final"], ensure_ascii=False, indent=2), flush=True)
    print(f"[LEFT-ARM-PICK] status={report['status']} report={args.report}", flush=True)

    if args.hold_open:
        while app.is_running():
            app.update()
            time.sleep(0.01)
    else:
        timeline.stop()
        app.close()


if __name__ == "__main__":
    main()
