#!/usr/bin/env python3
"""Generate a cuRobo-backed tray handoff demo trajectory."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
try:
    from pxr import Usd, UsdGeom
except ModuleNotFoundError:
    # cuRobo planning workers run outside Isaac Sim's Python environment.
    # Keep pure planning helpers importable there; USD-dependent functions
    # validate pxr availability when called.
    Usd = None
    UsdGeom = None
from scipy.optimize import least_squares

from ik_sanity import joint_limits
from kinematics_probe import (
    ARM_JOINTS,
    CUROBO_JOINTS,
    DEFAULT_ARM_URDF,
    DEFAULT_CUROBO_URDF,
    GRIPPER_ROOT_SUFFIX,
    chain_to_link,
    fk,
    get_world_pose,
    load_joints,
    matrix_to_quat_wxyz,
    pose_dict,
    relative_pose,
)


import os as _os
PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]   # simforge/
_ROBOT_DIR      = PLAYGROUND_ROOT / "robot"

DEFAULT_SCENE = _os.environ.get("SIMFORGE_SCENE") or str(PLAYGROUND_ROOT / "scenes" / "main.usd")
CUROBO_CFG    = _os.environ.get("SIMFORGE_CUROBO_CFG") or str(_ROBOT_DIR / "jaka_minicobo_curobo.yml")

TRAY_START_WORLD = np.array([0.35, 0.35, 1.015], dtype=float)
TRAY_LIFT_WORLD = np.array([0.35, 0.35, 1.08], dtype=float)
TRAY_HANDOFF_WORLD = np.array([0.0, 0.65, 1.032], dtype=float)
TRAY_FINAL_WORLD = np.array([-0.75, 0.58, 0.952], dtype=float)
TRAY_PAD_CLEARANCE_M = 0.168
TRAY_PRE_PICK_CLEARANCE_M = 0.285
PAD_LOCAL_UP_WORLD = np.array([0.0, 0.0, 1.0], dtype=float)
PAD_LOCAL_FORWARD_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)
HANDOFF_GRIPPER_SEPARATION_M = 0.155
HANDOFF_LATERAL_AXIS_WORLD = np.array([1.0, 0.0, 0.0], dtype=float)
RIGHT_FORWARD_EXTENSION_M = 0.10
RIGHT_FORWARD_AXIS_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)
RIGHT_FORWARD_READY_POSITION_TOLERANCE_M = 0.002
RIGHT_FORWARD_READY_FORWARD_TOLERANCE_DEG = 5.0

OBSTACLE_SPECS = [
    ("table", "/World/Table", np.array([0.04, 0.04, 0.03], dtype=float)),
    ("loading_equip", "/World/LoadingEquip", np.array([0.04, 0.04, 0.03], dtype=float)),
    ("dryer", "/World/Dryer", np.array([0.05, 0.05, 0.05], dtype=float)),
]


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def pose_from_xyz(xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


def axis_angle_error_deg(axis: np.ndarray, target: np.ndarray = PAD_LOCAL_UP_WORLD) -> float:
    axis_norm = max(1e-12, float(np.linalg.norm(axis)))
    target_norm = max(1e-12, float(np.linalg.norm(target)))
    cos_angle = float(np.dot(axis, target) / (axis_norm * target_norm))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_angle)))))


def normalized(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=float) / max(1e-12, float(np.linalg.norm(v)))


def interpolate_axis(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    blended = (1.0 - alpha) * normalized(a) + alpha * normalized(b)
    if float(np.linalg.norm(blended)) < 1e-8:
        return normalized(b)
    return normalized(blended)


def transform_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
    rot_delta = a[:3, :3].T @ b[:3, :3]
    cos_angle = (float(np.trace(rot_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_angle)))))


def pad_world_transform(
    arm_chain,
    base_world: np.ndarray,
    link6_to_pad: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    return base_world @ fk(arm_chain, dict(zip(ARM_JOINTS, q.tolist()))) @ link6_to_pad


def interpolate_xyz(a: np.ndarray, b: np.ndarray, count: int) -> List[np.ndarray]:
    rows = []
    for i in range(max(2, count)):
        alpha = smoothstep(i / max(1, count - 1))
        rows.append((1.0 - alpha) * a + alpha * b)
    return rows


def selected_pad_midpoint(stage, cache, side: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = f"/World/robot/jaka_minicobo_{side}"
    base = get_world_pose(stage, cache, f"{root}/Link_0")
    link6 = get_world_pose(stage, cache, f"{root}/Link_6")
    left = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/left_pad")
    right = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/right_pad")
    if base is None or link6 is None or left is None or right is None:
        raise RuntimeError(f"Could not read base/link6/pads for {side}")
    mid = np.array(left, copy=True)
    mid[:3, 3] = (left[:3, 3] + right[:3, 3]) * 0.5
    return base, mid, relative_pose(link6, mid)


def solve_pad_ik(
    chain,
    lower,
    upper,
    base_world: np.ndarray,
    link6_to_pad: np.ndarray,
    target_world_xyz: np.ndarray,
    seeds: List[np.ndarray],
    axis_weight: float = 0.22,
    reference_q: np.ndarray | None = None,
    continuity_weight: float = 0.0,
) -> Tuple[np.ndarray, float, float, bool, str]:

    def residual(q: np.ndarray) -> np.ndarray:
        T_pad = pad_world_transform(chain, base_world, link6_to_pad, q)
        axis = T_pad[:3, 1]
        axis = axis / max(1e-12, float(np.linalg.norm(axis)))
        parts = [
            T_pad[:3, 3] - target_world_xyz,
            axis_weight * (axis - PAD_LOCAL_UP_WORLD),
        ]
        if reference_q is not None and continuity_weight > 0.0:
            parts.append(continuity_weight * (q - reference_q))
        return np.concatenate(parts)

    best = None
    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=1600,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        T_pad = pad_world_transform(chain, base_world, link6_to_pad, result.x)
        pos_err = float(np.linalg.norm(T_pad[:3, 3] - target_world_xyz))
        axis_err = axis_angle_error_deg(T_pad[:3, 1])
        continuity_score = (
            float(np.linalg.norm(result.x - reference_q)) if reference_q is not None else 0.0
        )
        score = (
            pos_err
            + 0.001 * axis_err
            + 0.02 * continuity_score
            + 1e-4 * float(np.linalg.norm(result.x))
        )
        if best is None or score < best[0]:
            best = (score, result.x, pos_err, axis_err, bool(result.success), result.message)
    assert best is not None
    return best[1], best[2], best[3], best[4], best[5]


def solve_pad_pose_ik(
    chain,
    lower,
    upper,
    base_world: np.ndarray,
    link6_to_pad: np.ndarray,
    target_world_xyz: np.ndarray,
    target_up_world: np.ndarray,
    target_forward_world: np.ndarray,
    seeds: List[np.ndarray],
    axis_weight: float = 0.07,
    forward_weight: float = 0.07,
    reference_q: np.ndarray | None = None,
    continuity_weight: float = 0.0,
    target_jaw_world: np.ndarray | None = None,
    jaw_weight: float = 0.0,
) -> Tuple[np.ndarray, float, float, float, bool, str]:
    """IK for pad pose.

    pad frame axes (see gripper.py):
      col 0 (X) = jaw direction  (which way the two fingers open/close)
      col 1 (Y) = lateral axis
      col 2 (Z) = approach / forward direction

    target_jaw_world: if given, constrains pad X-axis (jaw direction).
    jaw_weight:       weight for the jaw constraint (when > 0).
    axis_weight:      weight for pad Y-axis ("up") constraint.
    forward_weight:   weight for pad Z-axis (approach) constraint.
    Setting jaw_weight > 0 with axis_weight=0, forward_weight=0 gives
    vertical-jaw + free-yaw behaviour.
    """
    target_up = np.asarray(target_up_world, dtype=float)
    target_up = target_up / max(1e-12, float(np.linalg.norm(target_up)))
    target_forward = np.asarray(target_forward_world, dtype=float)
    target_forward = target_forward / max(1e-12, float(np.linalg.norm(target_forward)))
    target_jaw = (
        np.asarray(target_jaw_world, dtype=float) / max(1e-12, float(np.linalg.norm(target_jaw_world)))
        if target_jaw_world is not None else None
    )

    def residual(q: np.ndarray) -> np.ndarray:
        T_pad = pad_world_transform(chain, base_world, link6_to_pad, q)
        parts = [T_pad[:3, 3] - target_world_xyz]
        if axis_weight > 0.0:
            up = T_pad[:3, 1] / max(1e-12, float(np.linalg.norm(T_pad[:3, 1])))
            parts.append(axis_weight * (up - target_up))
        if forward_weight > 0.0:
            fwd = T_pad[:3, 2] / max(1e-12, float(np.linalg.norm(T_pad[:3, 2])))
            parts.append(forward_weight * (fwd - target_forward))
        if jaw_weight > 0.0 and target_jaw is not None:
            jaw = T_pad[:3, 0] / max(1e-12, float(np.linalg.norm(T_pad[:3, 0])))
            parts.append(jaw_weight * (jaw - target_jaw))
        if reference_q is not None and continuity_weight > 0.0:
            parts.append(continuity_weight * (q - reference_q))
        return np.concatenate(parts)

    best = None
    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=2200,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        T_pad = pad_world_transform(chain, base_world, link6_to_pad, result.x)
        pos_err = float(np.linalg.norm(T_pad[:3, 3] - target_world_xyz))
        up_err = axis_angle_error_deg(T_pad[:3, 1], target_up)
        forward_err = axis_angle_error_deg(T_pad[:3, 2], target_forward)
        jaw_err = (
            axis_angle_error_deg(T_pad[:3, 0], target_jaw)
            if target_jaw is not None else 0.0
        )
        continuity_score = (
            float(np.linalg.norm(result.x - reference_q)) if reference_q is not None else 0.0
        )
        score = (
            pos_err
            + 0.001 * up_err
            + 0.001 * forward_err
            + 0.001 * jaw_err
            + 0.02 * continuity_score
            + 1e-4 * float(np.linalg.norm(result.x))
        )
        if best is None or score < best[0]:
            best = (
                score,
                result.x,
                pos_err,
                up_err,
                forward_err,
                jaw_err,
                bool(result.success),
                result.message,
            )
    assert best is not None
    # Returns: q, pos_err, up_err, forward_err, jaw_err, ok, msg
    return best[1], best[2], best[3], best[4], best[5], best[6], best[7]


def curobo_tool_pose(curobo_chain, q_arm: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    qmap = dict(zip(CUROBO_JOINTS, q_arm.tolist()))
    qmap["4C2_Joint1"] = 0.0
    T = fk(curobo_chain, qmap)
    return T[:3, 3], matrix_to_quat_wxyz(T[:3, :3])


def make_goal(pos, quat):
    from curobo.types import GoalToolPose

    return GoalToolPose(
        tool_frames=["4C2_Link5"],
        position=torch.tensor(pos.tolist(), device="cuda:0", dtype=torch.float32).reshape(1, 1, 1, 1, 3),
        quaternion=torch.tensor(quat, device="cuda:0", dtype=torch.float32).reshape(1, 1, 1, 1, 4),
    )


def bbox_world(stage, bbox_cache, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = np.array(box.GetMin(), dtype=float)
    mx = np.array(box.GetMax(), dtype=float)
    if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
        return None
    dims = mx - mn
    if float(np.max(dims)) <= 1e-6:
        return None
    return (mn + mx) * 0.5, dims, mn, mx


def build_curobo_obstacles(stage, cache, base_world_by_side):
    from curobo._src.geom.types import Cuboid

    if Usd is None or UsdGeom is None:
        raise RuntimeError("pxr is required to build cuRobo obstacles from a USD stage")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    obstacles = {side: [] for side in base_world_by_side}
    report = {}
    for name, path, inflate in OBSTACLE_SPECS:
        bbox = bbox_world(stage, bbox_cache, path)
        if bbox is None:
            report[name] = {"path": path, "status": "missing_or_empty"}
            continue
        center_world, dims_world, mn, mx = bbox
        report[name] = {
            "path": path,
            "center_world_xyz": np.round(center_world, 9).tolist(),
            "dims_world_xyz": np.round(dims_world, 9).tolist(),
            "bbox_min_world_xyz": np.round(mn, 9).tolist(),
            "bbox_max_world_xyz": np.round(mx, 9).tolist(),
            "inflation_xyz": inflate.tolist(),
        }
        for side, base_world in base_world_by_side.items():
            T_base_obstacle = np.linalg.inv(base_world) @ pose_from_xyz(center_world)
            quat = matrix_to_quat_wxyz(T_base_obstacle[:3, :3])
            dims = np.maximum(dims_world + inflate, 0.01)
            obstacles[side].append(
                Cuboid(
                    name=f"{side}_{name}",
                    dims=np.round(dims, 6).tolist(),
                    pose=np.round(np.r_[T_base_obstacle[:3, 3], quat], 9).tolist(),
                )
            )
    return obstacles, report


def init_curobo_planner(obstacles=None):
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.scene import Scene

    obstacles = obstacles or []
    cfg = MotionPlannerCfg.create(
        robot=CUROBO_CFG,
        scene_model=Scene(cuboid=obstacles),
        collision_cache={"cuboid": max(8, len(obstacles))},
        optimizer_collision_activation_distance=0.02,
    )
    planner = MotionPlanner(cfg)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    return planner


def plan_curobo_segments(curobo_chain, waypoints: Dict[str, np.ndarray], obstacles=None):
    from curobo.types import JointState

    planner = init_curobo_planner(obstacles)
    cur = planner.default_joint_state.clone().unsqueeze(0)
    planned = {}
    statuses = {}
    for name, q_arm in waypoints.items():
        pos, quat = curobo_tool_pose(curobo_chain, q_arm)
        result = planner.plan_pose(make_goal(pos, quat), cur, max_attempts=20)
        if result is None:
            planned[name] = None
            statuses[name] = {"success": False, "reason": "plan_pose returned None"}
            cur_q7 = np.r_[q_arm, 0.0]
            cur = JointState.from_position(
                torch.tensor(cur_q7.reshape(1, -1), device="cuda:0", dtype=torch.float32),
                joint_names=cur.joint_names,
            )
            continue
        plan = result.get_interpolated_plan().position.cpu().numpy()[0, 0, :, :]
        correction_count = max(16, int(np.linalg.norm(plan[-1, :6] - q_arm) * 35))
        correction = fallback_path(plan[-1, :6], q_arm, count=correction_count)
        planned[name] = np.vstack([plan[:, :6], correction[1:]])
        statuses[name] = {
            "success": True,
            "sample_count": int(plan.shape[0]),
            "target_q_error_norm": float(np.linalg.norm(plan[-1, :6] - q_arm)),
            "correction_sample_count": int(correction.shape[0]),
        }
        cur_q7 = np.r_[q_arm, 0.0]
        cur = JointState.from_position(
            torch.tensor(cur_q7.reshape(1, -1), device="cuda:0", dtype=torch.float32),
            joint_names=cur.joint_names,
        )
    return planned, statuses


def fallback_path(q0: np.ndarray, q1: np.ndarray, count: int = 61) -> np.ndarray:
    rows = []
    for i in range(count):
        a = smoothstep(i / max(1, count - 1))
        rows.append((1.0 - a) * q0 + a * q1)
    return np.array(rows)


def constrained_axis_up_path(
    chain,
    lower,
    upper,
    base_world: np.ndarray,
    link6_to_pad: np.ndarray,
    q0: np.ndarray,
    start_world_xyz: np.ndarray,
    end_world_xyz: np.ndarray,
    seed_bank: List[np.ndarray],
    count: int = 61,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    rows = [np.array(q0, dtype=float)]
    report = []
    prev = np.array(q0, dtype=float)
    for idx, xyz in enumerate(interpolate_xyz(start_world_xyz, end_world_xyz, count)[1:], start=1):
        q, pos_err, axis_err, success, message = solve_pad_ik(
            chain,
            lower,
            upper,
            base_world,
            link6_to_pad,
            xyz,
            [prev] + seed_bank,
            reference_q=prev,
            continuity_weight=0.015,
        )
        rows.append(q)
        report.append(
            {
                "sample_index": idx,
                "target_pad_world_xyz": np.round(xyz, 9).tolist(),
                "position_error_m": pos_err,
                "pad_axis_up_error_deg": axis_err,
                "optimizer_success": bool(success),
                "message": message,
            }
        )
        prev = q
    return np.array(rows), report


def constrained_pose_path(
    chain,
    lower,
    upper,
    base_world: np.ndarray,
    link6_to_pad: np.ndarray,
    q0: np.ndarray,
    start_world_xyz: np.ndarray,
    end_world_xyz: np.ndarray,
    target_up_world: np.ndarray,
    target_forward_world: np.ndarray,
    seed_bank: List[np.ndarray],
    count: int = 61,
    axis_weight: float = 0.07,
    forward_weight: float = 0.07,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    rows = [np.array(q0, dtype=float)]
    report = []
    prev = np.array(q0, dtype=float)
    for idx, xyz in enumerate(interpolate_xyz(start_world_xyz, end_world_xyz, count)[1:], start=1):
        q, pos_err, up_err, forward_err, _jaw_err, success, message = solve_pad_pose_ik(
            chain,
            lower,
            upper,
            base_world,
            link6_to_pad,
            xyz,
            target_up_world,
            target_forward_world,
            [prev] + seed_bank,
            reference_q=prev,
            continuity_weight=0.015,
            axis_weight=axis_weight,
            forward_weight=forward_weight,
        )
        rows.append(q)
        report.append(
            {
                "sample_index": idx,
                "target_pad_world_xyz": np.round(xyz, 9).tolist(),
                "position_error_m": pos_err,
                "pad_axis_up_error_deg": up_err,
                "pad_forward_axis_error_deg": forward_err,
                "optimizer_success": bool(success),
                "message": message,
            }
        )
        prev = q
    return np.array(rows), report


def constrained_pose_ramp_path(
    chain,
    lower,
    upper,
    base_world: np.ndarray,
    link6_to_pad: np.ndarray,
    q0: np.ndarray,
    start_world_xyz: np.ndarray,
    end_world_xyz: np.ndarray,
    start_forward_world: np.ndarray,
    end_forward_world: np.ndarray,
    seed_bank: List[np.ndarray],
    count: int = 61,
    start_up_world: np.ndarray | None = None,
    end_up_world: np.ndarray | None = None,
    axis_weight: float = 0.07,
    forward_weight: float = 0.07,
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    _start_up = PAD_LOCAL_UP_WORLD if start_up_world is None else np.asarray(start_up_world, dtype=float)
    _end_up   = PAD_LOCAL_UP_WORLD if end_up_world   is None else np.asarray(end_up_world,   dtype=float)
    rows = [np.array(q0, dtype=float)]
    report = []
    prev = np.array(q0, dtype=float)
    xyz_rows = interpolate_xyz(start_world_xyz, end_world_xyz, count)
    for idx, xyz in enumerate(xyz_rows[1:], start=1):
        alpha = smoothstep(idx / max(1, count - 1))
        target_forward = interpolate_axis(start_forward_world, end_forward_world, alpha)
        target_up      = interpolate_axis(_start_up, _end_up, alpha)
        q, pos_err, up_err, forward_err, _jaw_err, success, message = solve_pad_pose_ik(
            chain,
            lower,
            upper,
            base_world,
            link6_to_pad,
            xyz,
            target_up,
            target_forward,
            [prev] + seed_bank,
            reference_q=prev,
            continuity_weight=0.015,
            axis_weight=axis_weight,
            forward_weight=forward_weight,
        )
        rows.append(q)
        report.append(
            {
                "sample_index": idx,
                "target_pad_world_xyz": np.round(xyz, 9).tolist(),
                "target_pad_forward_axis_world": np.round(target_forward, 9).tolist(),
                "position_error_m": pos_err,
                "pad_axis_up_error_deg": up_err,
                "pad_forward_axis_error_deg": forward_err,
                "optimizer_success": bool(success),
                "message": message,
            }
        )
        prev = q
    return np.array(rows), report


def sample_path(path: np.ndarray, alpha: float) -> np.ndarray:
    if len(path) == 1:
        return path[0].copy()
    u = max(0.0, min(1.0, alpha)) * (len(path) - 1)
    i = int(math.floor(u))
    j = min(len(path) - 1, i + 1)
    a = u - i
    return (1.0 - a) * path[i] + a * path[j]


def build_timeline(paths, fps: float):
    phases = [
        ("start_hold", 1.0, "hold_zero", "hold_zero", "table"),
        ("left_pre_pick", 2.0, "left_pre_pick", "hold_zero", "table"),
        ("left_descend_to_tray", 2.0, "left_pick", "hold_zero", "table"),
        ("left_lift_tray", 2.5, "left_lift", "hold_zero", "left"),
        ("move_to_chest_handoff", 4.0, "left_handoff", "right_handoff", "left"),
        ("handoff_hold", 2.0, "hold_left_handoff", "hold_right_handoff", "both"),
        ("right_takes_tray", 4.0, "left_retract", "right_forward_ready", "right"),
        ("right_face_forward", 1.0, "hold_zero", "hold_right_forward_ready", "right"),
        ("right_forward_extend", 2.0, "hold_zero", "right_forward_extend", "right"),
        ("right_horizontal_hold", 2.0, "hold_zero", "hold_right_forward_extend", "right"),
    ]
    total = sum(p[1] for p in phases)
    samples = []
    t0 = 0.0
    q_zero = np.zeros(6)
    fixed = {
        "hold_zero": q_zero,
        "hold_left_handoff": paths["left_handoff"][-1],
        "hold_right_handoff": paths["right_handoff"][-1],
        "hold_right_final": paths["right_final"][-1],
        "hold_right_forward_ready": paths["right_forward_ready"][-1],
        "hold_right_forward_extend": paths["right_forward_extend"][-1],
    }
    for phase_name, duration, left_key, right_key, tray_attachment in phases:
        n = max(2, int(round(duration * fps)))
        for i in range(n):
            if samples and i == 0:
                continue
            local = i / max(1, n - 1)
            row = {
                "time_sec": round(t0 + local * duration, 6),
                "phase": phase_name,
                "tray_attachment": tray_attachment,
                "arms": {},
            }
            for side, key in [("left", left_key), ("right", right_key)]:
                if key in fixed:
                    q = fixed[key]
                else:
                    q = sample_path(paths[key], local)
                row["arms"][side] = {
                    "joint_names": ARM_JOINTS,
                    "joint_position_rad": np.round(q, 9).tolist(),
                }
            samples.append(row)
        t0 += duration
    return samples, total


def annotate_tray(samples, arm_chain, base_world, link6_to_pad, calibration):
    tray_start_pose = np.array(calibration["tray_start_world_matrix4x4"], dtype=float)
    pad_to_tray = {
        side: np.array(calibration["pad_to_tray_matrix4x4"][side], dtype=float)
        for side in ("left", "right")
    }
    validation = {
        "max_pad_axis_up_error_deg": {"left": 0.0, "right": 0.0},
        "max_tray_level_error_deg": 0.0,
        "max_both_attachment_tray_agreement_m": 0.0,
        "max_both_attachment_tray_agreement_deg": 0.0,
    }
    for row in samples:
        pads = {}
        for side in ("left", "right"):
            q = np.array(row["arms"][side]["joint_position_rad"], dtype=float)
            T_pad = pad_world_transform(arm_chain, base_world[side], link6_to_pad[side], q)
            pads[side] = T_pad
            axis_error = axis_angle_error_deg(T_pad[:3, 1])
            validation["max_pad_axis_up_error_deg"][side] = max(
                validation["max_pad_axis_up_error_deg"][side], axis_error
            )
            row["arms"][side]["pad_midpoint_world_xyz_expected"] = np.round(T_pad[:3, 3], 9).tolist()
            row["arms"][side]["pad_midpoint_world_matrix4x4_expected"] = np.round(T_pad, 9).tolist()
            row["arms"][side]["pad_local_y_axis_world_expected"] = np.round(T_pad[:3, 1], 9).tolist()
            row["arms"][side]["pad_axis_up_error_deg_expected"] = round(axis_error, 9)
        attachment = row["tray_attachment"]
        if attachment == "table":
            tray_T = tray_start_pose
        elif attachment == "left":
            tray_T = pads["left"] @ pad_to_tray["left"]
        elif attachment == "right":
            tray_T = pads["right"] @ pad_to_tray["right"]
        else:
            tray_T = pads["left"] @ pad_to_tray["left"]
            right_tray_T = pads["right"] @ pad_to_tray["right"]
            agreement_m = float(np.linalg.norm(tray_T[:3, 3] - right_tray_T[:3, 3]))
            agreement_deg = transform_delta_deg(tray_T, right_tray_T)
            validation["max_both_attachment_tray_agreement_m"] = max(
                validation["max_both_attachment_tray_agreement_m"], agreement_m
            )
            validation["max_both_attachment_tray_agreement_deg"] = max(
                validation["max_both_attachment_tray_agreement_deg"], agreement_deg
            )
            row["both_attachment_right_tray_candidate"] = pose_dict(right_tray_T)
            row["both_attachment_tray_agreement_m"] = round(agreement_m, 9)
            row["both_attachment_tray_agreement_deg"] = round(agreement_deg, 9)
        level_error = axis_angle_error_deg(tray_T[:3, 2])
        validation["max_tray_level_error_deg"] = max(
            validation["max_tray_level_error_deg"], level_error
        )
        row["tray_world"] = pose_dict(tray_T)
        row["tray_world"]["level_error_deg_expected"] = round(level_error, 9)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--out", default=str(PLAYGROUND_ROOT / "runtime/tray_handoff_curobo_trajectory.json"))
    parser.add_argument("--report", default=str(PLAYGROUND_ROOT / "reports/tray_handoff_curobo_plan.json"))
    parser.add_argument("--position-tolerance", type=float, default=1e-3)
    parser.add_argument("--axis-tolerance-deg", type=float, default=2.0)
    args = parser.parse_args()

    if Usd is None or UsdGeom is None:
        raise RuntimeError("pxr is required to open USD scenes; run this entry point with Isaac Sim Python")

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        raise RuntimeError(f"Could not open scene: {args.scene}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    arm_joints = load_joints(Path(DEFAULT_ARM_URDF))
    arm_chain = chain_to_link(arm_joints, "Link_0", "Link_6")
    lower, upper = joint_limits(arm_chain)
    curobo_joints = load_joints(Path(DEFAULT_CUROBO_URDF))
    curobo_chain = chain_to_link(curobo_joints, "base_link", "4C2_Link5")

    base_world = {}
    link6_to_pad = {}
    for side in ("left", "right"):
        base, _pad, rel = selected_pad_midpoint(stage, cache, side)
        base_world[side] = base
        link6_to_pad[side] = rel

    tray_start_pose = get_world_pose(stage, cache, "/World/DemoTray")
    if tray_start_pose is None:
        tray_start_pose = pose_from_xyz(TRAY_START_WORLD)
    tray_start_world = tray_start_pose[:3, 3]
    tray_lift_world = TRAY_LIFT_WORLD.copy()
    tray_lift_world[:2] = tray_start_world[:2]
    tray_lift_world[2] = max(tray_lift_world[2], tray_start_world[2] + 0.065)
    tray_handoff_world = TRAY_HANDOFF_WORLD.copy()
    tray_final_world = TRAY_FINAL_WORLD.copy()

    lateral_axis = HANDOFF_LATERAL_AXIS_WORLD / max(
        1e-12, float(np.linalg.norm(HANDOFF_LATERAL_AXIS_WORLD))
    )
    forward_axis = RIGHT_FORWARD_AXIS_WORLD / max(
        1e-12, float(np.linalg.norm(RIGHT_FORWARD_AXIS_WORLD))
    )
    tray_forward_extend_world = tray_final_world.copy()
    tray_forward_ready_world = tray_forward_extend_world - forward_axis * RIGHT_FORWARD_EXTENSION_M
    left_tray_contact_offset_world = -lateral_axis * (HANDOFF_GRIPPER_SEPARATION_M * 0.5)
    right_tray_contact_offset_world = -left_tray_contact_offset_world
    left_handoff_contact_offset_world = right_tray_contact_offset_world
    right_handoff_contact_offset_world = left_tray_contact_offset_world
    pad_offset_world = np.array([0.0, 0.0, TRAY_PAD_CLEARANCE_M], dtype=float)
    pad_targets = {
        "left_pre_pick": tray_start_world
        + left_tray_contact_offset_world
        + np.array([0.0, 0.0, TRAY_PRE_PICK_CLEARANCE_M], dtype=float),
        "left_pick": tray_start_world + left_tray_contact_offset_world + pad_offset_world,
        "left_lift": tray_lift_world + left_tray_contact_offset_world + pad_offset_world,
        "left_handoff": tray_handoff_world + left_handoff_contact_offset_world + pad_offset_world,
        "right_handoff": tray_handoff_world + right_handoff_contact_offset_world + pad_offset_world,
    }

    targets = {
        "left_pre_pick": ("left", pad_targets["left_pre_pick"]),
        "left_pick": ("left", pad_targets["left_pick"]),
        "left_lift": ("left", pad_targets["left_lift"]),
        "left_handoff": ("left", pad_targets["left_handoff"]),
        "right_handoff": ("right", pad_targets["right_handoff"]),
    }

    seed_bank = {
        "left": [
            np.zeros(6),
            np.array([-1.03, -1.23, 0.23, 0.08, 0.58, -0.41]),
            np.array([-1.46, -0.85, -0.03, 1.35, -0.63, 0.94]),
            np.array([-1.44, -0.49, -0.69, -0.23, -1.10, -0.12]),
        ],
        "right": [
            np.zeros(6),
            np.array([1.02, -0.74, -0.82, 1.22, -1.23, 0.57]),
            np.array([0.19, 0.66, -0.99, 0.63, -0.19, 0.64]),
            np.array([1.00, 0.46, -0.72, -0.26, -0.61, 0.34]),
            np.array([1.77, 1.38, -0.67, -0.14, -1.49, -1.70]),
            np.array([1.82, 1.21, -0.83, -0.19, -1.16, -1.67]),
        ],
    }
    q_targets = {}
    ik_report = {}
    last_seed_by_side = {}
    for name, (side, xyz) in targets.items():
        seeds = ([last_seed_by_side[side]] if side in last_seed_by_side else []) + seed_bank[side]
        axis_weight = 0.22
        q, err, axis_err, success, message = solve_pad_ik(
            arm_chain, lower, upper, base_world[side], link6_to_pad[side], xyz, seeds, axis_weight=axis_weight
        )
        if err > args.position_tolerance:
            raise RuntimeError(f"{name} IK error {err:.6f} m exceeds tolerance")
        if axis_err > args.axis_tolerance_deg:
            raise RuntimeError(f"{name} pad axis error {axis_err:.3f} deg exceeds tolerance")
        q_targets[name] = q
        last_seed_by_side[side] = q
        ik_report[name] = {
            "side": side,
            "target_pad_world_xyz": np.round(xyz, 9).tolist(),
            "success": bool(success) or err <= args.position_tolerance,
            "optimizer_success": bool(success),
            "message": message,
            "position_error_m": err,
            "pad_axis_up_error_deg": axis_err,
            "axis_weight": axis_weight,
            "joint_position_rad": np.round(q, 9).tolist(),
            "joint_position_deg": np.round(np.degrees(q), 6).tolist(),
        }

    curobo_obstacles, obstacle_report = build_curobo_obstacles(stage, cache, base_world)

    left_waypoints = {
        "left_pre_pick": q_targets["left_pre_pick"],
        "left_pick": q_targets["left_pick"],
        "left_lift": q_targets["left_lift"],
        "left_handoff": q_targets["left_handoff"],
    }
    right_waypoints = {
        "right_handoff": q_targets["right_handoff"],
    }

    left_plan, left_status = plan_curobo_segments(
        curobo_chain, left_waypoints, curobo_obstacles["left"]
    )
    right_plan, right_status = plan_curobo_segments(
        curobo_chain, right_waypoints, curobo_obstacles["right"]
    )

    paths = {}
    path_sources = {}
    constrained_path_report = {}
    phase_count = lambda seconds: max(2, int(round(seconds * args.fps)))

    paths["left_pre_pick"] = (
        left_plan["left_pre_pick"]
        if left_plan["left_pre_pick"] is not None
        else fallback_path(np.zeros(6), q_targets["left_pre_pick"], count=61)
    )
    path_sources["left_pre_pick"] = (
        "curobo_obstacle_plan" if left_plan["left_pre_pick"] is not None else "fallback_joint_smoothstep"
    )
    paths["left_pick"], constrained_path_report["left_pick"] = constrained_axis_up_path(
        arm_chain,
        lower,
        upper,
        base_world["left"],
        link6_to_pad["left"],
        paths["left_pre_pick"][-1],
        pad_targets["left_pre_pick"],
        pad_targets["left_pick"],
        seed_bank["left"],
        count=phase_count(2.0),
    )
    paths["left_lift"], constrained_path_report["left_lift"] = constrained_axis_up_path(
        arm_chain,
        lower,
        upper,
        base_world["left"],
        link6_to_pad["left"],
        paths["left_pick"][-1],
        pad_targets["left_pick"],
        pad_targets["left_lift"],
        seed_bank["left"],
        count=phase_count(2.5),
    )
    paths["left_handoff"], constrained_path_report["left_handoff"] = constrained_axis_up_path(
        arm_chain,
        lower,
        upper,
        base_world["left"],
        link6_to_pad["left"],
        paths["left_lift"][-1],
        pad_targets["left_lift"],
        pad_targets["left_handoff"],
        seed_bank["left"],
        count=phase_count(4.0),
    )
    path_sources.update(
        {
            "left_pick": "axis_up_cartesian_ik_projection",
            "left_lift": "axis_up_cartesian_ik_projection",
            "left_handoff": "axis_up_cartesian_ik_projection",
        }
    )

    paths["right_handoff"] = (
        right_plan["right_handoff"]
        if right_plan["right_handoff"] is not None
        else fallback_path(np.zeros(6), q_targets["right_handoff"], count=81)
    )
    path_sources["right_handoff"] = (
        "curobo_obstacle_plan" if right_plan["right_handoff"] is not None else "fallback_joint_smoothstep"
    )

    T_left_pick_pad = pad_world_transform(
        arm_chain, base_world["left"], link6_to_pad["left"], paths["left_pick"][-1]
    )
    T_left_handoff_pad = pad_world_transform(
        arm_chain, base_world["left"], link6_to_pad["left"], paths["left_handoff"][-1]
    )
    T_left_pad_to_tray = np.linalg.inv(T_left_pick_pad) @ tray_start_pose
    T_left_tray_at_handoff = T_left_handoff_pad @ T_left_pad_to_tray

    T_right_handoff_pad = pad_world_transform(
        arm_chain, base_world["right"], link6_to_pad["right"], paths["right_handoff"][-1]
    )
    T_right_pad_to_tray = np.linalg.inv(T_right_handoff_pad) @ T_left_tray_at_handoff

    pad_forward_x_world = np.cross(PAD_LOCAL_UP_WORLD, PAD_LOCAL_FORWARD_WORLD)
    pad_forward_x_world = normalized(pad_forward_x_world)
    pad_forward_R = np.column_stack(
        [
            pad_forward_x_world,
            normalized(PAD_LOCAL_UP_WORLD),
            normalized(PAD_LOCAL_FORWARD_WORLD),
        ]
    )
    tray_final_pose = np.array(T_left_tray_at_handoff, copy=True)
    tray_final_pose[:3, 3] = tray_forward_ready_world
    tray_final_forward_pose = np.array(tray_final_pose, copy=True)
    tray_final_forward_pose[:3, :3] = pad_forward_R @ T_right_pad_to_tray[:3, :3]
    tray_final_forward_pose[:3, 3] = tray_forward_ready_world
    T_right_forward_ready_pad_goal = tray_final_forward_pose @ np.linalg.inv(T_right_pad_to_tray)
    pad_targets["right_forward_ready"] = T_right_forward_ready_pad_goal[:3, 3]
    q, err, up_err, forward_err, _jaw_err, success, message = solve_pad_pose_ik(
        arm_chain,
        lower,
        upper,
        base_world["right"],
        link6_to_pad["right"],
        pad_targets["right_forward_ready"],
        PAD_LOCAL_UP_WORLD,
        PAD_LOCAL_FORWARD_WORLD,
        [paths["right_handoff"][-1]] + seed_bank["right"],
    )
    ready_forward_tolerance = RIGHT_FORWARD_READY_FORWARD_TOLERANCE_DEG
    ready_position_tolerance = RIGHT_FORWARD_READY_POSITION_TOLERANCE_M
    if err > ready_position_tolerance:
        raise RuntimeError(
            f"right_forward_ready IK error {err:.6f} m exceeds accepted tolerance "
            f"{ready_position_tolerance:.6f}"
        )
    if up_err > args.axis_tolerance_deg:
        raise RuntimeError(f"right_forward_ready pad up-axis error {up_err:.3f} deg exceeds tolerance")
    if forward_err > ready_forward_tolerance:
        raise RuntimeError(
            f"right_forward_ready pad forward-axis error {forward_err:.3f} deg exceeds accepted tolerance "
            f"{ready_forward_tolerance:.3f}"
        )
    q_targets["right_forward_ready"] = q
    q_targets["right_final"] = q.copy()
    ik_report["right_forward_ready"] = {
        "side": "right",
        "target_pad_world_xyz": np.round(pad_targets["right_forward_ready"], 9).tolist(),
        "desired_tray_world_xyz": np.round(tray_forward_ready_world, 9).tolist(),
        "target_pad_forward_axis_world": np.round(PAD_LOCAL_FORWARD_WORLD, 9).tolist(),
        "success": bool(success) or err <= args.position_tolerance,
        "optimizer_success": bool(success),
        "message": message,
        "position_error_m": err,
        "accepted_position_tolerance_m": ready_position_tolerance,
        "pad_axis_up_error_deg": up_err,
        "pad_forward_axis_error_deg": forward_err,
        "accepted_pad_forward_axis_tolerance_deg": ready_forward_tolerance,
        "joint_position_rad": np.round(q, 9).tolist(),
        "joint_position_deg": np.round(np.degrees(q), 6).tolist(),
    }
    pad_targets["right_final"] = pad_targets["right_forward_ready"].copy()
    ik_report["right_final"] = dict(ik_report["right_forward_ready"])
    ik_report["right_final"]["phase_meaning"] = "legacy_alias_of_right_forward_ready"

    tray_forward_extend_pose = np.array(tray_final_forward_pose, copy=True)
    tray_forward_extend_pose[:3, 3] = tray_forward_extend_world
    T_right_forward_extend_pad_goal = tray_forward_extend_pose @ np.linalg.inv(T_right_pad_to_tray)
    pad_targets["right_forward_extend"] = T_right_forward_extend_pad_goal[:3, 3]
    q, err, up_err, forward_err, _jaw_err, success, message = solve_pad_pose_ik(
        arm_chain,
        lower,
        upper,
        base_world["right"],
        link6_to_pad["right"],
        pad_targets["right_forward_extend"],
        PAD_LOCAL_UP_WORLD,
        PAD_LOCAL_FORWARD_WORLD,
        [q_targets["right_forward_ready"]] + seed_bank["right"],
    )
    if err > args.position_tolerance:
        raise RuntimeError(f"right_forward_extend IK error {err:.6f} m exceeds tolerance")
    if up_err > args.axis_tolerance_deg:
        raise RuntimeError(f"right_forward_extend pad up-axis error {up_err:.3f} deg exceeds tolerance")
    if forward_err > args.axis_tolerance_deg:
        raise RuntimeError(f"right_forward_extend pad forward-axis error {forward_err:.3f} deg exceeds tolerance")
    q_targets["right_forward_extend"] = q
    ik_report["right_forward_extend"] = {
        "side": "right",
        "target_pad_world_xyz": np.round(pad_targets["right_forward_extend"], 9).tolist(),
        "desired_tray_world_xyz": np.round(tray_forward_extend_pose[:3, 3], 9).tolist(),
        "target_pad_forward_axis_world": np.round(PAD_LOCAL_FORWARD_WORLD, 9).tolist(),
        "success": bool(success) or err <= args.position_tolerance,
        "optimizer_success": bool(success),
        "message": message,
        "position_error_m": err,
        "pad_axis_up_error_deg": up_err,
        "pad_forward_axis_error_deg": forward_err,
        "joint_position_rad": np.round(q, 9).tolist(),
        "joint_position_deg": np.round(np.degrees(q), 6).tolist(),
    }

    paths["right_forward_ready"], constrained_path_report["right_forward_ready"] = constrained_pose_ramp_path(
        arm_chain,
        lower,
        upper,
        base_world["right"],
        link6_to_pad["right"],
        paths["right_handoff"][-1],
        pad_targets["right_handoff"],
        pad_targets["right_forward_ready"],
        T_right_handoff_pad[:3, 2],
        PAD_LOCAL_FORWARD_WORLD,
        [q_targets["right_forward_ready"]] + seed_bank["right"],
        count=phase_count(4.0),
    )
    paths["right_final"] = paths["right_forward_ready"]
    constrained_path_report["right_final"] = constrained_path_report["right_forward_ready"]
    path_sources["right_forward_ready"] = "forward_ramp_pose_cartesian_ik_projection"
    path_sources["right_final"] = "legacy_alias_of_right_forward_ready"
    paths["right_forward_extend"], constrained_path_report["right_forward_extend"] = constrained_pose_path(
        arm_chain,
        lower,
        upper,
        base_world["right"],
        link6_to_pad["right"],
        paths["right_forward_ready"][-1],
        pad_targets["right_forward_ready"],
        pad_targets["right_forward_extend"],
        PAD_LOCAL_UP_WORLD,
        PAD_LOCAL_FORWARD_WORLD,
        [q_targets["right_forward_extend"], q_targets["right_forward_ready"]] + seed_bank["right"],
        count=phase_count(2.0),
    )
    path_sources["right_forward_extend"] = "forward_pose_cartesian_ik_projection"
    paths["left_retract"] = fallback_path(paths["left_handoff"][-1], np.zeros(6), count=81)
    path_sources["left_retract"] = "fallback_joint_smoothstep"

    calibration = {
        "base_world_matrix4x4": {side: np.round(base_world[side], 9).tolist() for side in ("left", "right")},
        "link6_to_pad_matrix4x4": {
            side: np.round(link6_to_pad[side], 9).tolist() for side in ("left", "right")
        },
        "pad_to_tray_matrix4x4": {
            "left": np.round(T_left_pad_to_tray, 9).tolist(),
            "right": np.round(T_right_pad_to_tray, 9).tolist(),
        },
        "tray_start_world_matrix4x4": np.round(tray_start_pose, 9).tolist(),
        "tray_handoff_world_matrix4x4_expected": np.round(T_left_tray_at_handoff, 9).tolist(),
        "tray_right_pre_extend_world_matrix4x4_desired": np.round(tray_final_pose, 9).tolist(),
        "tray_final_world_matrix4x4_desired": np.round(tray_forward_extend_pose, 9).tolist(),
        "tray_forward_ready_world_matrix4x4_desired": np.round(tray_final_forward_pose, 9).tolist(),
        "tray_forward_extend_world_matrix4x4_desired": np.round(tray_forward_extend_pose, 9).tolist(),
        "handoff_gripper_separation_m": HANDOFF_GRIPPER_SEPARATION_M,
        "left_tray_contact_offset_world_xyz": np.round(left_tray_contact_offset_world, 9).tolist(),
        "right_tray_contact_offset_world_xyz": np.round(right_tray_contact_offset_world, 9).tolist(),
        "left_handoff_contact_offset_world_xyz": np.round(left_handoff_contact_offset_world, 9).tolist(),
        "right_handoff_contact_offset_world_xyz": np.round(right_handoff_contact_offset_world, 9).tolist(),
        "right_forward_extension_m": RIGHT_FORWARD_EXTENSION_M,
        "right_forward_axis_world_xyz": np.round(forward_axis, 9).tolist(),
    }

    samples, total = build_timeline(paths, args.fps)
    validation = annotate_tray(samples, arm_chain, base_world, link6_to_pad, calibration)

    trajectory = {
        "schema": "tray_handoff_curobo_demo/v3",
        "scene": args.scene,
        "arm_urdf": DEFAULT_ARM_URDF,
        "curobo_cfg": str(CUROBO_CFG),
        "curobo_urdf": DEFAULT_CUROBO_URDF,
        "fps": args.fps,
        "loop_duration_sec": round(total, 6),
        "joint_names": ARM_JOINTS,
        "tray": {
            "path": "/World/DemoTray",
            "start_world_xyz": np.round(tray_start_world, 9).tolist(),
            "lift_world_xyz": np.round(tray_lift_world, 9).tolist(),
            "handoff_world_xyz_expected": np.round(T_left_tray_at_handoff[:3, 3], 9).tolist(),
            "forward_ready_world_xyz_desired": np.round(tray_forward_ready_world, 9).tolist(),
            "final_world_xyz_desired": np.round(tray_forward_extend_world, 9).tolist(),
            "forward_extend_world_xyz_desired": np.round(tray_forward_extend_pose[:3, 3], 9).tolist(),
            "pad_clearance_m": TRAY_PAD_CLEARANCE_M,
            "handoff_gripper_separation_m": HANDOFF_GRIPPER_SEPARATION_M,
            "right_forward_extension_m": RIGHT_FORWARD_EXTENSION_M,
        },
        "calibration": calibration,
        "path_sources": path_sources,
        "validation_expected": validation,
        "samples": samples,
    }
    report = {
        "trajectory": {
            "path": args.out,
            "sample_count": len(samples),
            "loop_duration_sec": round(total, 6),
        },
        "ik": ik_report,
        "curobo": {"left": left_status, "right": right_status},
        "obstacles": obstacle_report,
        "path_sources": path_sources,
        "constrained_path": {
            name: {
                "sample_count": len(rows),
                "max_position_error_m": max((float(r["position_error_m"]) for r in rows), default=0.0),
                "max_pad_axis_up_error_deg": max(
                    (float(r["pad_axis_up_error_deg"]) for r in rows), default=0.0
                ),
                "max_pad_forward_axis_error_deg": max(
                    (float(r.get("pad_forward_axis_error_deg", 0.0)) for r in rows), default=0.0
                ),
            }
            for name, rows in constrained_path_report.items()
        },
        "validation_expected": validation,
        "scene": args.scene,
    }

    out = Path(args.out)
    report_path = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    print(f"wrote {report_path}")
    for name, item in ik_report.items():
        print(f"{name}: IK err={item['position_error_m']:.9f} q={np.round(q_targets[name], 4).tolist()}")
    print("cuRobo left", left_status)
    print("cuRobo right", right_status)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
