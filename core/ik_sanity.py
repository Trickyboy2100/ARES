#!/usr/bin/env python3
"""Solve a small URDF IK problem for one arm in the current USD world frame."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pxr import Usd, UsdGeom
from scipy.optimize import least_squares

from kinematics_probe import (
    ARM_JOINTS,
    DEFAULT_ARM_URDF,
    DEFAULT_SCENE,
    GRIPPER_ROOT_SUFFIX,
    chain_to_link,
    diff_pose,
    fk,
    get_world_pose,
    load_joints,
    pose_dict,
    relative_pose,
)


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]


FRAME_SUFFIXES = {
    "bare": "Link_6/dummy_tcp",
    "link6": "Link_6",
    "gripper_root": GRIPPER_ROOT_SUFFIX,
    "pad_midpoint": None,
}


def parse_xyz(text: str) -> np.ndarray:
    parts = [float(v) for v in text.split(",") if v.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z, got {text!r}")
    return np.array(parts, dtype=float)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    cos_angle = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    axis /= 2.0 * math.sin(angle)
    return axis * angle


def joint_limits(chain) -> Tuple[np.ndarray, np.ndarray]:
    lowers: List[float] = []
    uppers: List[float] = []
    for joint in chain:
        if joint.name not in ARM_JOINTS:
            continue
        lowers.append(joint.lower if joint.lower is not None else -2.0 * math.pi)
        uppers.append(joint.upper if joint.upper is not None else 2.0 * math.pi)
    return np.array(lowers, dtype=float), np.array(uppers, dtype=float)


def selected_frame_pose(
    stage: Usd.Stage,
    cache: UsdGeom.XformCache,
    side: str,
    frame: str,
) -> np.ndarray:
    root = f"/World/robot/jaka_minicobo_{side}"
    if frame == "pad_midpoint":
        left = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/left_pad")
        right = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/right_pad")
        if left is None or right is None:
            raise RuntimeError("Could not read left/right pad poses")
        T = np.array(left, copy=True)
        T[:3, 3] = (left[:3, 3] + right[:3, 3]) * 0.5
        return T
    suffix = FRAME_SUFFIXES[frame]
    pose = get_world_pose(stage, cache, f"{root}/{suffix}")
    if pose is None:
        raise RuntimeError(f"Could not read frame {frame!r} for {side}")
    return pose


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--arm-urdf", default=DEFAULT_ARM_URDF)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument(
        "--frame",
        choices=["bare", "link6", "gripper_root", "pad_midpoint"],
        default="bare",
        help="Frame that should reach --target-world.",
    )
    parser.add_argument(
        "--target-world",
        default=None,
        help="Target position x,y,z in USD world. Defaults to the selected frame's current position.",
    )
    parser.add_argument(
        "--match-orientation",
        action="store_true",
        help="Also match the selected frame's current orientation.",
    )
    parser.add_argument("--seed", default="0,0,0,0,0,0")
    parser.add_argument("--position-tolerance", type=float, default=1e-3)
    parser.add_argument("--out", default=str(PLAYGROUND_ROOT / "reports/ik_sanity.json"))
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(Path(args.scene)))
    if stage is None:
        raise RuntimeError(f"Could not open USD scene: {args.scene}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    root = f"/World/robot/jaka_minicobo_{args.side}"
    base_world = get_world_pose(stage, cache, f"{root}/Link_0")
    link6_world = get_world_pose(stage, cache, f"{root}/Link_6")
    if base_world is None or link6_world is None:
        raise RuntimeError(f"Could not read base/link6 for {args.side}")

    current_frame_world = selected_frame_pose(stage, cache, args.side, args.frame)
    T_link6_frame = relative_pose(link6_world, current_frame_world)

    target_world = np.array(current_frame_world, copy=True)
    if args.target_world:
        target_world[:3, 3] = parse_xyz(args.target_world)

    joints = load_joints(Path(args.arm_urdf))
    chain = chain_to_link(joints, "Link_0", "Link_6")
    lower, upper = joint_limits(chain)
    seed_values = [float(v) for v in args.seed.split(",") if v.strip()]
    if len(seed_values) != 6:
        raise ValueError("--seed must contain six comma-separated joint values")
    seed = np.array(seed_values, dtype=float)

    target_in_base = np.linalg.inv(base_world) @ target_world

    def residual(q: np.ndarray) -> np.ndarray:
        q_by_name: Dict[str, float] = dict(zip(ARM_JOINTS, q.tolist()))
        candidate = fk(chain, q_by_name) @ T_link6_frame
        pos_error = candidate[:3, 3] - target_in_base[:3, 3]
        if not args.match_orientation:
            return pos_error
        R_error = target_in_base[:3, :3].T @ candidate[:3, :3]
        return np.concatenate([pos_error, 0.25 * rotvec_from_matrix(R_error)])

    result = least_squares(
        residual,
        seed,
        bounds=(lower, upper),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=2000,
    )

    q_by_name = dict(zip(ARM_JOINTS, result.x.tolist()))
    solved_link6_world = base_world @ fk(chain, q_by_name)
    solved_frame_world = solved_link6_world @ T_link6_frame
    pose_error = diff_pose(target_world, solved_frame_world)
    position_goal_reached = pose_error["position_error_m"] <= float(args.position_tolerance)
    orientation_goal_reached = (
        pose_error["rotation_error_rad"] <= 1e-3 if args.match_orientation else None
    )

    report = {
        "inputs": {
            "scene": args.scene,
            "arm_urdf": args.arm_urdf,
            "side": args.side,
            "frame": args.frame,
            "target_world_position_xyz": target_world[:3, 3].round(9).tolist(),
            "match_orientation": bool(args.match_orientation),
            "seed_rad": seed.round(9).tolist(),
        },
        "solution": {
            "success": bool(result.success) or bool(position_goal_reached and not args.match_orientation),
            "optimizer_success": bool(result.success),
            "position_goal_reached": bool(position_goal_reached),
            "orientation_goal_reached": orientation_goal_reached,
            "status": int(result.status),
            "message": result.message,
            "joint_names": ARM_JOINTS,
            "joint_position_rad": np.round(result.x, 9).tolist(),
            "joint_position_deg": np.round(np.degrees(result.x), 6).tolist(),
            "residual_norm": float(np.linalg.norm(result.fun)),
            "pose_error": pose_error,
            "solved_link6_world": pose_dict(solved_link6_world),
            "solved_selected_frame_world": pose_dict(solved_frame_world),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"{args.side} {args.frame}: success={bool(result.success) or bool(position_goal_reached and not args.match_orientation)} "
        f"optimizer_success={result.success} residual={np.linalg.norm(result.fun):.9g}"
    )
    print(f"  q_rad={np.round(result.x, 6).tolist()}")
    print(
        "  pose_error="
        f"{pose_error['position_error_m']:.9g} m, "
        f"{pose_error['rotation_error_deg']:.6f} deg"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
