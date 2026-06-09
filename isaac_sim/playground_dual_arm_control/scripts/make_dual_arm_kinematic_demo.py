#!/usr/bin/env python3
"""Generate a verified zero -> common target -> straight demo trajectory."""

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

from ik_sanity import joint_limits
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


def parse_xyz(text: str) -> np.ndarray:
    parts = [float(v) for v in text.split(",") if v.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got {text!r}")
    return np.array(parts, dtype=float)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def selected_pad_midpoint(stage, cache, side: str) -> Tuple[np.ndarray, np.ndarray]:
    root = f"/World/robot/jaka_minicobo_{side}"
    link6 = get_world_pose(stage, cache, f"{root}/Link_6")
    left = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/left_pad")
    right = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/right_pad")
    if link6 is None or left is None or right is None:
        raise RuntimeError(f"Could not read Link_6/pads for {side}")
    mid = np.array(left, copy=True)
    mid[:3, 3] = (left[:3, 3] + right[:3, 3]) * 0.5
    return mid, relative_pose(link6, mid)


def solve_pad_ik(
    chain,
    lower,
    upper,
    base_world: np.ndarray,
    link6_to_pad_mid: np.ndarray,
    target_world_xyz: np.ndarray,
    seeds: List[np.ndarray],
) -> Tuple[np.ndarray, float, bool, str]:
    target_world = np.eye(4)
    target_world[:3, 3] = target_world_xyz
    target_base = np.linalg.inv(base_world) @ target_world

    def residual(q: np.ndarray) -> np.ndarray:
        candidate = fk(chain, dict(zip(ARM_JOINTS, q.tolist()))) @ link6_to_pad_mid
        return candidate[:3, 3] - target_base[:3, 3]

    best = None
    for seed in seeds:
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=1200,
        )
        err = float(np.linalg.norm(residual(result.x)))
        score = err + 1e-4 * float(np.linalg.norm(result.x))
        if best is None or score < best[0]:
            best = (score, result.x, err, bool(result.success), result.message)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def build_samples(
    q_target_by_side: Dict[str, np.ndarray],
    fps: float,
    hold_start: float,
    reach: float,
    hold_target: float,
    ret: float,
    hold_end: float,
) -> List[Dict[str, object]]:
    total = hold_start + reach + hold_target + ret + hold_end
    count = int(round(total * fps)) + 1
    samples: List[Dict[str, object]] = []
    for i in range(count):
        t = i / fps
        if t < hold_start:
            phase = "zero_hold"
            alpha = 0.0
        elif t < hold_start + reach:
            phase = "reach_common_target"
            alpha = smoothstep((t - hold_start) / reach)
        elif t < hold_start + reach + hold_target:
            phase = "target_hold"
            alpha = 1.0
        elif t < hold_start + reach + hold_target + ret:
            phase = "return_straight"
            alpha = 1.0 - smoothstep((t - hold_start - reach - hold_target) / ret)
        else:
            phase = "straight_hold"
            alpha = 0.0

        row = {"time_sec": round(t, 6), "phase": phase, "alpha": round(alpha, 9), "arms": {}}
        for side, q_target in q_target_by_side.items():
            q = alpha * q_target
            row["arms"][side] = {
                "joint_names": ARM_JOINTS,
                "joint_position_rad": np.round(q, 9).tolist(),
            }
        samples.append(row)
    return samples


def validate_samples(
    chain,
    base_world_by_side: Dict[str, np.ndarray],
    link6_to_pad_by_side: Dict[str, np.ndarray],
    target_world_xyz: np.ndarray,
    samples: List[Dict[str, object]],
) -> Dict[str, object]:
    target_pose = np.eye(4)
    target_pose[:3, 3] = target_world_xyz
    max_target_hold_error = 0.0
    max_link_step_rad = 0.0
    previous: Dict[str, np.ndarray] = {}
    target_hold_rows = 0

    for row in samples:
        for side in ("left", "right"):
            q = np.array(row["arms"][side]["joint_position_rad"], dtype=float)
            if side in previous:
                max_link_step_rad = max(max_link_step_rad, float(np.max(np.abs(q - previous[side]))))
            previous[side] = q
            if row["phase"] == "target_hold":
                target_hold_rows += 1
                T = base_world_by_side[side] @ fk(chain, dict(zip(ARM_JOINTS, q.tolist()))) @ link6_to_pad_by_side[side]
                max_target_hold_error = max(max_target_hold_error, diff_pose(target_pose, T)["position_error_m"])

    return {
        "target_hold_rows": target_hold_rows,
        "max_target_hold_pad_midpoint_error_m": max_target_hold_error,
        "max_joint_step_rad": max_link_step_rad,
        "max_joint_step_deg": math.degrees(max_link_step_rad),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--arm-urdf", default=DEFAULT_ARM_URDF)
    parser.add_argument("--target-world", default="0.0,0.65,1.20")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hold-start", type=float, default=1.5)
    parser.add_argument("--reach-sec", type=float, default=6.0)
    parser.add_argument("--hold-target", type=float, default=2.0)
    parser.add_argument("--return-sec", type=float, default=6.0)
    parser.add_argument("--hold-end", type=float, default=1.5)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--out", default=str(PLAYGROUND_ROOT / "runtime/dual_arm_common_target_trajectory.json"))
    parser.add_argument("--report", default=str(PLAYGROUND_ROOT / "reports/dual_arm_common_target_plan.json"))
    args = parser.parse_args()

    target_world_xyz = parse_xyz(args.target_world)
    stage = Usd.Stage.Open(str(Path(args.scene)))
    if stage is None:
        raise RuntimeError(f"Could not open USD scene: {args.scene}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    joints = load_joints(Path(args.arm_urdf))
    chain = chain_to_link(joints, "Link_0", "Link_6")
    lower, upper = joint_limits(chain)

    base_world_by_side = {}
    current_pad_by_side = {}
    link6_to_pad_by_side = {}
    for side in ("left", "right"):
        root = f"/World/robot/jaka_minicobo_{side}"
        base_world = get_world_pose(stage, cache, f"{root}/Link_0")
        if base_world is None:
            raise RuntimeError(f"Could not read base for {side}")
        pad_mid, link6_to_pad = selected_pad_midpoint(stage, cache, side)
        base_world_by_side[side] = base_world
        current_pad_by_side[side] = pad_mid
        link6_to_pad_by_side[side] = link6_to_pad

    generic_seeds = [
        np.zeros(6),
        np.array([1.5, 0.15, 1.7, -2.0, 0.1, -0.6]),
        np.array([1.6, 0.13, 1.73, -1.83, 0.11, -0.61]),
        np.array([-1.6, 0.13, 1.74, 2.43, 0.05, 0.63]),
        np.array([-1.5, -0.2, -1.3, 2.5, 0.5, 0.6]),
        np.array([1.2, -0.7, -0.7, 1.0, -1.0, -0.7]),
        np.array([-1.2, 0.7, 0.7, -1.0, 1.0, 0.7]),
    ]

    q_target_by_side = {}
    ik_report = {}
    for side in ("left", "right"):
        q, err, optimizer_success, message = solve_pad_ik(
            chain,
            lower,
            upper,
            base_world_by_side[side],
            link6_to_pad_by_side[side],
            target_world_xyz,
            generic_seeds,
        )
        if err > args.tolerance:
            raise RuntimeError(f"{side} IK error {err:.6f} m exceeds tolerance {args.tolerance:.6f} m")
        q_target_by_side[side] = q
        solved = base_world_by_side[side] @ fk(chain, dict(zip(ARM_JOINTS, q.tolist()))) @ link6_to_pad_by_side[side]
        ik_report[side] = {
            "optimizer_success": optimizer_success,
            "message": message,
            "joint_names": ARM_JOINTS,
            "target_joint_position_rad": np.round(q, 9).tolist(),
            "target_joint_position_deg": np.round(np.degrees(q), 6).tolist(),
            "target_error_m": err,
            "current_pad_midpoint_world": pose_dict(current_pad_by_side[side]),
            "solved_pad_midpoint_world": pose_dict(solved),
        }

    samples = build_samples(
        q_target_by_side,
        args.fps,
        args.hold_start,
        args.reach_sec,
        args.hold_target,
        args.return_sec,
        args.hold_end,
    )
    validation = validate_samples(
        chain,
        base_world_by_side,
        link6_to_pad_by_side,
        target_world_xyz,
        samples,
    )

    trajectory = {
        "schema": "dual_arm_kinematic_demo/v1",
        "scene": args.scene,
        "arm_urdf": args.arm_urdf,
        "target_frame": "pad_midpoint",
        "target_world_xyz": np.round(target_world_xyz, 9).tolist(),
        "fps": float(args.fps),
        "loop_duration_sec": samples[-1]["time_sec"] if samples else 0.0,
        "joint_names": ARM_JOINTS,
        "samples": samples,
    }
    report = {
        "trajectory": {
            "path": str(Path(args.out)),
            "sample_count": len(samples),
            "loop_duration_sec": trajectory["loop_duration_sec"],
        },
        "target_world_xyz": trajectory["target_world_xyz"],
        "ik": ik_report,
        "validation": validation,
    }

    out = Path(args.out)
    report_path = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"target_world={trajectory['target_world_xyz']}")
    for side, side_report in ik_report.items():
        print(
            f"{side}: target_err={side_report['target_error_m']:.9f} m "
            f"q_rad={np.round(q_target_by_side[side], 6).tolist()}"
        )
    print(
        "validation: "
        f"target_hold_max_err={validation['max_target_hold_pad_midpoint_error_m']:.9f} m, "
        f"max_step={validation['max_joint_step_deg']:.4f} deg"
    )
    print(f"wrote {out}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
