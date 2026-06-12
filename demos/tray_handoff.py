#!/usr/bin/env python3
"""GUI playback for the cuRobo-backed tray handoff trajectory."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
import sys
_CORE_DIR = str(Path(__file__).resolve().parents[1] / "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)


import numpy as np

from kinematics_probe import (
    ARM_JOINTS,
    chain_to_link,
    fk,
    get_world_pose,
    load_joints,
    matrix_to_quat_wxyz,
)


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = PLAYGROUND_ROOT / "runtime/tray_handoff_curobo_trajectory.json"
DEFAULT_OUT_DIR = PLAYGROUND_ROOT / "logs/tray_handoff_curobo_demo"
ARM_ROOTS = {
    "left": "/World/robot/jaka_minicobo_left",
    "right": "/World/robot/jaka_minicobo_right",
}
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]
GRIPPER_SUFFIX = "Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2"
GRIPPER_LINK_JOINT_CHAINS = {
    "left_outer_link": [
        ((-0.04, -0.009, 0.079), (0.0, -1.0, 0.0)),
    ],
    "left_inner_link": [
        ((-0.03, -0.009, 0.081), (0.0, -1.0, 0.0)),
    ],
    "right_inner_link": [
        ((0.03, -0.009, 0.081), (0.0, 1.0, 0.0)),
    ],
    "right_outer_link": [
        ((0.04, -0.009, 0.079), (0.0, 1.0, 0.0)),
    ],
    "left_pad": [
        ((-0.04, -0.009, 0.079), (0.0, -1.0, 0.0)),
        ((0.0223, 0.003, 0.035591), (0.0, 1.0, 0.0)),
    ],
    "right_pad": [
        ((0.04, -0.009, 0.079), (0.0, 1.0, 0.0)),
        ((-0.0223, 0.003, 0.035591), (0.0, -1.0, 0.0)),
    ],
}
# q=0 is the imported closed pose. q=0.624 rad gives the same ~85.4 mm
# opening used in the previous visual demo while preserving the linkage.
GRIPPER_OPEN_ANGLE_RAD = 0.6240523114116221


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--scene", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def atomic_write_json(path: Path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def gf_matrix_from_column_transform(T):
    from pxr import Gf

    return Gf.Matrix4d(*sum(np.asarray(T, dtype=float).T.tolist(), []))


def translate_matrix(xyz):
    T = np.eye(4)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def axis_angle_matrix(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(1e-12, float(np.linalg.norm(axis)))
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    R = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=float,
    )
    T = np.eye(4)
    T[:3, :3] = R
    return T


def gripper_link_transform(link_name: str, joint_angle_rad: float):
    T = np.eye(4)
    for xyz, axis in GRIPPER_LINK_JOINT_CHAINS[link_name]:
        T = T @ translate_matrix(xyz) @ axis_angle_matrix(axis, joint_angle_rad)
    return T


def axis_angle_error_deg(axis, target=np.array([0.0, 0.0, 1.0])):
    axis = np.asarray(axis, dtype=float)
    target = np.asarray(target, dtype=float)
    denom = max(1e-12, float(np.linalg.norm(axis) * np.linalg.norm(target)))
    cos_angle = float(np.dot(axis, target) / denom)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_angle)))))


def transform_delta_deg(a, b):
    rot_delta = a[:3, :3].T @ b[:3, :3]
    cos_angle = (float(np.trace(rot_delta)) - 1.0) * 0.5
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos_angle)))))


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def vec(xyz):
    return [round(float(v), 6) for v in xyz]


def transform_point(T, xyz):
    p = np.r_[np.array(xyz, dtype=float), 1.0]
    return (np.asarray(T, dtype=float) @ p)[:3]


def load_trajectory(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("samples"):
        raise RuntimeError(f"Trajectory has no samples: {path}")
    if "calibration" not in payload:
        raise RuntimeError(f"Trajectory is missing v2 calibration data: {path}")
    return payload


def interpolate_sample(samples, elapsed: float, loop_duration: float, should_loop: bool):
    t = elapsed % loop_duration if should_loop else min(elapsed, loop_duration)
    if t <= samples[0]["time_sec"]:
        return samples[0]
    if t >= samples[-1]["time_sec"]:
        return samples[-1]
    lo, hi = 0, len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid]["time_sec"] <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    span = max(1e-9, float(b["time_sec"]) - float(a["time_sec"]))
    alpha = (t - float(a["time_sec"])) / span
    row = {
        "time_sec": t,
        "phase": a["phase"] if alpha < 0.5 else b["phase"],
        "tray_attachment": a["tray_attachment"] if alpha < 0.5 else b["tray_attachment"],
        "arms": {},
    }
    for side in ("left", "right"):
        qa = np.array(a["arms"][side]["joint_position_rad"], dtype=float)
        qb = np.array(b["arms"][side]["joint_position_rad"], dtype=float)
        q = (1.0 - alpha) * qa + alpha * qb
        row["arms"][side] = {
            "joint_position_rad": q,
        }
    return row


def tray_transform_from_attachment(attachment, expected_pad, calibration):
    tray_start = np.array(calibration["tray_start_world_matrix4x4"], dtype=float)
    pad_to_tray = {
        side: np.array(calibration["pad_to_tray_matrix4x4"][side], dtype=float)
        for side in ("left", "right")
    }
    if attachment == "table":
        return tray_start, None
    if attachment == "left":
        return expected_pad["left"] @ pad_to_tray["left"], None
    if attachment == "right":
        return expected_pad["right"] @ pad_to_tray["right"], None
    left_tray = expected_pad["left"] @ pad_to_tray["left"]
    right_tray = expected_pad["right"] @ pad_to_tray["right"]
    return left_tray, right_tray


def build_phase_ranges(samples):
    ranges = {}
    for row in samples:
        phase = row["phase"]
        t = float(row["time_sec"])
        if phase not in ranges:
            ranges[phase] = [t, t]
        else:
            ranges[phase][1] = t
    return ranges


def phase_alpha(sample, phase_ranges):
    phase = sample["phase"]
    start, end = phase_ranges.get(phase, [float(sample["time_sec"]), float(sample["time_sec"])])
    span = max(1e-9, end - start)
    return max(0.0, min(1.0, (float(sample["time_sec"]) - start) / span))


def gripper_closed_fraction(sample, phase_ranges):
    phase = sample["phase"]
    alpha = phase_alpha(sample, phase_ranges)
    left = 0.0
    right = 0.0

    if phase == "left_descend_to_tray":
        left = smoothstep(alpha)
    elif phase in {"left_lift_tray", "move_to_chest_handoff", "handoff_hold"}:
        left = 1.0
    elif phase == "right_takes_tray":
        left = 1.0 - smoothstep(min(1.0, alpha * 2.0))

    if phase == "handoff_hold":
        right = smoothstep(min(1.0, alpha * 2.0))
    elif phase in {"right_takes_tray", "right_face_forward", "right_forward_extend", "right_horizontal_hold"}:
        right = 1.0

    return {"left": left, "right": right}


def gripper_joint_angle_from_closed_fraction(closed_fraction):
    return GRIPPER_OPEN_ANGLE_RAD * (1.0 - float(closed_fraction))


def main():
    args = parse_args()
    trajectory_path = Path(args.trajectory)
    trajectory = load_trajectory(trajectory_path)
    scene = args.scene or trajectory["scene"]
    samples = trajectory["samples"]
    calibration = trajectory["calibration"]
    phase_ranges = build_phase_ranges(samples)
    loop_duration = float(trajectory["loop_duration_sec"])
    should_loop = not args.no_loop

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "motion_log.jsonl"
    heartbeat_path = out_dir / "heartbeat.json"
    summary_path = out_dir / "summary.json"

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={"renderer": "RaytracedLighting", "headless": args.headless}
    )

    from pxr import Usd, UsdGeom
    import omni.kit.app
    import omni.usd

    keep_running = True

    def handle_signal(_signum, _frame):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    print(f"[TRAY-DEMO] Opening scene: {scene}", flush=True)
    ctx.open_stage(scene)

    stage = None
    for _ in range(220):
        app.update()
        stage = ctx.get_stage()
        if stage and stage.GetPrimAtPath("/World/DemoTray").IsValid() and all(
            stage.GetPrimAtPath(path).IsValid() for path in ARM_ROOTS.values()
        ):
            break
        time.sleep(0.05)
    if stage is None:
        raise RuntimeError("No stage after opening scene")

    arm_joints = load_joints(Path(trajectory["arm_urdf"]))
    chains = {name: chain_to_link(arm_joints, "Link_0", name) for name in LINK_NAMES}
    link6_chain = chains["Link_6"]
    base_world = {
        side: np.array(calibration["base_world_matrix4x4"][side], dtype=float)
        for side in ("left", "right")
    }
    link6_to_pad = {
        side: np.array(calibration["link6_to_pad_matrix4x4"][side], dtype=float)
        for side in ("left", "right")
    }

    link_ops = {}
    for side, root in ARM_ROOTS.items():
        link_ops[side] = {}
        for link_name in LINK_NAMES:
            prim = stage.GetPrimAtPath(f"{root}/{link_name}")
            if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                raise RuntimeError(f"Missing xformable link: {root}/{link_name}")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            link_ops[side][link_name] = xf.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble, "tray_demo_fk"
            )

    gripper_ops = {}
    for side, root in ARM_ROOTS.items():
        gripper_ops[side] = {}
        for link_name in GRIPPER_LINK_JOINT_CHAINS:
            path = f"{root}/{GRIPPER_SUFFIX}/{link_name}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                continue
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            found = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "tray_demo_gripper_fk")
            gripper_ops[side][link_name] = found

    def set_gripper_joint_angle(side: str, joint_angle_rad: float):
        touched = 0
        for link_name in GRIPPER_LINK_JOINT_CHAINS:
            op = gripper_ops.get(side, {}).get(link_name)
            if op is None:
                continue
            op.Set(gf_matrix_from_column_transform(gripper_link_transform(link_name, joint_angle_rad)))
            touched += 1
        return touched

    tray_prim = stage.GetPrimAtPath(trajectory["tray"]["path"])
    tray_xf = UsdGeom.Xformable(tray_prim)
    tray_xf.ClearXformOpOrder()
    tray_op = tray_xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "tray_demo")

    def read_actual_gripper_frame(cache, side: str, nominal_frame: np.ndarray):
        root = ARM_ROOTS[side]
        left_pad = get_world_pose(stage, cache, f"{root}/{GRIPPER_SUFFIX}/left_pad")
        right_pad = get_world_pose(stage, cache, f"{root}/{GRIPPER_SUFFIX}/right_pad")
        frame = np.array(nominal_frame, copy=True)
        if left_pad is None or right_pad is None:
            return frame, {
                "pad_midpoint_world_xyz": vec(frame[:3, 3]),
                "pad_origin_gap_m": None,
                "pad_probe_gap_m": None,
                "nominal_midpoint_offset_m": 0.0,
            }
        midpoint = (left_pad[:3, 3] + right_pad[:3, 3]) * 0.5
        left_probe = transform_point(left_pad, [0.0, 0.025, 0.025])
        right_probe = transform_point(right_pad, [0.0, 0.025, 0.025])
        frame[:3, 3] = midpoint
        return frame, {
            "pad_midpoint_world_xyz": vec(midpoint),
            "left_pad_world_xyz": vec(left_pad[:3, 3]),
            "right_pad_world_xyz": vec(right_pad[:3, 3]),
            "left_pad_probe_world_xyz": vec(left_probe),
            "right_pad_probe_world_xyz": vec(right_probe),
            "pad_origin_gap_m": float(np.linalg.norm(left_pad[:3, 3] - right_pad[:3, 3])),
            "pad_probe_gap_m": float(np.linalg.norm(left_probe - right_probe)),
            "nominal_midpoint_offset_m": float(np.linalg.norm(midpoint - nominal_frame[:3, 3])),
        }

    def read_gripper_linkage(cache, side: str):
        root = ARM_ROOTS[side]
        gripper_root = get_world_pose(stage, cache, f"{root}/{GRIPPER_SUFFIX}")
        if gripper_root is None:
            return {}
        root_inv = np.linalg.inv(gripper_root)
        linkage = {}
        for link_name in GRIPPER_LINK_JOINT_CHAINS:
            link_pose = get_world_pose(stage, cache, f"{root}/{GRIPPER_SUFFIX}/{link_name}")
            if link_pose is None:
                continue
            linkage[link_name] = vec(transform_point(root_inv, link_pose[:3, 3]))
        return linkage

    start = time.time()
    frame = 0
    max_pad_expected_error = {"left": 0.0, "right": 0.0}
    max_tray_expected_error = 0.0
    max_tray_level_error_deg = 0.0
    max_both_attachment_agreement_m = 0.0
    max_both_attachment_agreement_deg = 0.0
    max_gripper_nominal_midpoint_offset_m = {"left": 0.0, "right": 0.0}
    gripper_pad_gap_range_m = {
        "left": {"min": float("inf"), "max": 0.0},
        "right": {"min": float("inf"), "max": 0.0},
    }
    gripper_pad_probe_gap_range_m = {
        "left": {"min": float("inf"), "max": 0.0},
        "right": {"min": float("inf"), "max": 0.0},
    }
    gripper_joint_angle_range_deg = {
        "left": {"min": float("inf"), "max": float("-inf")},
        "right": {"min": float("inf"), "max": float("-inf")},
    }
    last_row = None
    log_file = open(log_path, "w", encoding="utf-8")

    def clean_range(payload):
        cleaned = {}
        for side, values in payload.items():
            mn = values["min"]
            mx = values["max"]
            cleaned[side] = {
                "min": None if not np.isfinite(mn) else round(float(mn), 9),
                "max": None if not np.isfinite(mx) else round(float(mx), 9),
            }
        return cleaned

    print(
        "[TRAY-DEMO] Playing cuRobo-backed tray handoff: table -> chest -> right hold. "
        f"loop={should_loop}",
        flush=True,
    )
    try:
        while keep_running:
            elapsed = time.time() - start
            if args.duration_sec > 0 and elapsed >= args.duration_sec:
                break
            if args.no_loop and elapsed > loop_duration + 0.2:
                break

            sample = interpolate_sample(samples, elapsed, loop_duration, should_loop)
            nominal_pad = {}
            for side in ("left", "right"):
                q = np.array(sample["arms"][side]["joint_position_rad"], dtype=float)
                q_map = dict(zip(ARM_JOINTS, q.tolist()))
                for link_name in LINK_NAMES:
                    link_ops[side][link_name].Set(gf_matrix_from_column_transform(fk(chains[link_name], q_map)))
                nominal_pad[side] = base_world[side] @ fk(link6_chain, q_map) @ link6_to_pad[side]

            closed_fraction = gripper_closed_fraction(sample, phase_ranges)
            gripper_command = {}
            for side in ("left", "right"):
                joint_angle = gripper_joint_angle_from_closed_fraction(closed_fraction[side])
                touched = set_gripper_joint_angle(side, joint_angle)
                joint_angle_deg = float(np.degrees(joint_angle))
                gripper_command[side] = {
                    "closed_fraction": float(closed_fraction[side]),
                    "joint_angle_rad": float(joint_angle),
                    "joint_angle_deg": joint_angle_deg,
                    "visual_links_touched": touched,
                }
                gripper_joint_angle_range_deg[side]["min"] = min(
                    gripper_joint_angle_range_deg[side]["min"], joint_angle_deg
                )
                gripper_joint_angle_range_deg[side]["max"] = max(
                    gripper_joint_angle_range_deg[side]["max"], joint_angle_deg
                )

            app.update()
            control_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            actual_gripper_frame = {}
            gripper_readback = {}
            for side in ("left", "right"):
                actual_gripper_frame[side], gripper_readback[side] = read_actual_gripper_frame(
                    control_cache, side, nominal_pad[side]
                )

            tray_expected_T, right_tray_candidate_T = tray_transform_from_attachment(
                sample["tray_attachment"], actual_gripper_frame, calibration
            )
            tray_op.Set(gf_matrix_from_column_transform(tray_expected_T))
            app.update()

            if frame % max(1, args.log_every) == 0:
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                row = {
                    "unix": round(time.time(), 3),
                    "elapsed_sec": round(elapsed, 3),
                    "trajectory_time_sec": round(float(sample["time_sec"]), 3),
                    "frame": frame,
                    "phase": sample["phase"],
                    "tray_attachment": sample["tray_attachment"],
                    "control_mode": "curobo_backed_gui_fk_playback_visual_grippers_no_physx",
                    "arms": {},
                    "grippers": {},
                }
                for side, root in ARM_ROOTS.items():
                    q = np.array(sample["arms"][side]["joint_position_rad"], dtype=float)
                    nominal_pad_T = nominal_pad[side]
                    actual_pad_T, grip_info = read_actual_gripper_frame(cache, side, nominal_pad_T)
                    pad_expected = actual_pad_T[:3, 3]
                    # The current stage has the real pad prims; read them back.
                    left_pad = get_world_pose(stage, cache, f"{root}/Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2/left_pad")
                    right_pad = get_world_pose(stage, cache, f"{root}/Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2/right_pad")
                    link6 = get_world_pose(stage, cache, f"{root}/Link_6")
                    if left_pad is None or right_pad is None or link6 is None:
                        raise RuntimeError(f"Could not read actual pad/link pose for {side}")
                    pad_actual = (left_pad[:3, 3] + right_pad[:3, 3]) * 0.5
                    err = float(np.linalg.norm(pad_actual - pad_expected))
                    max_pad_expected_error[side] = max(max_pad_expected_error[side], err)
                    max_gripper_nominal_midpoint_offset_m[side] = max(
                        max_gripper_nominal_midpoint_offset_m[side],
                        float(grip_info["nominal_midpoint_offset_m"]),
                    )
                    if grip_info["pad_origin_gap_m"] is not None:
                        gap = float(grip_info["pad_origin_gap_m"])
                        gripper_pad_gap_range_m[side]["min"] = min(gripper_pad_gap_range_m[side]["min"], gap)
                        gripper_pad_gap_range_m[side]["max"] = max(gripper_pad_gap_range_m[side]["max"], gap)
                    if grip_info["pad_probe_gap_m"] is not None:
                        probe_gap = float(grip_info["pad_probe_gap_m"])
                        gripper_pad_probe_gap_range_m[side]["min"] = min(
                            gripper_pad_probe_gap_range_m[side]["min"], probe_gap
                        )
                        gripper_pad_probe_gap_range_m[side]["max"] = max(
                            gripper_pad_probe_gap_range_m[side]["max"], probe_gap
                        )
                    row["arms"][side] = {
                        "joint_names": ARM_JOINTS,
                        "joint_position_rad": np.round(q, 6).tolist(),
                        "joint_position_deg": np.round(np.degrees(q), 3).tolist(),
                        "link6_world_xyz": vec(link6[:3, 3]),
                        "pad_midpoint_world_xyz": vec(pad_actual),
                        "expected_pad_midpoint_world_xyz": vec(pad_expected),
                        "nominal_fk_pad_midpoint_world_xyz": vec(nominal_pad_T[:3, 3]),
                        "nominal_to_visual_pad_midpoint_offset_m": round(
                            float(grip_info["nominal_midpoint_offset_m"]), 9
                        ),
                        "pad_axis_up_error_deg_expected": round(axis_angle_error_deg(nominal_pad_T[:3, 1]), 9),
                        "pad_expected_error_m": round(err, 9),
                    }
                    row["grippers"][side] = {
                        "joint_angle_rad": round(float(gripper_command[side]["joint_angle_rad"]), 9),
                        "joint_angle_deg": round(float(gripper_command[side]["joint_angle_deg"]), 6),
                        "closed_fraction": round(float(gripper_command[side]["closed_fraction"]), 6),
                        "visual_links_touched": int(gripper_command[side]["visual_links_touched"]),
                        "pad_origin_gap_m": None
                        if grip_info["pad_origin_gap_m"] is None
                        else round(float(grip_info["pad_origin_gap_m"]), 9),
                        "pad_probe_gap_m": None
                        if grip_info["pad_probe_gap_m"] is None
                        else round(float(grip_info["pad_probe_gap_m"]), 9),
                        "left_pad_world_xyz": grip_info.get("left_pad_world_xyz"),
                        "right_pad_world_xyz": grip_info.get("right_pad_world_xyz"),
                        "left_pad_probe_world_xyz": grip_info.get("left_pad_probe_world_xyz"),
                        "right_pad_probe_world_xyz": grip_info.get("right_pad_probe_world_xyz"),
                        "link_root_local_xyz": read_gripper_linkage(cache, side),
                    }

                tray_world = get_world_pose(stage, cache, trajectory["tray"]["path"])
                if tray_world is None:
                    raise RuntimeError("Could not read DemoTray world pose")
                tray_actual = tray_world[:3, 3]
                tray_expected = tray_expected_T[:3, 3]
                tray_err = float(np.linalg.norm(tray_actual - tray_expected))
                up = tray_world[:3, 2]
                level_error = axis_angle_error_deg(up)
                max_tray_expected_error = max(max_tray_expected_error, tray_err)
                max_tray_level_error_deg = max(max_tray_level_error_deg, level_error)
                row["tray"] = {
                    "world_xyz": vec(tray_actual),
                    "expected_world_xyz": vec(tray_expected),
                    "expected_error_m": round(tray_err, 9),
                    "level_error_deg": round(level_error, 9),
                    "quaternion_wxyz_expected": [
                        round(float(v), 9) for v in matrix_to_quat_wxyz(tray_expected_T[:3, :3])
                    ],
                }
                if right_tray_candidate_T is not None:
                    agreement_m = float(np.linalg.norm(tray_expected_T[:3, 3] - right_tray_candidate_T[:3, 3]))
                    agreement_deg = transform_delta_deg(tray_expected_T, right_tray_candidate_T)
                    if (
                        gripper_command["left"]["closed_fraction"] >= 0.95
                        and gripper_command["right"]["closed_fraction"] >= 0.95
                    ):
                        max_both_attachment_agreement_m = max(max_both_attachment_agreement_m, agreement_m)
                        max_both_attachment_agreement_deg = max(max_both_attachment_agreement_deg, agreement_deg)
                    row["tray"]["right_candidate_world_xyz"] = vec(right_tray_candidate_T[:3, 3])
                    row["tray"]["both_attachment_agreement_m"] = round(agreement_m, 9)
                    row["tray"]["both_attachment_agreement_deg"] = round(agreement_deg, 9)
                log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_file.flush()
                last_row = row
                atomic_write_json(
                    heartbeat_path,
                    {
                        "status": "running",
                        "scene": scene,
                        "trajectory": str(trajectory_path),
                        "frame": frame,
                        "elapsed_sec": round(elapsed, 3),
                        "phase": sample["phase"],
                        "tray_attachment": sample["tray_attachment"],
                        "tray_world_xyz": row["tray"]["world_xyz"],
                        "left_pad_world_xyz": row["arms"]["left"]["pad_midpoint_world_xyz"],
                        "right_pad_world_xyz": row["arms"]["right"]["pad_midpoint_world_xyz"],
                        "grippers": {
                            side: {
                                "joint_angle_deg": row["grippers"][side]["joint_angle_deg"],
                                "closed_fraction": row["grippers"][side]["closed_fraction"],
                                "pad_origin_gap_m": row["grippers"][side]["pad_origin_gap_m"],
                                "pad_probe_gap_m": row["grippers"][side]["pad_probe_gap_m"],
                            }
                            for side in ("left", "right")
                        },
                        "max_pad_expected_error_m": max_pad_expected_error,
                        "max_tray_expected_error_m": max_tray_expected_error,
                        "max_tray_level_error_deg": max_tray_level_error_deg,
                        "max_both_attachment_agreement_m": max_both_attachment_agreement_m,
                        "max_both_attachment_agreement_deg": max_both_attachment_agreement_deg,
                        "max_gripper_nominal_midpoint_offset_m": max_gripper_nominal_midpoint_offset_m,
                        "gripper_pad_gap_range_m": clean_range(gripper_pad_gap_range_m),
                        "gripper_pad_probe_gap_range_m": clean_range(gripper_pad_probe_gap_range_m),
                        "gripper_joint_angle_range_deg": clean_range(gripper_joint_angle_range_deg),
                        "updated_unix": time.time(),
                    },
                )
            frame += 1
    finally:
        log_file.close()
        summary = {
            "status": "stopped",
            "scene": scene,
            "trajectory": str(trajectory_path),
            "frames": frame,
            "max_pad_expected_error_m": max_pad_expected_error,
            "max_tray_expected_error_m": max_tray_expected_error,
            "max_tray_level_error_deg": max_tray_level_error_deg,
            "max_both_attachment_agreement_m": max_both_attachment_agreement_m,
            "max_both_attachment_agreement_deg": max_both_attachment_agreement_deg,
            "max_gripper_nominal_midpoint_offset_m": max_gripper_nominal_midpoint_offset_m,
            "gripper_pad_gap_range_m": clean_range(gripper_pad_gap_range_m),
            "gripper_pad_probe_gap_range_m": clean_range(gripper_pad_probe_gap_range_m),
            "gripper_joint_angle_range_deg": clean_range(gripper_joint_angle_range_deg),
            "last_row": last_row,
            "updated_unix": time.time(),
        }
        atomic_write_json(summary_path, summary)
        atomic_write_json(heartbeat_path, summary)
        for _ in range(8):
            app.update()
        simulation_app.close()


if __name__ == "__main__":
    main()
