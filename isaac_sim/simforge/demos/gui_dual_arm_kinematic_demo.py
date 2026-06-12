#!/usr/bin/env python3
"""Isaac GUI playback for the verified dual-arm kinematic demo."""


# ── SimForge path injection ───────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_HERE = str(_Path(__file__).resolve().parent)
_SIMFORGE = str(_Path(__file__).resolve().parents[1])
_CORE = str(_Path(__file__).resolve().parents[1] / "core")
for _p in (_HERE, _SIMFORGE, _CORE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import numpy as np

from kinematics_probe import (
    ARM_JOINTS,
    DEFAULT_ARM_URDF,
    DEFAULT_SCENE,
    GRIPPER_ROOT_SUFFIX,
    chain_to_link,
    fk,
    get_world_pose,
    load_joints,
    relative_pose,
)


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = PLAYGROUND_ROOT / "runtime/dual_arm_common_target_trajectory.json"
DEFAULT_OUT_DIR = PLAYGROUND_ROOT / "logs/dual_arm_kinematic_demo"
ARM_ROOTS = {
    "left": "/World/robot/jaka_minicobo_left",
    "right": "/World/robot/jaka_minicobo_right",
}
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--arm-urdf", default=DEFAULT_ARM_URDF)
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


def vec(value):
    return [round(float(v), 6) for v in value]


def load_trajectory(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    if not samples:
        raise RuntimeError(f"Trajectory has no samples: {path}")
    return payload


def interpolate_q(samples, elapsed: float, loop_duration: float, should_loop: bool):
    if should_loop:
        t = elapsed % loop_duration
    else:
        t = min(elapsed, loop_duration)

    if t <= samples[0]["time_sec"]:
        return samples[0]
    if t >= samples[-1]["time_sec"]:
        return samples[-1]

    lo = 0
    hi = len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid]["time_sec"] <= t:
            lo = mid
        else:
            hi = mid

    a = samples[lo]
    b = samples[hi]
    span = max(1e-9, float(b["time_sec"]) - float(a["time_sec"]))
    alpha = (t - float(a["time_sec"])) / span
    row = {"time_sec": t, "phase": a["phase"] if alpha < 0.5 else b["phase"], "arms": {}}
    for side in ("left", "right"):
        qa = np.array(a["arms"][side]["joint_position_rad"], dtype=float)
        qb = np.array(b["arms"][side]["joint_position_rad"], dtype=float)
        q = (1.0 - alpha) * qa + alpha * qb
        row["arms"][side] = {"joint_position_rad": q}
    return row


def pad_midpoint_pose(stage, cache, side: str):
    root = ARM_ROOTS[side]
    left = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/left_pad")
    right = get_world_pose(stage, cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/right_pad")
    if left is None or right is None:
        return None
    midpoint = np.array(left, copy=True)
    midpoint[:3, 3] = (left[:3, 3] + right[:3, 3]) * 0.5
    return midpoint


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "motion_log.jsonl"
    heartbeat_path = out_dir / "heartbeat.json"
    summary_path = out_dir / "summary.json"

    trajectory_path = Path(args.trajectory)
    trajectory = load_trajectory(trajectory_path)
    samples = trajectory["samples"]
    loop_duration = float(trajectory["loop_duration_sec"])
    target_world = np.array(trajectory["target_world_xyz"], dtype=float)
    should_loop = not args.no_loop

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
    print(f"[KIN-DEMO] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)

    stage = None
    for _ in range(200):
        app.update()
        stage = ctx.get_stage()
        if stage and all(stage.GetPrimAtPath(path).IsValid() for path in ARM_ROOTS.values()):
            break
        time.sleep(0.05)
    if stage is None:
        raise RuntimeError("No stage after opening scene")

    joints = load_joints(Path(args.arm_urdf))
    chains = {name: chain_to_link(joints, "Link_0", name) for name in LINK_NAMES}

    link_ops = {}
    for side, root in ARM_ROOTS.items():
        link_ops[side] = {}
        for link_name in LINK_NAMES:
            path = f"{root}/{link_name}"
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                raise RuntimeError(f"Missing xformable link: {path}")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            link_ops[side][link_name] = xf.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble, "fk_playback"
            )

    # Cache the current fixed Link_6 -> pad midpoint relation for expected-pose checks.
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link6_to_pad = {}
    for side, root in ARM_ROOTS.items():
        link6 = get_world_pose(stage, cache, f"{root}/Link_6")
        pad_mid = pad_midpoint_pose(stage, cache, side)
        if link6 is None or pad_mid is None:
            raise RuntimeError(f"Could not read initial Link_6/pad midpoint for {side}")
        link6_to_pad[side] = relative_pose(link6, pad_mid)

    start = time.time()
    frame = 0
    max_pad_target_error = {"left": 0.0, "right": 0.0}
    max_expected_actual_error = {"left": 0.0, "right": 0.0}
    last_row = None
    log_file = open(log_path, "w", encoding="utf-8")
    print(
        "[KIN-DEMO] Playing verified FK trajectory: zero -> common target -> straight. "
        f"target={vec(target_world)} loop={should_loop}",
        flush=True,
    )
    try:
        while keep_running:
            elapsed = time.time() - start
            if args.duration_sec > 0 and elapsed >= args.duration_sec:
                break
            if args.no_loop and elapsed > loop_duration + 0.2:
                break

            sample = interpolate_q(samples, elapsed, loop_duration, should_loop)
            expected = {}
            for side, root in ARM_ROOTS.items():
                q = np.array(sample["arms"][side]["joint_position_rad"], dtype=float)
                q_by_name = dict(zip(ARM_JOINTS, q.tolist()))
                for link_name in LINK_NAMES:
                    T = fk(chains[link_name], q_by_name)
                    link_ops[side][link_name].Set(gf_matrix_from_column_transform(T))
                link6_world = get_world_pose(stage, UsdGeom.XformCache(Usd.TimeCode.Default()), f"{root}/Link_0") @ fk(chains["Link_6"], q_by_name)
                expected[side] = link6_world @ link6_to_pad[side]

            app.update()

            if frame % max(1, args.log_every) == 0:
                cache = UsdGeom.XformCache(Usd.TimeCode.Default())
                row = {
                    "unix": round(time.time(), 3),
                    "elapsed_sec": round(elapsed, 3),
                    "trajectory_time_sec": round(float(sample["time_sec"]), 3),
                    "frame": frame,
                    "phase": sample["phase"],
                    "target_world_xyz": vec(target_world),
                    "control_mode": "gui_fk_playback_no_physics",
                    "arms": {},
                }
                for side, root in ARM_ROOTS.items():
                    q = np.array(sample["arms"][side]["joint_position_rad"], dtype=float)
                    pad = pad_midpoint_pose(stage, cache, side)
                    link6 = get_world_pose(stage, cache, f"{root}/Link_6")
                    if pad is None or link6 is None:
                        raise RuntimeError(f"Could not read actual pose for {side}")
                    expected_actual_error = float(np.linalg.norm(pad[:3, 3] - expected[side][:3, 3]))
                    pad_target_error = float(np.linalg.norm(pad[:3, 3] - target_world))
                    if sample["phase"] == "target_hold":
                        max_pad_target_error[side] = max(max_pad_target_error[side], pad_target_error)
                    max_expected_actual_error[side] = max(max_expected_actual_error[side], expected_actual_error)
                    row["arms"][side] = {
                        "joint_names": ARM_JOINTS,
                        "joint_position_rad": np.round(q, 6).tolist(),
                        "joint_position_deg": np.round(np.degrees(q), 3).tolist(),
                        "link6_world_xyz": vec(link6[:3, 3]),
                        "pad_midpoint_world_xyz": vec(pad[:3, 3]),
                        "expected_pad_midpoint_world_xyz": vec(expected[side][:3, 3]),
                        "pad_target_error_m": round(pad_target_error, 9),
                        "expected_actual_error_m": round(expected_actual_error, 9),
                    }
                log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_file.flush()
                last_row = row
                atomic_write_json(
                    heartbeat_path,
                    {
                        "status": "running",
                        "scene": args.scene,
                        "trajectory": str(trajectory_path),
                        "frame": frame,
                        "elapsed_sec": round(elapsed, 3),
                        "phase": sample["phase"],
                        "target_world_xyz": vec(target_world),
                        "left_pad_midpoint_world_xyz": row["arms"]["left"]["pad_midpoint_world_xyz"],
                        "right_pad_midpoint_world_xyz": row["arms"]["right"]["pad_midpoint_world_xyz"],
                        "max_expected_actual_error_m": max_expected_actual_error,
                        "updated_unix": time.time(),
                    },
                )
            frame += 1
    finally:
        log_file.close()
        summary = {
            "status": "stopped",
            "scene": args.scene,
            "trajectory": str(trajectory_path),
            "frames": frame,
            "target_world_xyz": vec(target_world),
            "max_target_hold_pad_target_error_m": max_pad_target_error,
            "max_expected_actual_error_m": max_expected_actual_error,
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
