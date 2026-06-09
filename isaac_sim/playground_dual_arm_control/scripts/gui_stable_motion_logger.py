#!/usr/bin/env python3
"""Smooth GUI motion loop with JSONL logging for the dual-arm scene."""

import argparse
import json
import math
import os
import signal
import time
from pathlib import Path


SCENE = "/home/andyee/isaacsim/playground/2026060721_control_fixed.usd"
OUT_DIR = Path("/home/andyee/isaacsim/playground/aruco_eval/left_right_control")
LEFT_ARM = "/World/robot/jaka_minicobo_left"
RIGHT_ARM = "/World/robot/jaka_minicobo_right"
GRIPPER_SUFFIX = "Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2"
GRIPPER_VISUAL_LINKS = {
    "left_outer_link": 1.0,
    "left_inner_link": 0.65,
    "left_pad": 1.0,
    "right_outer_link": -1.0,
    "right_inner_link": -0.65,
    "right_pad": -1.0,
}
NEUTRAL_DEG = [0.0, -8.0, 8.0, 0.0, -8.0, 0.0]
AMPLITUDE_DEG = [0.6, 0.35, 0.35, 0.45, 0.3, 0.5]
PHASE_OFFSETS = [0.0, 0.7, 1.4, 2.1, 2.8, 3.5]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until stopped.")
    parser.add_argument("--log-every", type=int, default=10, help="Write one JSONL row every N frames.")
    parser.add_argument("--period-sec", type=float, default=14.0, help="Seconds per smooth motion cycle.")
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--drive-stiffness", type=float, default=120.0)
    parser.add_argument("--drive-damping", type=float, default=60.0)
    parser.add_argument("--drive-max-force", type=float, default=800.0)
    parser.add_argument("--render", action="store_true", help="Render while headless. GUI always renders.")
    return parser.parse_args()


def atomic_write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def vec3_to_list(value):
    if value is None:
        return None
    return [round(float(value[0]), 6), round(float(value[1]), 6), round(float(value[2]), 6)]


def dist(a, b):
    if a is None or b is None:
        return None
    return round(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))), 6)


def has_bad_number(values):
    for value in values:
        if value is None:
            continue
        try:
            if not math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            return True
    return False


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "motion_log.jsonl"
    heartbeat_path = out_dir / "stable_loop_heartbeat.json"

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={"renderer": "RaytracedLighting", "headless": args.headless}
    )

    import numpy as np
    from pxr import UsdGeom, UsdPhysics
    import omni.kit.app
    import omni.usd
    from omni.isaac.core import SimulationContext
    from isaacsim.core.prims import Articulation

    keep_running = True

    def handle_signal(_signum, _frame):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    print(f"[STABLE] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    stage = None
    for _ in range(160):
        app.update()
        stage = ctx.get_stage()
        if stage and stage.GetPrimAtPath(LEFT_ARM) and stage.GetPrimAtPath(RIGHT_ARM):
            break
        time.sleep(0.05)
    if stage is None:
        raise RuntimeError("No stage after opening scene")

    def prim(path):
        p = stage.GetPrimAtPath(path)
        return p if p and p.IsValid() else None

    def arm_joint_path(arm_path, index):
        return f"{arm_path}/joints/joint_{index}"

    def gripper_root(arm_path):
        return f"{arm_path}/{GRIPPER_SUFFIX}"

    def world_pos(path):
        p = prim(path)
        if not p or not p.IsA(UsdGeom.Xformable):
            return None
        return omni.usd.get_world_transform_matrix(p).ExtractTranslation()

    def set_attr(getter, creator, value):
        attr = getter()
        if attr:
            attr.Set(value)
        else:
            creator(value)

    def angular_drive(joint_prim):
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        set_attr(drive.GetTypeAttr, drive.CreateTypeAttr, "position")
        set_attr(drive.GetStiffnessAttr, drive.CreateStiffnessAttr, float(args.drive_stiffness))
        set_attr(drive.GetDampingAttr, drive.CreateDampingAttr, float(args.drive_damping))
        set_attr(drive.GetMaxForceAttr, drive.CreateMaxForceAttr, float(args.drive_max_force))
        return drive

    def get_or_add_rotate_y(path):
        p = prim(path)
        if not p or not p.IsA(UsdGeom.Xformable):
            return None
        xf = UsdGeom.Xformable(p)
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:rotateY:ctrl":
                return op
        return xf.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble, "ctrl")

    def set_gripper_angle(arm_path, aperture_deg):
        root = gripper_root(arm_path)
        touched = 0
        for link_name, sign in GRIPPER_VISUAL_LINKS.items():
            op = get_or_add_rotate_y(f"{root}/{link_name}")
            if op:
                op.Set(float(aperture_deg) * sign)
                touched += 1
        return touched

    def set_targets(arm_path, targets_deg):
        for index, target in enumerate(targets_deg, start=1):
            joint = prim(arm_joint_path(arm_path, index))
            if not joint or not joint.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(f"Missing revolute joint: {arm_joint_path(arm_path, index)}")
            angular_drive(joint).GetTargetPositionAttr().Set(float(target))

    joint_indices = np.arange(6, dtype=np.int32)

    def deg_to_rad_row(targets_deg):
        return np.deg2rad(np.array([targets_deg], dtype=np.float64))

    def apply_articulation_targets(articulation, targets_deg):
        positions = deg_to_rad_row(targets_deg)
        articulation.set_joint_position_targets(positions, joint_indices=joint_indices)

    def zero_articulation_velocities(articulation):
        articulation.set_joint_velocities(np.zeros((1, 6), dtype=np.float64), joint_indices=joint_indices)

    def articulation_positions_deg(articulation):
        positions = articulation.get_joint_positions(joint_indices=joint_indices)
        if positions is None:
            return [None] * 6
        return np.rad2deg(np.asarray(positions)[0]).tolist()

    def joint_sample(arm_path, targets_deg, articulation=None):
        art_states = articulation_positions_deg(articulation) if articulation is not None else None
        joints = []
        bad = False
        for index, target in enumerate(targets_deg, start=1):
            joint = prim(arm_joint_path(arm_path, index))
            state = None
            lower = None
            upper = None
            if joint:
                state = as_float(art_states[index - 1]) if art_states is not None else as_float(
                    joint.GetAttribute("state:angular:physics:position").Get()
                )
                lower = as_float(joint.GetAttribute("physics:lowerLimit").Get())
                upper = as_float(joint.GetAttribute("physics:upperLimit").Get())
            out_of_limit = False
            if state is not None and lower is not None and upper is not None:
                span = upper - lower
                state_for_limit = state
                if span >= 719.0:
                    state_for_limit = ((state + 180.0) % 360.0) - 180.0
                out_of_limit = state_for_limit < lower - 0.5 or state_for_limit > upper + 0.5
            bad = bad or has_bad_number([state, lower, upper, target])
            joints.append(
                {
                    "name": f"joint_{index}",
                    "target_deg": round(float(target), 4),
                    "state_deg": None if state is None else round(state, 4),
                    "error_deg": None if state is None else round(float(target) - state, 4),
                    "lower_deg": lower,
                    "upper_deg": upper,
                    "out_of_limit": out_of_limit,
                }
            )
        return joints, bad

    def targets_for_phase(phase, left_base=None, right_base=None):
        left_base = left_base or NEUTRAL_DEG
        right_base = right_base or NEUTRAL_DEG
        left = []
        right = []
        mirror = [-1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
        for left_neutral, right_neutral, amplitude, offset, sign in zip(
            left_base, right_base, AMPLITUDE_DEG, PHASE_OFFSETS, mirror
        ):
            value = amplitude * math.sin(phase + offset)
            left.append(left_neutral + value)
            right.append(right_neutral + sign * value)
        gripper = 16.0 + 8.0 * (0.5 + 0.5 * math.sin(phase + 0.4))
        return left, right, gripper

    for arm in [LEFT_ARM, RIGHT_ARM]:
        root_joint = prim(f"{arm}/root_joint")
        if root_joint:
            attr = root_joint.GetAttribute("physxArticulation:articulationEnabled")
            if attr:
                attr.Set(True)

    sim_ctx = SimulationContext()
    sim_ctx.initialize_physics()
    left_art = Articulation(LEFT_ARM, name="left_arm_view", reset_xform_properties=False)
    right_art = Articulation(RIGHT_ARM, name="right_arm_view", reset_xform_properties=False)
    left_art.initialize()
    right_art.initialize()
    left_base = articulation_positions_deg(left_art)
    right_base = articulation_positions_deg(right_art)
    zero_articulation_velocities(left_art)
    zero_articulation_velocities(right_art)
    left0, right0, gripper0 = targets_for_phase(0.0, left_base, right_base)
    set_targets(LEFT_ARM, left0)
    set_targets(RIGHT_ARM, right0)
    set_gripper_angle(LEFT_ARM, gripper0)
    set_gripper_angle(RIGHT_ARM, gripper0)
    apply_articulation_targets(left_art, left0)
    apply_articulation_targets(right_art, right0)
    sim_ctx.play()
    for _ in range(max(args.warmup_frames, 0)):
        apply_articulation_targets(left_art, left0)
        apply_articulation_targets(right_art, right0)
        sim_ctx.step(render=(not args.headless) or args.render)

    previous = {"left_link6": None, "right_link6": None}
    start = time.time()
    frame = 0
    status = "running"
    log_file = open(log_path, "w", encoding="utf-8")
    print(f"[STABLE] Running smooth motion. JSONL log: {log_path}", flush=True)
    try:
        while keep_running:
            elapsed = time.time() - start
            if args.duration_sec > 0 and elapsed >= args.duration_sec:
                break
            phase = 2.0 * math.pi * (elapsed / max(args.period_sec, 0.1))
            left_targets, right_targets, gripper_aperture = targets_for_phase(phase, left_base, right_base)
            set_targets(LEFT_ARM, left_targets)
            set_targets(RIGHT_ARM, right_targets)
            apply_articulation_targets(left_art, left_targets)
            apply_articulation_targets(right_art, right_targets)
            left_gripper_links = set_gripper_angle(LEFT_ARM, gripper_aperture)
            right_gripper_links = set_gripper_angle(RIGHT_ARM, gripper_aperture)
            sim_ctx.step(render=(not args.headless) or args.render)

            if frame % max(args.log_every, 1) == 0:
                left_link6 = vec3_to_list(world_pos(f"{LEFT_ARM}/Link_6"))
                right_link6 = vec3_to_list(world_pos(f"{RIGHT_ARM}/Link_6"))
                left_flange = vec3_to_list(world_pos(f"{LEFT_ARM}/{GRIPPER_SUFFIX}"))
                right_flange = vec3_to_list(world_pos(f"{RIGHT_ARM}/{GRIPPER_SUFFIX}"))
                left_joints, left_bad = joint_sample(LEFT_ARM, left_targets, left_art)
                right_joints, right_bad = joint_sample(RIGHT_ARM, right_targets, right_art)
                row = {
                    "unix": round(time.time(), 3),
                    "elapsed_sec": round(elapsed, 3),
                    "frame": frame,
                    "phase_rad": round(phase, 4),
                    "left": {
                        "joints": left_joints,
                        "link6_world": left_link6,
                        "flange_world": left_flange,
                        "link6_delta_m": dist(previous["left_link6"], left_link6),
                        "bad_number": left_bad or has_bad_number(left_link6 or []),
                    },
                    "right": {
                        "joints": right_joints,
                        "link6_world": right_link6,
                        "flange_world": right_flange,
                        "link6_delta_m": dist(previous["right_link6"], right_link6),
                        "bad_number": right_bad or has_bad_number(right_link6 or []),
                    },
                    "gripper": {
                        "aperture_deg": round(gripper_aperture, 4),
                        "left_visual_links": left_gripper_links,
                        "right_visual_links": right_gripper_links,
                    },
                }
                log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_file.flush()
                previous["left_link6"] = left_link6
                previous["right_link6"] = right_link6
                atomic_write_json(
                    heartbeat_path,
                    {
                        "status": status,
                        "scene": args.scene,
                        "log_path": str(log_path),
                        "control_mode": "pd_current_pose_small_amplitude",
                        "left_base_deg": [round(float(v), 4) if v is not None else None for v in left_base],
                        "right_base_deg": [round(float(v), 4) if v is not None else None for v in right_base],
                        "frame": frame,
                        "elapsed_sec": round(elapsed, 3),
                        "left_link6_world": left_link6,
                        "right_link6_world": right_link6,
                        "gripper_aperture_deg": round(gripper_aperture, 4),
                        "updated_unix": time.time(),
                    },
                )
            frame += 1
    except Exception as exc:
        status = "error"
        atomic_write_json(
            heartbeat_path,
            {
                "status": status,
                "error": repr(exc),
                "frame": frame,
                "updated_unix": time.time(),
            },
        )
        raise
    finally:
        log_file.close()
        atomic_write_json(
            heartbeat_path,
            {
                "status": "stopped" if status == "running" else status,
                "scene": args.scene,
                "log_path": str(log_path),
                "frame": frame,
                "updated_unix": time.time(),
            },
        )
        sim_ctx.stop()
        for _ in range(12):
            app.update()
        simulation_app.close()


if __name__ == "__main__":
    main()
