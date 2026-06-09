#!/usr/bin/env python3
"""GUI-only visual motion loop for the dual-arm scene.

This intentionally does not play physics. The current USD has unstable arm
physics/articulation state, so this script gives a conservative visual sanity
check: both arm assemblies sway gently and both grippers open/close.
"""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--period-sec", type=float, default=12.0)
    parser.add_argument("--log-every", type=int, default=15)
    return parser.parse_args()


def atomic_write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def vec3_to_list(value):
    if value is None:
        return None
    return [round(float(value[0]), 6), round(float(value[1]), 6), round(float(value[2]), 6)]


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "visual_safe_motion_log.jsonl"
    heartbeat_path = out_dir / "visual_safe_heartbeat.json"

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={"renderer": "RaytracedLighting", "headless": args.headless}
    )

    from pxr import UsdGeom
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
    print(f"[VISUAL] Opening scene: {args.scene}", flush=True)
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

    def xformable(path):
        p = prim(path)
        if not p or not p.IsA(UsdGeom.Xformable):
            return None
        return UsdGeom.Xformable(p)

    def get_or_add_rotate(path, axis, suffix):
        xf = xformable(path)
        if not xf:
            return None
        op_name = f"xformOp:rotate{axis}:{suffix}"
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == op_name:
                return op
        if axis == "X":
            return xf.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble, suffix)
        if axis == "Y":
            return xf.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble, suffix)
        return xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble, suffix)

    def world_pos(path):
        p = prim(path)
        if not p or not p.IsA(UsdGeom.Xformable):
            return None
        return omni.usd.get_world_transform_matrix(p).ExtractTranslation()

    def gripper_root(arm_path):
        return f"{arm_path}/{GRIPPER_SUFFIX}"

    def set_gripper_angle(arm_path, aperture_deg):
        touched = 0
        root = gripper_root(arm_path)
        for link_name, sign in GRIPPER_VISUAL_LINKS.items():
            op = get_or_add_rotate(f"{root}/{link_name}", "Y", "visual_grip")
            if op:
                op.Set(float(aperture_deg) * sign)
                touched += 1
        return touched

    controls = {}
    for side, arm_path, mirror in [("left", LEFT_ARM, 1.0), ("right", RIGHT_ARM, -1.0)]:
        controls[side] = {
            "mirror": mirror,
            "base": get_or_add_rotate(arm_path, "Z", "visual_base_sway"),
            "link2": get_or_add_rotate(f"{arm_path}/Link_2", "X", "visual_elbow"),
            "link4": get_or_add_rotate(f"{arm_path}/Link_4", "Y", "visual_wrist"),
            "link6": get_or_add_rotate(f"{arm_path}/Link_6", "Z", "visual_tool"),
        }

    start = time.time()
    frame = 0
    log_file = open(log_path, "w", encoding="utf-8")
    print("[VISUAL] Running GUI-only safe visual motion. Stop with Ctrl+C or kill the PID.", flush=True)
    try:
        while keep_running:
            elapsed = time.time() - start
            if args.duration_sec > 0 and elapsed >= args.duration_sec:
                break
            phase = 2.0 * math.pi * (elapsed / max(args.period_sec, 0.1))
            aperture = 16.0 + 10.0 * (0.5 + 0.5 * math.sin(phase + 0.4))
            command = {}
            for side, arm_path in [("left", LEFT_ARM), ("right", RIGHT_ARM)]:
                mirror = controls[side]["mirror"]
                angles = {
                    "base_z_deg": mirror * 2.0 * math.sin(phase),
                    "link2_x_deg": 1.2 * math.sin(phase + 0.7),
                    "link4_y_deg": mirror * 1.4 * math.sin(phase + 1.4),
                    "link6_z_deg": mirror * 2.0 * math.sin(phase + 2.1),
                }
                if controls[side]["base"]:
                    controls[side]["base"].Set(angles["base_z_deg"])
                if controls[side]["link2"]:
                    controls[side]["link2"].Set(angles["link2_x_deg"])
                if controls[side]["link4"]:
                    controls[side]["link4"].Set(angles["link4_y_deg"])
                if controls[side]["link6"]:
                    controls[side]["link6"].Set(angles["link6_z_deg"])
                touched = set_gripper_angle(arm_path, aperture)
                command[side] = {
                    **{k: round(v, 4) for k, v in angles.items()},
                    "gripper_aperture_deg": round(aperture, 4),
                    "gripper_visual_links": touched,
                    "link6_world": vec3_to_list(world_pos(f"{arm_path}/Link_6")),
                    "flange_world": vec3_to_list(world_pos(f"{arm_path}/{GRIPPER_SUFFIX}")),
                }

            app.update()

            if frame % max(args.log_every, 1) == 0:
                row = {
                    "unix": round(time.time(), 3),
                    "elapsed_sec": round(elapsed, 3),
                    "frame": frame,
                    "phase_rad": round(phase, 4),
                    "control_mode": "gui_visual_no_physics",
                    "left": command["left"],
                    "right": command["right"],
                }
                log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                log_file.flush()
                atomic_write_json(
                    heartbeat_path,
                    {
                        "status": "running",
                        "scene": args.scene,
                        "log_path": str(log_path),
                        "frame": frame,
                        "elapsed_sec": round(elapsed, 3),
                        "control_mode": "gui_visual_no_physics",
                        "left_link6_world": command["left"]["link6_world"],
                        "right_link6_world": command["right"]["link6_world"],
                        "updated_unix": time.time(),
                    },
                )
            frame += 1
    finally:
        log_file.close()
        atomic_write_json(
            heartbeat_path,
            {
                "status": "stopped",
                "scene": args.scene,
                "log_path": str(log_path),
                "frame": frame,
                "updated_unix": time.time(),
            },
        )
        for _ in range(12):
            app.update()
        simulation_app.close()


if __name__ == "__main__":
    main()
