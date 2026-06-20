#!/usr/bin/env python3
"""GUI demo loop for left/right JAKA arms and EG2-4C2 grippers."""


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
import argparse
import json
import os
import signal
import time
from pathlib import Path


SCENE = "/home/andyee/isaacsim/playground/2026060721.usd"
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
    parser.add_argument("--frames-per-pose", type=int, default=90)
    parser.add_argument("--settle-frames", type=int, default=60)
    return parser.parse_args()


def atomic_write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = out_dir / "gui_loop_heartbeat.json"

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        launch_config={"renderer": "RaytracedLighting", "headless": False}
    )

    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.kit.app
    import omni.usd
    from omni.isaac.core import SimulationContext

    keep_running = True

    def handle_signal(_signum, _frame):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    print(f"[LOOP] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(120):
        app.update()
        stage = ctx.get_stage()
        if stage and stage.GetPrimAtPath(LEFT_ARM) and stage.GetPrimAtPath(RIGHT_ARM):
            break
        time.sleep(0.05)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No stage after opening scene")

    def prim(path):
        p = stage.GetPrimAtPath(path)
        return p if p and p.IsValid() else None

    def gripper_root(arm_path):
        return f"{arm_path}/{GRIPPER_SUFFIX}"

    def arm_joint_path(arm_path, joint_index):
        return f"{arm_path}/joints/joint_{joint_index}"

    def angular_drive(joint_prim, stiffness=2500.0, damping=150.0, max_force=5000.0):
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        if not drive.GetTypeAttr().Get():
            drive.CreateTypeAttr("position")
        for getter, creator, value in [
            (drive.GetStiffnessAttr, drive.CreateStiffnessAttr, stiffness),
            (drive.GetDampingAttr, drive.CreateDampingAttr, damping),
            (drive.GetMaxForceAttr, drive.CreateMaxForceAttr, max_force),
        ]:
            attr = getter()
            if attr:
                attr.Set(value)
            else:
                creator(value)
        return drive

    def set_arm_pose(arm_path, targets_deg):
        for i, target in enumerate(targets_deg, start=1):
            j = prim(arm_joint_path(arm_path, i))
            if not j or not j.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(f"Missing revolute joint: {arm_joint_path(arm_path, i)}")
            angular_drive(j).GetTargetPositionAttr().Set(float(target))

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
        for link_name, sign in GRIPPER_VISUAL_LINKS.items():
            op = get_or_add_rotate_y(f"{root}/{link_name}")
            if op:
                op.Set(float(aperture_deg) * sign)

    def disable_broken_gripper_physics(arm_path):
        root = prim(gripper_root(arm_path))
        if not root:
            return
        for p in Usd.PrimRange(root):
            if "Joint" in p.GetTypeName():
                p.SetActive(False)
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                p.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if p.HasAPI(UsdPhysics.MassAPI):
                p.RemoveAPI(UsdPhysics.MassAPI)
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    for arm in [LEFT_ARM, RIGHT_ARM]:
        root_joint = prim(f"{arm}/root_joint")
        if root_joint:
            attr = root_joint.GetAttribute("physxArticulation:articulationEnabled")
            if attr:
                attr.Set(True)
        disable_broken_gripper_physics(arm)

    sim_ctx = SimulationContext()
    sim_ctx.play()

    poses = [
        {
            "left": [12.0, -18.0, 16.0, 10.0, -14.0, 8.0],
            "right": [-12.0, -18.0, 16.0, -10.0, -14.0, -8.0],
            "gripper": 38.0,
        },
        {
            "left": [-10.0, -5.0, 22.0, -14.0, -5.0, -12.0],
            "right": [10.0, -5.0, 22.0, 14.0, -5.0, 12.0],
            "gripper": 4.0,
        },
        {
            "left": [4.0, -25.0, 8.0, 18.0, -18.0, 20.0],
            "right": [-4.0, -25.0, 8.0, -18.0, -18.0, -20.0],
            "gripper": 32.0,
        },
        {
            "left": [0.0, -8.0, 8.0, 0.0, -8.0, 0.0],
            "right": [0.0, -8.0, 8.0, 0.0, -8.0, 0.0],
            "gripper": 8.0,
        },
    ]

    print("[LOOP] Running continuous GUI motion. Stop with Ctrl+C or kill the PID.", flush=True)
    cycle = 0
    try:
        while keep_running:
            for pose_index, pose in enumerate(poses):
                if not keep_running:
                    break
                set_arm_pose(LEFT_ARM, pose["left"])
                set_arm_pose(RIGHT_ARM, pose["right"])
                set_gripper_angle(LEFT_ARM, pose["gripper"])
                set_gripper_angle(RIGHT_ARM, pose["gripper"])
                atomic_write_json(
                    heartbeat_path,
                    {
                        "status": "running",
                        "cycle": cycle,
                        "pose_index": pose_index,
                        "left_targets_deg": pose["left"],
                        "right_targets_deg": pose["right"],
                        "gripper_aperture_deg": pose["gripper"],
                        "updated_unix": time.time(),
                    },
                )
                for _ in range(args.frames_per_pose):
                    if not keep_running:
                        break
                    sim_ctx.step(render=True)
            cycle += 1
    finally:
        atomic_write_json(
            heartbeat_path,
            {"status": "stopped", "cycle": cycle, "updated_unix": time.time()},
        )
        sim_ctx.stop()
        for _ in range(args.settle_frames):
            app.update()
        simulation_app.close()


if __name__ == "__main__":
    main()
