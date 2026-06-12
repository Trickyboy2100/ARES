#!/usr/bin/env python3
"""Minimal demo: drive the left arm through a sequence of joint poses via PhysX DriveAPI.

Uses the same joint-drive mechanism as control_left_right_with_grippers.py —
UsdPhysics.DriveAPI angular targetPosition (degrees) + SimulationContext.step.
No FK playback, no proxy cubes, no IK.

Run:
    /home/andyee/isaacsim/python.sh scripts/gui_left_arm_joint_drive_demo.py
Stop:
    pkill -f gui_left_arm_joint_drive_demo.py
"""


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
import os
import signal
import time
from pathlib import Path

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
LEFT_ARM = "/World/robot/jaka_minicobo_left"
GRIPPER_SUFFIX = "Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2"
GRIPPER_VISUAL_LINKS = {
    "left_outer_link": 1.0,
    "left_inner_link": 0.65,
    "left_pad": 1.0,
    "right_outer_link": -1.0,
    "right_inner_link": -0.65,
    "right_pad": -1.0,
}

# Sequence of (joint_deg[6], gripper_aperture_deg) poses for the left arm.
# Each pose is held for --frames-per-pose physics steps before moving on.
POSES = [
    ([0.0,   0.0,  0.0,  0.0,  0.0,  0.0],  35.0),   # rest / open
    ([15.0, -20.0, 18.0,  8.0, -12.0,  10.0],  35.0),   # reach up-left
    ([15.0, -20.0, 18.0,  8.0, -12.0,  10.0],   4.0),   # close gripper
    ([0.0,  -10.0, 25.0,  0.0, -10.0,   0.0],   4.0),   # lift arm
    ([0.0,   -5.0, 10.0,  0.0,  -5.0,   0.0],  35.0),   # retract / open
    ([0.0,   0.0,  0.0,  0.0,  0.0,  0.0],  35.0),   # back to rest
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--frames-per-pose", type=int, default=120,
                        help="Physics steps to dwell at each pose.")
    parser.add_argument("--loops", type=int, default=0,
                        help="Repeat count (0 = loop forever).")
    return parser.parse_args()


def main():
    args = parse_args()

    from isaacsim import SimulationApp
    sim_app = SimulationApp(launch_config={"renderer": "RaytracedLighting", "headless": False})

    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.kit.app
    import omni.usd
    from omni.isaac.core import SimulationContext

    keep_running = True

    def _stop(*_):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    print(f"[DRIVE] Opening: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(120):
        app.update()
        stage = ctx.get_stage()
        if stage and stage.GetPrimAtPath(LEFT_ARM).IsValid():
            break
        time.sleep(0.05)
    stage = ctx.get_stage()
    if stage is None or not stage.GetPrimAtPath(LEFT_ARM).IsValid():
        raise RuntimeError(f"Left arm prim not found: {LEFT_ARM}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _prim(path):
        p = stage.GetPrimAtPath(path)
        return p if p.IsValid() else None

    def _ensure_drive(joint_path):
        j = _prim(joint_path)
        if j is None:
            raise RuntimeError(f"Missing joint: {joint_path}")
        drive = UsdPhysics.DriveAPI.Get(j, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(j, "angular")
        if not drive.GetTypeAttr().Get():
            drive.CreateTypeAttr("position")
        for getter, creator, val in [
            (drive.GetStiffnessAttr, drive.CreateStiffnessAttr, 2500.0),
            (drive.GetDampingAttr,   drive.CreateDampingAttr,   150.0),
            (drive.GetMaxForceAttr,  drive.CreateMaxForceAttr,  5000.0),
        ]:
            attr = getter()
            (attr if attr else creator()).Set(val)
        return drive

    def set_arm_joints(targets_deg):
        for i, deg in enumerate(targets_deg, start=1):
            _ensure_drive(f"{LEFT_ARM}/joints/joint_{i}").GetTargetPositionAttr().Set(float(deg))

    def _gripper_rotate_y_op(link_path):
        p = _prim(link_path)
        if p is None or not p.IsA(UsdGeom.Xformable):
            return None
        xf = UsdGeom.Xformable(p)
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:rotateY:ctrl":
                return op
        return xf.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble, "ctrl")

    def set_gripper(aperture_deg):
        root = f"{LEFT_ARM}/{GRIPPER_SUFFIX}"
        for link, sign in GRIPPER_VISUAL_LINKS.items():
            op = _gripper_rotate_y_op(f"{root}/{link}")
            if op:
                op.Set(float(aperture_deg) * sign)

    def _disable_broken_gripper_physics():
        root = _prim(f"{LEFT_ARM}/{GRIPPER_SUFFIX}")
        if root is None:
            return
        for p in Usd.PrimRange(root):
            if "Joint" in p.GetTypeName():
                p.SetActive(False)
            for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI):
                if p.HasAPI(api):
                    p.RemoveAPI(api)

    # ── Setup ──────────────────────────────────────────────────────────────────

    root_joint = _prim(f"{LEFT_ARM}/root_joint")
    if root_joint:
        attr = root_joint.GetAttribute("physxArticulation:articulationEnabled")
        if attr:
            attr.Set(True)
    _disable_broken_gripper_physics()

    sim_ctx = SimulationContext()
    sim_ctx.play()

    # ── Motion loop ────────────────────────────────────────────────────────────

    loop = 0
    print("[DRIVE] Running. Stop with Ctrl+C.", flush=True)
    try:
        while keep_running:
            for idx, (joints_deg, gripper_deg) in enumerate(POSES):
                if not keep_running:
                    break
                set_arm_joints(joints_deg)
                set_gripper(gripper_deg)
                print(
                    f"[DRIVE] loop={loop} pose={idx}  "
                    f"joints={[round(d,1) for d in joints_deg]}  "
                    f"gripper={gripper_deg}°",
                    flush=True,
                )
                for _ in range(args.frames_per_pose):
                    if not keep_running:
                        break
                    sim_ctx.step(render=True)
            loop += 1
            if args.loops > 0 and loop >= args.loops:
                break
    finally:
        sim_ctx.stop()
        for _ in range(30):
            app.update()
        sim_app.close()


if __name__ == "__main__":
    main()
