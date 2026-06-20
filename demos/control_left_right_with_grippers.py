#!/usr/bin/env python3
"""Control and validate the left/right JAKA arms plus EG2-4C2 grippers."""


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
import math
import os
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
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--frames", type=int, default=120, help="Frames to settle after each command.")
    parser.add_argument(
        "--keep-gripper-physics",
        action="store_true",
        help="Keep the broken EG2-4C2 PhysX articulation active instead of using visual-only gripper control.",
    )
    return parser.parse_args()


def atomic_write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def vec3_to_list(v):
    if v is None:
        return None
    return [round(float(v[0]), 6), round(float(v[1]), 6), round(float(v[2]), 6)]


def distance(a, b):
    if a is None or b is None:
        return 0.0
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def main():
    args = parse_args()

    running_in_kit = False
    try:
        import omni.usd  # noqa: F401

        running_in_kit = omni.usd.get_context().get_stage() is not None
    except Exception:
        running_in_kit = False

    simulation_app = None
    if not running_in_kit:
        from isaacsim import SimulationApp

        simulation_app = SimulationApp(
            launch_config={"renderer": "RaytracedLighting", "headless": args.headless}
        )

    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from omni.isaac.core import SimulationContext

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "left_right_control_report.json"

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()
    print(f"[CTRL] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(80):
        app.update()
        stage = ctx.get_stage()
        if stage and stage.GetPrimAtPath(LEFT_ARM) and stage.GetPrimAtPath(RIGHT_ARM):
            break
        time.sleep(0.05)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No USD stage after open_stage")

    def prim(path):
        p = stage.GetPrimAtPath(path)
        return p if p and p.IsValid() else None

    def world_pos(path):
        p = prim(path)
        if not p or not p.IsA(UsdGeom.Xformable):
            return None
        mat = omni.usd.get_world_transform_matrix(p)
        return mat.ExtractTranslation()

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

    def arm_joint_path(arm_path, i):
        return f"{arm_path}/joints/joint_{i}"

    def gripper_root(arm_path):
        return f"{arm_path}/{GRIPPER_SUFFIX}"

    def gripper_joint_path(arm_path):
        return f"{gripper_root(arm_path)}/joints/gripper_joint"

    def disable_broken_gripper_physics(arm_path):
        root = prim(gripper_root(arm_path))
        if not root:
            return {"root": gripper_root(arm_path), "found": False}
        disabled = {"root": str(root.GetPath()), "joints_deactivated": [], "rigid_apis_removed": []}
        for p in Usd.PrimRange(root):
            if "Joint" in p.GetTypeName():
                p.SetActive(False)
                disabled["joints_deactivated"].append(str(p.GetPath()))
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                p.RemoveAPI(UsdPhysics.RigidBodyAPI)
                disabled["rigid_apis_removed"].append(str(p.GetPath()))
            if p.HasAPI(UsdPhysics.MassAPI):
                p.RemoveAPI(UsdPhysics.MassAPI)
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        return disabled

    def set_revolute_target_deg(joint_path, target_deg, stiffness=2500.0, damping=150.0):
        j = prim(joint_path)
        if not j or not j.IsA(UsdPhysics.RevoluteJoint):
            raise RuntimeError(f"Missing revolute joint: {joint_path}")
        drive = angular_drive(j, stiffness=stiffness, damping=damping)
        drive.GetTargetPositionAttr().Set(float(target_deg))
        return {
            "path": joint_path,
            "target_deg": round(float(target_deg), 4),
            "lower_deg": j.GetAttribute("physics:lowerLimit").Get(),
            "upper_deg": j.GetAttribute("physics:upperLimit").Get(),
        }

    def get_joint_state(joint_path):
        j = prim(joint_path)
        if not j:
            return None
        state = j.GetAttribute("state:angular:physics:position")
        target = j.GetAttribute("drive:angular:physics:targetPosition")
        return {
            "state_deg": state.Get() if state else None,
            "target_deg": target.Get() if target else None,
        }

    gripper_physics_patch = None
    if not args.keep_gripper_physics:
        gripper_physics_patch = {
            "left": disable_broken_gripper_physics(LEFT_ARM),
            "right": disable_broken_gripper_physics(RIGHT_ARM),
        }

    sim_ctx = SimulationContext()
    sim_ctx.play()

    def step(frames):
        for _ in range(frames):
            sim_ctx.step(render=False)

    def get_or_add_rotate_y(path):
        p = prim(path)
        if not p or not p.IsA(UsdGeom.Xformable):
            raise RuntimeError(f"Missing xformable gripper link: {path}")
        xf = UsdGeom.Xformable(p)
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:rotateY:ctrl":
                return op
        return xf.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble, "ctrl")

    def set_gripper_visual_angle(arm_path, aperture_deg):
        root = gripper_root(arm_path)
        touched = []
        for link_name, sign in GRIPPER_VISUAL_LINKS.items():
            path = f"{root}/{link_name}"
            if not prim(path):
                continue
            op = get_or_add_rotate_y(path)
            op.Set(float(aperture_deg) * sign)
            touched.append({"path": path, "rotateY_ctrl_deg": round(float(aperture_deg) * sign, 4)})
        for _ in range(3):
            app.update()
        return touched

    def probe_point(path):
        p = prim(path)
        if not p or not p.IsA(UsdGeom.Xformable):
            return None
        mat = omni.usd.get_world_transform_matrix(p)
        v = mat.Transform(Gf.Vec3d(0.0, 0.025, 0.025))
        return v

    def verify_arm(arm_label, arm_path, targets):
        link6_path = f"{arm_path}/Link_6"
        before = world_pos(link6_path)
        commands = []
        for i, target in enumerate(targets, start=1):
            commands.append(set_revolute_target_deg(arm_joint_path(arm_path, i), target))
        step(args.frames)
        after = world_pos(link6_path)
        return {
            "arm": arm_label,
            "arm_path": arm_path,
            "commands": commands,
            "link6_before": vec3_to_list(before),
            "link6_after": vec3_to_list(after),
            "link6_delta_m": round(distance(before, after), 6),
            "joint_states": {
                f"joint_{i}": get_joint_state(arm_joint_path(arm_path, i)) for i in range(1, 7)
            },
        }

    def verify_gripper(arm_label, arm_path, open_deg, close_deg):
        root = gripper_root(arm_path)
        joint = gripper_joint_path(arm_path)
        probe_links = [
            f"{root}/left_outer_link",
            f"{root}/right_outer_link",
            f"{root}/left_pad",
            f"{root}/right_pad",
        ]
        before = {p: probe_point(p) for p in probe_links}
        open_cmd = None
        if args.keep_gripper_physics:
            open_cmd = set_revolute_target_deg(joint, open_deg, stiffness=600.0, damping=60.0)
        visual_open = set_gripper_visual_angle(arm_path, open_deg)
        step(args.frames)
        opened = {p: probe_point(p) for p in probe_links}
        close_cmd = None
        if args.keep_gripper_physics:
            close_cmd = set_revolute_target_deg(joint, close_deg, stiffness=600.0, damping=60.0)
        visual_close = set_gripper_visual_angle(arm_path, close_deg)
        step(args.frames)
        closed = {p: probe_point(p) for p in probe_links}
        return {
            "arm": arm_label,
            "gripper_root": root,
            "commands": {
                "physics_open": open_cmd,
                "physics_close": close_cmd,
                "visual_open": visual_open,
                "visual_close": visual_close,
            },
            "joint_state": get_joint_state(joint) if args.keep_gripper_physics else None,
            "control_mode": "physx_drive_plus_visual" if args.keep_gripper_physics else "visual_xform",
            "probe_delta_open_m": {
                p: round(distance(before[p], opened[p]), 6) for p in probe_links
            },
            "probe_delta_close_m": {
                p: round(distance(opened[p], closed[p]), 6) for p in probe_links
            },
            "probe_positions": {
                "before": {p: vec3_to_list(v) for p, v in before.items()},
                "opened": {p: vec3_to_list(v) for p, v in opened.items()},
                "closed": {p: vec3_to_list(v) for p, v in closed.items()},
            },
        }

    report = {
        "scene": args.scene,
        "left_arm_exists": prim(LEFT_ARM) is not None,
        "right_arm_exists": prim(RIGHT_ARM) is not None,
        "gripper_physics_patch": gripper_physics_patch,
        "notes": [
            "EG2-4C2 gripper PhysX articulation is invalid in this scene because rigid bodies are nested under arm Link_6; default control uses visual Xform gripper motion."
        ]
        if not args.keep_gripper_physics
        else [],
    }
    if not report["left_arm_exists"] or not report["right_arm_exists"]:
        report["final_verdict"] = "ARM_PRIM_MISSING"
        atomic_write_json(report_path, report)
        raise RuntimeError("Left or right arm prim is missing")

    for arm in [LEFT_ARM, RIGHT_ARM]:
        root_joint = prim(f"{arm}/root_joint")
        if root_joint:
            attr = root_joint.GetAttribute("physxArticulation:articulationEnabled")
            if attr:
                attr.Set(True)

    print("[CTRL] Moving both arms to asymmetric test poses...", flush=True)
    left_pose = [12.0, -18.0, 16.0, 10.0, -14.0, 8.0]
    right_pose = [-12.0, -18.0, 16.0, -10.0, -14.0, -8.0]
    report["arms"] = {
        "left": verify_arm("left", LEFT_ARM, left_pose),
        "right": verify_arm("right", RIGHT_ARM, right_pose),
    }

    print("[CTRL] Opening and closing both grippers...", flush=True)
    report["grippers"] = {
        "left": verify_gripper("left", LEFT_ARM, open_deg=38.0, close_deg=4.0),
        "right": verify_gripper("right", RIGHT_ARM, open_deg=38.0, close_deg=4.0),
    }

    print("[CTRL] Returning arms to neutral-ish pose...", flush=True)
    neutral = [0.0, -8.0, 8.0, 0.0, -8.0, 0.0]
    report["return_to_neutral"] = {
        "left": verify_arm("left", LEFT_ARM, neutral),
        "right": verify_arm("right", RIGHT_ARM, neutral),
    }

    arm_ok = all(v["link6_delta_m"] > 0.002 for v in report["arms"].values())
    grip_open_ok = all(
        max(v["probe_delta_open_m"].values() or [0.0]) > 0.0005
        for v in report["grippers"].values()
    )
    grip_close_ok = all(
        max(v["probe_delta_close_m"].values() or [0.0]) > 0.0005
        for v in report["grippers"].values()
    )
    report["checks"] = {
        "arm_motion_ok": arm_ok,
        "gripper_open_motion_ok": grip_open_ok,
        "gripper_close_motion_ok": grip_close_ok,
    }
    report["final_verdict"] = (
        "LEFT_RIGHT_ARMS_AND_GRIPPERS_MOVED"
        if arm_ok and grip_open_ok and grip_close_ok
        else "MOTION_INCOMPLETE"
    )

    sim_ctx.stop()
    atomic_write_json(report_path, report)
    print(f"[CTRL] Report: {report_path}", flush=True)
    print(f"[CTRL] Verdict: {report['final_verdict']}", flush=True)

    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
