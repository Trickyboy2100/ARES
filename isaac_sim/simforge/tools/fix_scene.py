#!/usr/bin/env python3
"""Create a control-friendly USD from the current dual-arm scene.

The original scene keeps EG2-4C2 gripper rigid bodies below the arm Link_6
rigid body. PhysX rejects that nested rigid-body articulation, so this script
turns the grippers into visual/end-effector children and leaves the JAKA arms
as the only active articulations.
"""

import argparse
import json
import os
from pathlib import Path


SRC = "/home/andyee/isaacsim/playground/2026060721.usd"
OUT = "/home/andyee/isaacsim/playground/2026060721_control_fixed.usd"
REPORT = "/home/andyee/isaacsim/playground/aruco_eval/left_right_control/scene_fix_report.json"
LEFT_ARM = "/World/robot/jaka_minicobo_left"
RIGHT_ARM = "/World/robot/jaka_minicobo_right"
STANDALONE = "/jaka_minicobo"
GRIPPER_SUFFIX = "Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2"
NEUTRAL = [0.0, -8.0, 8.0, 0.0, -8.0, 0.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=SRC)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--report", default=REPORT)
    return parser.parse_args()


def set_or_create(attr_getter, attr_creator, value):
    attr = attr_getter()
    if attr:
        attr.Set(value)
    else:
        attr_creator(value)


def main():
    args = parse_args()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(args.src)
    if not stage:
        raise RuntimeError(f"Could not open {args.src}")

    report = {
        "src": args.src,
        "out": args.out,
        "standalone_deactivated": False,
        "default_prim": None,
        "arms": {},
    }

    world = stage.GetPrimAtPath("/World")
    if world:
        stage.SetDefaultPrim(world)
        report["default_prim"] = str(world.GetPath())

    standalone = stage.GetPrimAtPath(STANDALONE)
    if standalone:
        standalone.SetActive(False)
        report["standalone_deactivated"] = True

    for arm_path in [LEFT_ARM, RIGHT_ARM]:
        arm_report = {
            "exists": False,
            "root_joint_body1": None,
            "drives_tuned": [],
            "joint_states_reset": [],
            "gripper_root": None,
            "gripper_joints_deactivated": [],
            "gripper_rigid_apis_removed": [],
            "gripper_mass_apis_removed": [],
            "gripper_articulation_apis_removed": [],
        }
        report["arms"][arm_path] = arm_report

        arm = stage.GetPrimAtPath(arm_path)
        if not arm:
            continue
        arm_report["exists"] = True

        root_joint = stage.GetPrimAtPath(f"{arm_path}/root_joint")
        if root_joint:
            body1 = root_joint.GetRelationship("physics:body1")
            targets = [str(t) for t in body1.GetTargets()] if body1 else []
            arm_report["root_joint_body1"] = targets
            attr = root_joint.GetAttribute("physxArticulation:articulationEnabled")
            if attr:
                attr.Set(True)

        for i, neutral_target in enumerate(NEUTRAL, start=1):
            joint_path = f"{arm_path}/joints/joint_{i}"
            joint = stage.GetPrimAtPath(joint_path)
            if not joint or not joint.IsA(UsdPhysics.RevoluteJoint):
                continue
            drive = UsdPhysics.DriveAPI.Get(joint, "angular")
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
            # Keep USD's force drive type, but make it much less floppy and reset targets.
            set_or_create(drive.GetTargetPositionAttr, drive.CreateTargetPositionAttr, float(neutral_target))
            set_or_create(drive.GetStiffnessAttr, drive.CreateStiffnessAttr, 1200.0)
            set_or_create(drive.GetDampingAttr, drive.CreateDampingAttr, 120.0)
            set_or_create(drive.GetMaxForceAttr, drive.CreateMaxForceAttr, 5000.0)
            position_attr = joint.GetAttribute("state:angular:physics:position")
            if not position_attr:
                position_attr = joint.CreateAttribute(
                    "state:angular:physics:position", Sdf.ValueTypeNames.Double
                )
            velocity_attr = joint.GetAttribute("state:angular:physics:velocity")
            if not velocity_attr:
                velocity_attr = joint.CreateAttribute(
                    "state:angular:physics:velocity", Sdf.ValueTypeNames.Double
                )
            position_attr.Set(float(neutral_target))
            velocity_attr.Set(0.0)
            arm_report["drives_tuned"].append(
                {
                    "joint": joint_path,
                    "target_deg": neutral_target,
                    "stiffness": 1200.0,
                    "damping": 120.0,
                    "max_force": 5000.0,
                }
            )
            arm_report["joint_states_reset"].append(
                {"joint": joint_path, "position_deg": neutral_target, "velocity_deg_s": 0.0}
            )

        gripper_root = stage.GetPrimAtPath(f"{arm_path}/{GRIPPER_SUFFIX}")
        if gripper_root:
            arm_report["gripper_root"] = str(gripper_root.GetPath())
            for prim in Usd.PrimRange(gripper_root):
                if "Joint" in prim.GetTypeName():
                    prim.SetActive(False)
                    arm_report["gripper_joints_deactivated"].append(str(prim.GetPath()))
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    arm_report["gripper_rigid_apis_removed"].append(str(prim.GetPath()))
                if prim.HasAPI(UsdPhysics.MassAPI):
                    prim.RemoveAPI(UsdPhysics.MassAPI)
                    arm_report["gripper_mass_apis_removed"].append(str(prim.GetPath()))
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                    arm_report["gripper_articulation_apis_removed"].append(str(prim.GetPath()))

    stage.GetRootLayer().Export(args.out)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[FIX] Wrote fixed scene: {args.out}", flush=True)
    print(f"[FIX] Report: {args.report}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
