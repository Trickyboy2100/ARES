#!/usr/bin/env python3
"""L0: compare two independent FK implementations for pinned JAKA URDF poses.

No Isaac launch, no cuRobo solve, no JAKA/AMR/gripper hardware access.
Model parity is NOT a certification of controller TCP or real contact geometry.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def run(ares: Path, ares_r: Path, samples=16, seed=20261013):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("L0 requires numpy in the selected local Python environment") from exc
    core = ares / "isaac_sim/simforge/core"
    real_src = ares_r / "src"
    sys.path.insert(0, str(core))
    sys.path.insert(0, str(real_src))
    try:
        from kinematics import load_joints, chain_to_link, fk, transform_from_xyz_rpy
        from ares_r.motion.runtime_goal_ik import RuntimeGoalIK
    except ImportError as exc:
        raise RuntimeError("Cannot import ARES and ARES-R FK modules; verify --ares and --ares-r") from exc
    from research.skill_handoff_lab.tools.align_repos import joint_limits, soft_limit_overlap, read_json
    urdf = ares / "isaac_sim/simforge/robot/jaka_minicobo_gripper.urdf"
    real_config = ares_r / "config"
    world = read_json(real_config/"robot_world.json")
    site = read_json(real_config/"jaka_mini2_motion.site.json")
    poses = read_json(real_config/"named_poses.json")["poses"]
    overlap, regions = soft_limit_overlap(joint_limits(urdf), site)
    if not overlap:
        raise RuntimeError("site soft joint limits and ARES URDF have empty overlap")
    low = np.array([regions["joint%d"%i]["sample_lower_rad"] for i in range(1,7)],dtype=float)
    high = np.array([regions["joint%d"%i]["sample_upper_rad"] for i in range(1,7)],dtype=float)
    rng = np.random.default_rng(int(seed))
    qs = [("zero", np.zeros(6))]
    for i in range(samples):
        qs.append(("sample_%02d"%i, rng.uniform(low, high)))
    joints = load_joints(urdf)
    chain = chain_to_link(joints, "base_link", "link6")
    reports = []
    for side in ("left", "right"):
        mount = world["arms"][side]
        T_body_base = transform_from_xyz_rpy(
            np.asarray(mount["base_xyz_m"],dtype=float),
            np.asarray(mount["base_rpy_rad"],dtype=float))
        solver = RuntimeGoalIK(str(urdf), T_body_base, np.eye(4))
        side_qs = list(qs)
        for pose_name, pose in poses.items():
            joint_data = pose.get("arms",{}).get(side,{})
            q = joint_data.get("ik_joint_rad",joint_data.get("joint_rad"))
            if isinstance(q,list) and len(q)==6:
                q = np.asarray(q,dtype=float)
                if np.all(q>=low) and np.all(q<=high):
                    side_qs.append(("site_pose_"+pose_name,q))
        for name,q in side_qs:
            q = np.asarray(q,dtype=float)
            Ta = T_body_base @ fk(chain, {"joint%d"%i:float(q[i-1]) for i in range(1,7)})
            Tb = solver.fk(q)
            translation = float(np.linalg.norm(Ta[:3,3]-Tb[:3,3]))
            rel = Ta[:3,:3].T @ Tb[:3,:3]
            angle = math.degrees(math.acos(float(np.clip((np.trace(rel)-1.0)*.5,-1.0,1.0))))
            reports.append({"arm":side,"pose":name,"position_error_m":translation,
                           "orientation_error_deg":angle,
                           "passed":translation<=1e-5 and angle<=.001})
    status = "FK_IMPLEMENTATION_PARITY_PASS" if reports and all(x["passed"] for x in reports) else "FK_IMPLEMENTATION_PARITY_FAIL"
    return {
        "schema_version":1,
        "status":status,
        "scope":"URDF_FLANGE_FK_ONLY_NOT_REAL_TOOL_TCP",
        "actual_motion":False,
        "samples":len(reports),
        "maximum_position_error_m":max(x["position_error_m"] for x in reports),
        "maximum_orientation_error_deg":max(x["orientation_error_deg"] for x in reports),
        "controller_tcp_consistency":"NOT_TESTED",
        "isaac_usd_frame_consistency":"NOT_TESTED",
        "results":reports,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ares", required=True, type=Path)
    p.add_argument("--ares-r", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--samples", type=int, default=16)
    args = p.parse_args(argv)
    value = run(args.ares.resolve(),args.ares_r.resolve(),args.samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    print(value["status"],"samples",value["samples"],"max_m",value["maximum_position_error_m"])
    return 0 if value["status"]=="FK_IMPLEMENTATION_PARITY_PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
