#!/usr/bin/env python3
"""Probe dual-arm frames and cross-check USD poses against URDF FK."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from pxr import Usd, UsdGeom
except ModuleNotFoundError:
    # Isaac's python.sh exposes pxr after SimulationApp starts. The pure
    # kinematics helpers in this module are still usable before that point.
    Usd = None
    UsdGeom = None


DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_control_fixed.usd"
DEFAULT_ARM_URDF = (
    "/home/andyee/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/"
    "jaka_minicobo.urdf"
)
DEFAULT_CUROBO_URDF = (
    "/home/andyee/Developer/PG-JY/jinyu_ros_pkg/nodes/simulation/"
    "jaka_minicobo_gripper.urdf"
)
PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
CUROBO_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

GRIPPER_ROOT_SUFFIX = (
    "Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2"
)
GRIPPER_MARKERS = {
    "gripper_mount": "Link_6/CAM_Mount",
    "force_sensor": "Link_6/CAM_Mount/force_sensor",
    "gripper_flange": "Link_6/CAM_Mount/force_sensor/gripper_flange",
    "gripper_root": GRIPPER_ROOT_SUFFIX,
    "left_pad": f"{GRIPPER_ROOT_SUFFIX}/left_pad",
    "right_pad": f"{GRIPPER_ROOT_SUFFIX}/right_pad",
}


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: Optional[float]
    upper: Optional[float]


def parse_vec(text: Optional[str], default: Iterable[float]) -> np.ndarray:
    if not text:
        return np.array(list(default), dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    # URDF rpy is fixed-axis roll/pitch/yaw: R = Rz(yaw) Ry(pitch) Rx(roll).
    T[:3, :3] = rot_z(float(rpy[2])) @ rot_y(float(rpy[1])) @ rot_x(float(rpy[0]))
    T[:3, 3] = xyz
    return T


def transform_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = axis_angle(axis, angle)
    return T


def matrix_to_quat_wxyz(R: np.ndarray) -> List[float]:
    trace = float(np.trace(R))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q_np = np.array(q, dtype=float)
    norm = float(np.linalg.norm(q_np))
    return (q_np / norm).tolist() if norm > 1e-12 else [1.0, 0.0, 0.0, 0.0]


def pose_dict(T: np.ndarray) -> Dict[str, object]:
    return {
        "position_xyz": np.round(T[:3, 3], 9).tolist(),
        "quaternion_wxyz": np.round(matrix_to_quat_wxyz(T[:3, :3]), 9).tolist(),
        "matrix4x4": np.round(T, 9).tolist(),
    }


def load_joints(urdf_path: Path) -> Dict[str, Joint]:
    root = ET.parse(urdf_path).getroot()
    joints: Dict[str, Joint] = {}
    for elem in root.findall("joint"):
        name = elem.attrib["name"]
        origin = elem.find("origin")
        parent = elem.find("parent")
        child = elem.find("child")
        axis = elem.find("axis")
        limit = elem.find("limit")
        if parent is None or child is None:
            continue
        joints[name] = Joint(
            name=name,
            joint_type=elem.attrib.get("type", "fixed"),
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            xyz=parse_vec(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0]),
            rpy=parse_vec(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0]),
            axis=parse_vec(axis.attrib.get("xyz") if axis is not None else None, [0, 0, 1]),
            lower=float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            upper=float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
        )
    return joints


def chain_to_link(joints: Dict[str, Joint], base_link: str, tip_link: str) -> List[Joint]:
    by_parent: Dict[str, List[Joint]] = {}
    for joint in joints.values():
        by_parent.setdefault(joint.parent, []).append(joint)

    def visit(link: str, chain: List[Joint], seen: set[str]) -> Optional[List[Joint]]:
        if link == tip_link:
            return chain
        for joint in by_parent.get(link, []):
            if joint.child in seen:
                continue
            found = visit(joint.child, chain + [joint], seen | {joint.child})
            if found is not None:
                return found
        return None

    chain = visit(base_link, [], {base_link})
    if chain is None:
        raise ValueError(f"No URDF chain from {base_link!r} to {tip_link!r}")
    return chain


def fk(chain: List[Joint], q_by_name: Dict[str, float]) -> np.ndarray:
    T = np.eye(4)
    for joint in chain:
        T = T @ transform_from_xyz_rpy(joint.xyz, joint.rpy)
        if joint.joint_type in {"revolute", "continuous"}:
            T = T @ transform_from_axis_angle(joint.axis, q_by_name.get(joint.name, 0.0))
        elif joint.joint_type == "prismatic":
            delta = np.eye(4)
            delta[:3, 3] = joint.axis * q_by_name.get(joint.name, 0.0)
            T = T @ delta
    return T


def usd_matrix_to_column_major_np(matrix) -> np.ndarray:
    # USD/Gf stores row-vector transforms with translation in row 3. Transpose it
    # into the column-vector convention used by the FK math in this script.
    return np.array([[float(matrix[i][j]) for j in range(4)] for i in range(4)]).T


def get_world_pose(stage: Usd.Stage, cache: UsdGeom.XformCache, path: str) -> Optional[np.ndarray]:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    return usd_matrix_to_column_major_np(cache.GetLocalToWorldTransform(prim))


def relative_pose(T_parent_world: np.ndarray, T_child_world: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_parent_world) @ T_child_world


def diff_pose(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    pos_err = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rot_delta = a[:3, :3].T @ b[:3, :3]
    cos_angle = max(-1.0, min(1.0, (float(np.trace(rot_delta)) - 1.0) / 2.0))
    return {
        "position_error_m": pos_err,
        "rotation_error_rad": float(math.acos(cos_angle)),
        "rotation_error_deg": float(math.degrees(math.acos(cos_angle))),
    }


def parse_joint_values(names: List[str], values: str) -> Dict[str, float]:
    parts = [float(v) for v in values.split(",") if v.strip()]
    if len(parts) != len(names):
        raise ValueError(f"Expected {len(names)} comma-separated values, got {len(parts)}")
    return dict(zip(names, parts))


def build_side_report(
    stage: Usd.Stage,
    cache: UsdGeom.XformCache,
    side: str,
    arm_link6_fk: np.ndarray,
    arm_dummy_fk: np.ndarray,
    curobo_tool_fk: np.ndarray,
) -> Dict[str, object]:
    root = f"/World/robot/jaka_minicobo_{side}"
    paths = {
        "arm_root": root,
        "base_link": f"{root}/Link_0",
        "bare_link6": f"{root}/Link_6",
        "bare_dummy_tcp": f"{root}/Link_6/dummy_tcp",
    }
    paths.update({name: f"{root}/{suffix}" for name, suffix in GRIPPER_MARKERS.items()})

    world_poses: Dict[str, object] = {}
    raw: Dict[str, Optional[np.ndarray]] = {}
    for name, path in paths.items():
        pose = get_world_pose(stage, cache, path)
        raw[name] = pose
        world_poses[name] = {"path": path, "exists": pose is not None}
        if pose is not None:
            world_poses[name].update(pose_dict(pose))

    base_world = raw["base_link"]
    cross_checks: Dict[str, object] = {}
    if base_world is not None:
        predicted_link6 = base_world @ arm_link6_fk
        predicted_dummy = base_world @ arm_dummy_fk
        predicted_curobo_tool = base_world @ curobo_tool_fk

        cross_checks["urdf_link6_from_base"] = pose_dict(predicted_link6)
        cross_checks["urdf_dummy_tcp_from_base"] = pose_dict(predicted_dummy)
        cross_checks["curobo_4C2_Link5_from_base"] = pose_dict(predicted_curobo_tool)

        if raw["bare_link6"] is not None:
            cross_checks["usd_link6_vs_arm_urdf_link6"] = diff_pose(predicted_link6, raw["bare_link6"])
        if raw["bare_dummy_tcp"] is not None:
            cross_checks["usd_dummy_tcp_vs_arm_urdf_dummy_tcp"] = diff_pose(predicted_dummy, raw["bare_dummy_tcp"])
        if raw["gripper_root"] is not None:
            cross_checks["usd_gripper_root_relative_to_link6"] = pose_dict(
                relative_pose(raw["bare_link6"], raw["gripper_root"])
            )
        if raw["left_pad"] is not None and raw["right_pad"] is not None:
            midpoint = np.eye(4)
            midpoint[:3, :3] = raw["left_pad"][:3, :3]
            midpoint[:3, 3] = (raw["left_pad"][:3, 3] + raw["right_pad"][:3, 3]) * 0.5
            cross_checks["usd_pad_midpoint_world"] = pose_dict(midpoint)
            cross_checks["usd_pad_midpoint_relative_to_link6"] = pose_dict(
                relative_pose(raw["bare_link6"], midpoint)
            )

    return {
        "paths": paths,
        "world_poses": world_poses,
        "cross_checks": cross_checks,
    }


def main() -> int:
    global Usd, UsdGeom
    if Usd is None or UsdGeom is None:
        from pxr import Usd as _Usd, UsdGeom as _UsdGeom

        Usd = _Usd
        UsdGeom = _UsdGeom

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--arm-urdf", default=DEFAULT_ARM_URDF)
    parser.add_argument("--curobo-urdf", default=DEFAULT_CUROBO_URDF)
    parser.add_argument("--arm-q", default="0,0,0,0,0,0", help="joint_1..joint_6 radians")
    parser.add_argument("--curobo-gripper-q", type=float, default=0.0)
    parser.add_argument(
        "--out",
        default=str(PLAYGROUND_ROOT / "reports/kinematics_probe.json"),
    )
    args = parser.parse_args()

    scene = Path(args.scene)
    arm_urdf = Path(args.arm_urdf)
    curobo_urdf = Path(args.curobo_urdf)
    out = Path(args.out)

    arm_q = parse_joint_values(ARM_JOINTS, args.arm_q)
    curobo_q = dict(zip(CUROBO_JOINTS, [arm_q[name] for name in ARM_JOINTS]))
    curobo_q["4C2_Joint1"] = float(args.curobo_gripper_q)

    arm_joints = load_joints(arm_urdf)
    curobo_joints = load_joints(curobo_urdf)

    arm_link6_chain = chain_to_link(arm_joints, "Link_0", "Link_6")
    arm_dummy_chain = chain_to_link(arm_joints, "Link_0", "dummy_tcp")
    curobo_tool_chain = chain_to_link(curobo_joints, "base_link", "4C2_Link5")

    arm_link6_fk = fk(arm_link6_chain, arm_q)
    arm_dummy_fk = fk(arm_dummy_chain, arm_q)
    curobo_tool_fk = fk(curobo_tool_chain, curobo_q)

    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise RuntimeError(f"Could not open USD scene: {scene}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    report = {
        "inputs": {
            "scene": str(scene),
            "arm_urdf": str(arm_urdf),
            "curobo_urdf": str(curobo_urdf),
            "arm_q_rad": arm_q,
            "curobo_gripper_q_rad": float(args.curobo_gripper_q),
        },
        "frame_convention": {
            "base_link": "USD /World/robot/jaka_minicobo_<side>/Link_0; URDF Link_0/base_link.",
            "bare_end_effector": "JAKA wrist flange: USD Link_6/dummy_tcp; URDF dummy_tcp, currently coincident with Link_6.",
            "mounted_gripper": f"USD {GRIPPER_ROOT_SUFFIX}; cuRobo tool frame is URDF 4C2_Link5.",
            "math": "Matrices in this report use column-vector convention: p_world = T_world_frame @ p_frame.",
        },
        "urdf_fk": {
            "arm_Link0_to_Link6": pose_dict(arm_link6_fk),
            "arm_Link0_to_dummy_tcp": pose_dict(arm_dummy_fk),
            "curobo_base_link_to_4C2_Link5": pose_dict(curobo_tool_fk),
            "arm_chain": [joint.name for joint in arm_link6_chain],
            "curobo_tool_chain": [joint.name for joint in curobo_tool_chain],
        },
        "sides": {},
    }

    for side in ("left", "right"):
        report["sides"][side] = build_side_report(
            stage, cache, side, arm_link6_fk, arm_dummy_fk, curobo_tool_fk
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for side, side_report in report["sides"].items():
        base = side_report["world_poses"]["base_link"].get("position_xyz")
        link6 = side_report["world_poses"]["bare_link6"].get("position_xyz")
        grip = side_report["world_poses"]["gripper_root"].get("position_xyz")
        check = side_report["cross_checks"].get("usd_link6_vs_arm_urdf_link6", {})
        print(f"{side}: base={base} link6={link6} gripper_root={grip}")
        if check:
            print(
                "  usd Link_6 vs URDF FK: "
                f"pos_err={check['position_error_m']:.9f} m, "
                f"rot_err={check['rotation_error_deg']:.6f} deg"
            )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
