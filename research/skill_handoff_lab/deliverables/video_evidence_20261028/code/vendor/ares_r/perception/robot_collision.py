"""Canonical whole-robot collision geometry in the ARES-R BODY frame.

One geometry snapshot feeds overlay, point-cloud self-filtering, inactive-arm
obstacles, and offline mutual-arm collision checks.  No control API is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def transform_xyz_rpy(xyz, rpy) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    result = np.eye(4)
    result[:3, :3] = rz @ ry @ rx
    result[:3, 3] = np.asarray(xyz, dtype=float)
    return result


def _origin(joint) -> np.ndarray:
    node = joint.find("origin")
    xyz = [0.0] * 3 if node is None else [float(v) for v in node.get("xyz", "0 0 0").split()]
    rpy = [0.0] * 3 if node is None else [float(v) for v in node.get("rpy", "0 0 0").split()]
    return transform_xyz_rpy(xyz, rpy)


def _axis_rotation(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis; c, s = math.cos(angle), math.sin(angle); q = 1.0 - c
    result = np.eye(4)
    result[:3, :3] = [[c + x*x*q, x*y*q-z*s, x*z*q+y*s],
                      [y*x*q+z*s, c+y*y*q, y*z*q-x*s],
                      [z*x*q-y*s, z*y*q+x*s, c+z*z*q]]
    return result


def arm_link_transforms(urdf_path: Path, joints_rad) -> dict:
    """Return model-base transforms for base_link and link1..link6."""
    joints = np.asarray(joints_rad, dtype=float)
    if joints.shape != (6,) or not np.isfinite(joints).all():
        raise ValueError("six finite arm joints required")
    root = ET.parse(str(urdf_path)).getroot()
    result = {"base_link": np.eye(4)}
    current = np.eye(4)
    for index, angle in enumerate(joints, 1):
        node = root.find("joint[@name='joint%d']" % index)
        if node is None:
            raise ValueError("joint%d missing from pinned URDF" % index)
        axis_node = node.find("axis")
        axis = [float(v) for v in axis_node.get("xyz", "0 0 1").split()]
        current = current @ _origin(node) @ _axis_rotation(axis, float(angle))
        result["link%d" % index] = current.copy()
    return result


@dataclass(frozen=True)
class CollisionBox:
    geometry_id: str
    owner: str
    kind: str
    center_body_m: tuple
    rotation_body: tuple
    half_extents_m: tuple
    filter_owned: bool = True
    source: str = ""

    def as_dict(self) -> dict:
        return {"geometry_id": self.geometry_id, "owner": self.owner, "kind": self.kind,
                "center_body_m": list(self.center_body_m),
                "rotation_body": [list(row) for row in self.rotation_body],
                "half_extents_m": list(self.half_extents_m),
                "filter_owned": self.filter_owned, "source": self.source}

    @classmethod
    def from_dict(cls, value):
        return cls(value["geometry_id"], value["owner"], value["kind"],
                   tuple(value["center_body_m"]), tuple(tuple(row) for row in value["rotation_body"]),
                   tuple(value["half_extents_m"]), bool(value.get("filter_owned", True)),
                   value.get("source", ""))


@dataclass(frozen=True)
class RobotGeometrySnapshot:
    boxes: tuple
    joints_rad: dict
    geometry_revision: str
    joint_snapshot_revision: str
    tool_revision: str
    scene_revision: str
    timings_s: dict
    provenance: dict

    def as_dict(self) -> dict:
        return {"schema_version": 1, "frame": "BODY", "unit": "m",
                "boxes": [box.as_dict() for box in self.boxes],
                "joints_rad": self.joints_rad, "geometry_revision": self.geometry_revision,
                "joint_snapshot_revision": self.joint_snapshot_revision,
                "tool_revision": self.tool_revision, "scene_revision": self.scene_revision,
                "timings_s": self.timings_s, "provenance": self.provenance,
                "execution_allowed": False}

    @classmethod
    def from_dict(cls, value):
        return cls(tuple(CollisionBox.from_dict(box) for box in value["boxes"]),
                   value["joints_rad"], value["geometry_revision"],
                   value["joint_snapshot_revision"], value["tool_revision"],
                   value["scene_revision"], value.get("timings_s", {}),
                   value.get("provenance", {}))


def _world_arm_transform(world: dict, side: str) -> np.ndarray:
    arm = world["arms"][side]
    return transform_xyz_rpy(arm["base_xyz_m"], arm["base_rpy_rad"])


def _box_from_local(identifier, owner, kind, local, transform, source,
                    filter_owned=True) -> CollisionBox:
    local_center = np.asarray(local["center_m"], dtype=float)
    center = transform[:3, :3] @ local_center + transform[:3, 3]
    return CollisionBox(identifier, owner, kind, tuple(center.tolist()),
                        tuple(tuple(row) for row in transform[:3, :3].tolist()),
                        tuple(float(v) for v in local["half_extents_m"]),
                        filter_owned, source)


def build_geometry_snapshot(model: dict, world: dict, audits: dict) -> RobotGeometrySnapshot:
    """Compile both live read-only arm states through the pinned ARES model."""
    started = time.perf_counter(); boxes = []; joints = {}; tools = {}
    urdf_path = Path(model["asset_root"]) / model["urdf"]
    fk_started = time.perf_counter()
    for side in ("left", "right"):
        audit = audits[side]
        diagnostics = audit["diagnostics"]
        q = [float(value) for value in diagnostics["joint_position_rad"]]
        joints[side] = q
        tool = diagnostics["tool_data"]
        tools[side] = {"tool_id": tool["tool_id"], "pose_mm_rad": tool["pose_mm_rad"]}
        correction = np.asarray(audit["T_controller_model"], dtype=float)
        if correction.shape != (4, 4):
            raise ValueError("%s T_controller_model must be 4x4" % side)
        body_model = _world_arm_transform(world, side) @ correction
        links = arm_link_transforms(urdf_path, q)
        for link_name, local in model["arm_link_boxes"].items():
            transform = body_model @ links[link_name]
            boxes.append(_box_from_local("%s/%s" % (side, link_name), "%s_arm" % side,
                                         "arm_link", local, transform,
                                         "pinned ARES %s mesh AABB" % link_name))
        link6 = body_model @ links["link6"]
        boxes.append(_box_from_local("%s/gripper" % side, "%s_gripper_tool" % side,
                                     "gripper_max_envelope",
                                     model["gripper_max_envelope_link6"], link6,
                                     "pinned ARES EG2-4C2 mesh/URDF max-opening envelope"))
        pose = list(tool["pose_mm_rad"])
        tcp_local = np.asarray(pose[:3], dtype=float) * 0.001
        tcp_body = link6[:3, :3] @ tcp_local + link6[:3, 3]
        boxes.append(CollisionBox("%s/tcp" % side, "%s_gripper_tool" % side,
                                  "tcp_site_marker", tuple(tcp_body.tolist()),
                                  tuple(tuple(row) for row in link6[:3, :3].tolist()),
                                  (0.012, 0.012, 0.012), True,
                                  "live JAKA tool %s; marker enclosed by gripper envelope" % tool["tool_id"]))
    fk_s = time.perf_counter() - fk_started
    for item in model["fixed_body_boxes"]:
        boxes.append(CollisionBox(item["geometry_id"], item["owner"], item["kind"],
                                  tuple(item["center_body_m"]),
                                  ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                                  tuple(np.asarray(item["dims_m"], dtype=float) / 2.0),
                                  bool(item["filter_owned"]), item["source"]))
    geometry_revision = model["geometry_revision"]
    joint_revision = _digest(joints); tool_revision = _digest(tools)
    scene_revision = _digest({"geometry": geometry_revision, "joints": joint_revision,
                              "tools": tool_revision})
    return RobotGeometrySnapshot(tuple(boxes), joints, geometry_revision, joint_revision,
                                 tool_revision, scene_revision,
                                 {"fk_and_body_export": fk_s,
                                  "total": time.perf_counter() - started},
                                 {"asset_revision": model["asset_revision"],
                                  "audits": {side: audits[side].get("report_path", "")
                                             for side in ("left", "right")}})


def save_geometry_snapshot(snapshot: RobotGeometrySnapshot, path: Path) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.as_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_geometry_snapshot(path: Path) -> RobotGeometrySnapshot:
    return RobotGeometrySnapshot.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _inside_box(points: np.ndarray, box: CollisionBox, margin_m: float) -> np.ndarray:
    center = np.asarray(box.center_body_m); rotation = np.asarray(box.rotation_body)
    local = (points - center) @ rotation
    return np.all(np.abs(local) <= np.asarray(box.half_extents_m) + margin_m, axis=1)


def self_filter_body_cloud(points_body_m: np.ndarray, snapshot: RobotGeometrySnapshot,
                           margin_m: float) -> tuple:
    """Remove points inside the exact same boxes exported for collision use."""
    started = time.perf_counter()
    points = np.asarray(points_body_m)
    if points.ndim != 2 or points.shape[1] != 3 or not 0 <= margin_m <= 0.1:
        raise ValueError("BODY Nx3 points and margin 0..0.1 m required")
    remaining = np.ones(len(points), dtype=bool); assignment = np.full(len(points), -1, dtype=np.int16)
    owned = [box for box in snapshot.boxes if box.filter_owned]
    owners = []
    for box in owned:
        if box.owner not in owners: owners.append(box.owner)
    for owner_index, owner in enumerate(owners):
        hit = np.zeros(len(points), dtype=bool)
        for box in owned:
            if box.owner == owner:
                hit |= _inside_box(points, box, margin_m)
        selected = remaining & hit
        assignment[selected] = owner_index
        remaining[selected] = False
    per_owner = {owner: int(np.sum(assignment == index)) for index, owner in enumerate(owners)}
    stats = {"margin_m": margin_m, "raw_points": int(len(points)),
             "filtered_points": int(np.sum(remaining)),
             "removed_points": int(np.sum(~remaining)), "per_owner_removed": per_owner,
             "elapsed_s": time.perf_counter() - started,
             "scene_revision": snapshot.scene_revision}
    return remaining, stats


def inactive_arm_obstacles(snapshot: RobotGeometrySnapshot, active_side: str) -> dict:
    if active_side not in ("left", "right"):
        raise ValueError("active_side must be left or right")
    started = time.perf_counter(); inactive = "right" if active_side == "left" else "left"
    selected = [box for box in snapshot.boxes
                if box.owner in ("%s_arm" % inactive, "%s_gripper_tool" % inactive)
                or box.kind == "central_exclusion"]
    revision = _digest({"active": active_side, "scene": snapshot.scene_revision,
                        "inactive_joints": snapshot.joints_rad[inactive],
                        "tool": snapshot.tool_revision,
                        "boxes": [box.as_dict() for box in selected]})
    return {"schema_version": 1, "frame": "BODY", "active_arm": active_side,
            "inactive_arm": inactive, "boxes": [box.as_dict() for box in selected],
            "revision": revision, "source_scene_revision": snapshot.scene_revision,
            "conversion_elapsed_s": time.perf_counter() - started}


def obb_overlap(first: CollisionBox, second: CollisionBox, epsilon=1e-9) -> bool:
    """15-axis separating-axis test for two BODY oriented boxes."""
    a = np.asarray(first.rotation_body); b = np.asarray(second.rotation_body)
    ea = np.asarray(first.half_extents_m); eb = np.asarray(second.half_extents_m)
    relative = a.T @ b; absolute = np.abs(relative) + epsilon
    translation = a.T @ (np.asarray(second.center_body_m) - np.asarray(first.center_body_m))
    for i in range(3):
        if abs(translation[i]) > ea[i] + np.dot(eb, absolute[i, :]): return False
    for j in range(3):
        if abs(np.dot(translation, relative[:, j])) > eb[j] + np.dot(ea, absolute[:, j]): return False
    for i in range(3):
        for j in range(3):
            ra = ea[(i+1)%3]*absolute[(i+2)%3, j] + ea[(i+2)%3]*absolute[(i+1)%3, j]
            rb = eb[(j+1)%3]*absolute[i, (j+2)%3] + eb[(j+2)%3]*absolute[i, (j+1)%3]
            value = abs(translation[(i+2)%3]*relative[(i+1)%3, j] -
                        translation[(i+1)%3]*relative[(i+2)%3, j])
            if value > ra + rb: return False
    return True


def mutual_arm_collisions(snapshot: RobotGeometrySnapshot, active_side: str) -> list:
    inactive = "right" if active_side == "left" else "left"
    active = [box for box in snapshot.boxes
              if box.owner in ("%s_arm" % active_side, "%s_gripper_tool" % active_side)]
    other = [box for box in snapshot.boxes
             if box.owner in ("%s_arm" % inactive, "%s_gripper_tool" % inactive)]
    return [{"active": a.geometry_id, "inactive": b.geometry_id,
             "gripper_involved": "gripper_tool" in a.owner or "gripper_tool" in b.owner}
            for a in active for b in other if obb_overlap(a, b)]
