#!/usr/bin/env python3
"""Reusable dual-arm planning API for sim-to-real trajectory generation.

This module is intentionally independent from the tray handoff demo timeline.
External task modules can provide per-arm start/end conditions and obstacle
objects, then receive timestamped joint trajectories suitable for a JAKA SDK
adapter or the Isaac Sim FK playback tools.
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
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
from pxr import Usd, UsdGeom

from ik_sanity import joint_limits
from kinematics_probe import (
    ARM_JOINTS,
    CUROBO_JOINTS,
    DEFAULT_ARM_URDF,
    DEFAULT_CUROBO_URDF,
    GRIPPER_ROOT_SUFFIX,
    chain_to_link,
    fk,
    get_world_pose,
    load_joints,
    matrix_to_quat_wxyz,
    relative_pose,
)
from make_tray_handoff_curobo_demo import (
    CUROBO_CFG,
    DEFAULT_SCENE,
    PAD_LOCAL_FORWARD_WORLD,
    PAD_LOCAL_UP_WORLD,
    axis_angle_error_deg,
    fallback_path,
    make_goal,
    normalized,
    pad_world_transform,
    pose_from_xyz,
    solve_pad_ik,
    solve_pad_pose_ik,
    smoothstep,
)


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
Side = Literal["left", "right"]
GoalType = Literal["joint", "pad_position", "pad_pose"]
BODY_MIDLINE_WORLD_X = 0.0
MIDLINE_WALL_X_THICKNESS_M = 2.0
MIDLINE_WALL_Y_SIZE_M = 6.0
MIDLINE_WALL_Z_SIZE_M = 4.0
MIDLINE_WALL_CENTER_Y_M = 0.5
MIDLINE_WALL_CENTER_Z_M = 1.0


@dataclass
class CuboidObstacleSpec:
    """World-frame cuboid obstacle input for cuRobo.

    Either provide `center_world_xyz` + `dims_world_xyz`, or provide a `usd_path`
    and let the API compute a world-aligned bounding box from the loaded scene.
    `inflation_xyz` is added to the cuboid dimensions.
    """

    name: str
    center_world_xyz: Optional[List[float]] = None
    dims_world_xyz: Optional[List[float]] = None
    usd_path: Optional[str] = None
    inflation_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class ArmPlanRequest:
    """Single-arm request.

    `start_joint_position_rad` is always required. End condition is selected by
    `goal_type`:

    - `joint`: set `goal_joint_position_rad`.
    - `pad_position`: set `goal_pad_world_xyz`; pad local Y is constrained up.
    - `pad_pose`: set `goal_pad_world_xyz`, `goal_pad_up_world`,
      and `goal_pad_forward_world`.
    """

    side: Side
    start_joint_position_rad: List[float]
    goal_type: GoalType
    goal_joint_position_rad: Optional[List[float]] = None
    goal_pad_world_xyz: Optional[List[float]] = None
    goal_pad_up_world: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    goal_pad_forward_world: List[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    duration_sec: float = 3.0
    sample_count: int = 121
    use_curobo: bool = True
    curobo_max_attempts: int = 8
    max_position_error_m: float = 0.002
    max_axis_error_deg: float = 5.0


@dataclass
class ArmPlanResult:
    side: Side
    success: bool
    source: str
    joint_names: List[str]
    duration_sec: float
    sample_count: int
    start_joint_position_rad: List[float]
    goal_joint_position_rad: List[float]
    trajectory: List[Dict[str, object]]
    diagnostics: Dict[str, object]


@dataclass
class DualArmPlanResult:
    success: bool
    left: Optional[ArmPlanResult]
    right: Optional[ArmPlanResult]
    synchronized_duration_sec: float
    diagnostics: Dict[str, object]


def _as_np(values: Iterable[float], name: str, size: int = 3) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {arr.tolist()}")
    return arr


def _joint_np(values: Iterable[float], name: str) -> np.ndarray:
    return _as_np(values, name, size=6)


def _midline_signed_margin(side: Side, world_x: float) -> float:
    return world_x - BODY_MIDLINE_WORLD_X if side == "left" else BODY_MIDLINE_WORLD_X - world_x


def _timestamped_points(path: np.ndarray, duration_sec: float) -> List[Dict[str, object]]:
    points = []
    n = len(path)
    for idx, q in enumerate(path):
        t = duration_sec * idx / max(1, n - 1)
        points.append(
            {
                "time_from_start_sec": round(float(t), 6),
                "joint_position_rad": np.round(q, 9).tolist(),
                "joint_position_deg": np.round(np.degrees(q), 6).tolist(),
            }
        )
    return points


def _resample_path(path: np.ndarray, count: int) -> np.ndarray:
    if len(path) == count:
        return path
    rows = []
    for idx in range(max(2, count)):
        u = idx / max(1, count - 1)
        pos = u * (len(path) - 1)
        lo = int(np.floor(pos))
        hi = min(len(path) - 1, lo + 1)
        a = pos - lo
        rows.append((1.0 - a) * path[lo] + a * path[hi])
    return np.asarray(rows, dtype=float)


def _bbox_world(stage, bbox_cache, path: str):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = np.array(box.GetMin(), dtype=float)
    mx = np.array(box.GetMax(), dtype=float)
    if not np.all(np.isfinite(mn)) or not np.all(np.isfinite(mx)):
        return None
    dims = mx - mn
    if float(np.max(dims)) <= 1e-6:
        return None
    return (mn + mx) * 0.5, dims, mn, mx


class DualArmPlanningContext:
    """Loaded scene, robot chains, arm mount transforms, and obstacle builders."""

    def __init__(
        self,
        scene_path: str = DEFAULT_SCENE,
        arm_urdf: str = DEFAULT_ARM_URDF,
        curobo_urdf: str = DEFAULT_CUROBO_URDF,
        curobo_cfg: str = str(CUROBO_CFG),
    ) -> None:
        self.scene_path = scene_path
        self.arm_urdf = arm_urdf
        self.curobo_urdf = curobo_urdf
        self.curobo_cfg = curobo_cfg

        self.stage = Usd.Stage.Open(scene_path)
        if self.stage is None:
            raise RuntimeError(f"Could not open scene: {scene_path}")
        self.cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        self.arm_chain = chain_to_link(load_joints(Path(arm_urdf)), "Link_0", "Link_6")
        self.curobo_chain = chain_to_link(load_joints(Path(curobo_urdf)), "base_link", "4C2_Link5")
        self.lower, self.upper = joint_limits(self.arm_chain)

        self.base_world: Dict[Side, np.ndarray] = {}
        self.link6_to_pad: Dict[Side, np.ndarray] = {}
        self.link6_to_gripper_pads: Dict[Side, Dict[str, np.ndarray]] = {}
        for side in ("left", "right"):
            base, rel, pad_rels = self._selected_pad_midpoint(side)
            self.base_world[side] = base
            self.link6_to_pad[side] = rel
            self.link6_to_gripper_pads[side] = pad_rels

    def _selected_pad_midpoint(self, side: Side) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        root = f"/World/robot/jaka_minicobo_{side}"
        base = get_world_pose(self.stage, self.cache, f"{root}/Link_0")
        link6 = get_world_pose(self.stage, self.cache, f"{root}/Link_6")
        left = get_world_pose(self.stage, self.cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/left_pad")
        right = get_world_pose(self.stage, self.cache, f"{root}/{GRIPPER_ROOT_SUFFIX}/right_pad")
        if base is None or link6 is None or left is None or right is None:
            raise RuntimeError(f"Could not read base/link6/pads for {side}")
        mid = np.array(left, copy=True)
        mid[:3, 3] = (left[:3, 3] + right[:3, 3]) * 0.5
        return (
            base,
            relative_pose(link6, mid),
            {
                "left_pad": relative_pose(link6, left),
                "right_pad": relative_pose(link6, right),
            },
        )

    def pad_world_transform(self, side: Side, q_arm: np.ndarray) -> np.ndarray:
        return pad_world_transform(self.arm_chain, self.base_world[side], self.link6_to_pad[side], q_arm)

    def gripper_pad_world_positions(self, side: Side, q_arm: np.ndarray) -> Dict[str, List[float]]:
        link6_world = self.base_world[side] @ fk(self.arm_chain, dict(zip(ARM_JOINTS, q_arm.tolist())))
        positions = {
            name: link6_world @ rel for name, rel in self.link6_to_gripper_pads[side].items()
        }
        midpoint = self.pad_world_transform(side, q_arm)
        positions["pad_midpoint"] = midpoint
        return {name: np.round(T[:3, 3], 9).tolist() for name, T in positions.items()}

    def validate_midline_constraint(self, side: Side, path: np.ndarray) -> Dict[str, object]:
        samples = []
        worst_margin = float("inf")
        violation_count = 0
        for idx, q in enumerate(path):
            positions = self.gripper_pad_world_positions(side, q)
            for point_name, xyz in positions.items():
                x = float(xyz[0])
                signed_margin = _midline_signed_margin(side, x)
                worst_margin = min(worst_margin, signed_margin)
                if signed_margin < -1e-6:
                    violation_count += 1
                    if len(samples) < 10:
                        samples.append(
                            {
                                "sample_index": idx,
                                "point": point_name,
                                "world_xyz": xyz,
                                "signed_margin_m": signed_margin,
                            }
                        )
        return {
            "enabled": True,
            "body_midline_world_x": BODY_MIDLINE_WORLD_X,
            "rule": "left gripper pads stay at world X >= 0; right gripper pads stay at world X <= 0",
            "ok": violation_count == 0,
            "violation_count": violation_count,
            "worst_signed_margin_m": None if not np.isfinite(worst_margin) else worst_margin,
            "sample_violations": samples,
        }

    def solve_goal_joint(self, request: ArmPlanRequest) -> Tuple[np.ndarray, Dict[str, object]]:
        side = request.side
        start = _joint_np(request.start_joint_position_rad, "start_joint_position_rad")
        seeds = [start, np.zeros(6)]
        if request.goal_type == "joint":
            if request.goal_joint_position_rad is None:
                raise ValueError("goal_joint_position_rad is required for goal_type='joint'")
            goal = _joint_np(request.goal_joint_position_rad, "goal_joint_position_rad")
            return goal, {"goal_type": "joint", "position_error_m": 0.0}

        if request.goal_pad_world_xyz is None:
            raise ValueError("goal_pad_world_xyz is required for pad goals")
        target_xyz = _as_np(request.goal_pad_world_xyz, "goal_pad_world_xyz")
        target_margin = _midline_signed_margin(side, float(target_xyz[0]))
        if target_margin < -1e-6:
            raise ValueError(
                f"{side} goal_pad_world_xyz crosses hard body midline: "
                f"x={target_xyz[0]:.6f}, midline_x={BODY_MIDLINE_WORLD_X:.6f}"
            )

        if request.goal_type == "pad_position":
            goal, pos_err, up_err, success, message = solve_pad_ik(
                self.arm_chain,
                self.lower,
                self.upper,
                self.base_world[side],
                self.link6_to_pad[side],
                target_xyz,
                seeds,
                reference_q=start,
                continuity_weight=0.01,
            )
            if pos_err > request.max_position_error_m or up_err > request.max_axis_error_deg:
                raise RuntimeError(
                    f"{side} pad_position IK failed: pos_err={pos_err:.6f} m, up_err={up_err:.3f} deg"
                )
            return goal, {
                "goal_type": "pad_position",
                "optimizer_success": bool(success),
                "message": message,
                "position_error_m": pos_err,
                "pad_axis_up_error_deg": up_err,
            }

        if request.goal_type == "pad_pose":
            goal, pos_err, up_err, forward_err, success, message = solve_pad_pose_ik(
                self.arm_chain,
                self.lower,
                self.upper,
                self.base_world[side],
                self.link6_to_pad[side],
                target_xyz,
                _as_np(request.goal_pad_up_world, "goal_pad_up_world"),
                _as_np(request.goal_pad_forward_world, "goal_pad_forward_world"),
                seeds,
                reference_q=start,
                continuity_weight=0.01,
            )
            if (
                pos_err > request.max_position_error_m
                or up_err > request.max_axis_error_deg
                or forward_err > request.max_axis_error_deg
            ):
                raise RuntimeError(
                    f"{side} pad_pose IK failed: pos_err={pos_err:.6f} m, "
                    f"up_err={up_err:.3f} deg, forward_err={forward_err:.3f} deg"
                )
            return goal, {
                "goal_type": "pad_pose",
                "optimizer_success": bool(success),
                "message": message,
                "position_error_m": pos_err,
                "pad_axis_up_error_deg": up_err,
                "pad_forward_axis_error_deg": forward_err,
            }

        raise ValueError(f"Unsupported goal_type: {request.goal_type}")

    def build_curobo_obstacles(self, specs: List[CuboidObstacleSpec]):
        from curobo._src.geom.types import Cuboid

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        obstacles: Dict[Side, List[object]] = {"left": [], "right": []}
        report: Dict[str, object] = {}
        for spec in specs:
            inflate = _as_np(spec.inflation_xyz, f"{spec.name}.inflation_xyz")
            if spec.usd_path:
                bbox = _bbox_world(self.stage, bbox_cache, spec.usd_path)
                if bbox is None:
                    report[spec.name] = {"status": "missing_or_empty", "usd_path": spec.usd_path}
                    continue
                center_world, dims_world, mn, mx = bbox
                report[spec.name] = {
                    "source": "usd_bbox",
                    "usd_path": spec.usd_path,
                    "center_world_xyz": np.round(center_world, 9).tolist(),
                    "dims_world_xyz": np.round(dims_world, 9).tolist(),
                    "bbox_min_world_xyz": np.round(mn, 9).tolist(),
                    "bbox_max_world_xyz": np.round(mx, 9).tolist(),
                    "inflation_xyz": np.round(inflate, 9).tolist(),
                }
            else:
                if spec.center_world_xyz is None or spec.dims_world_xyz is None:
                    raise ValueError(
                        f"Obstacle {spec.name!r} requires either usd_path or center_world_xyz+dims_world_xyz"
                    )
                center_world = _as_np(spec.center_world_xyz, f"{spec.name}.center_world_xyz")
                dims_world = _as_np(spec.dims_world_xyz, f"{spec.name}.dims_world_xyz")
                report[spec.name] = {
                    "source": "explicit_world_cuboid",
                    "center_world_xyz": np.round(center_world, 9).tolist(),
                    "dims_world_xyz": np.round(dims_world, 9).tolist(),
                    "inflation_xyz": np.round(inflate, 9).tolist(),
                }
            dims = np.maximum(dims_world + inflate, 0.01)
            for side, base_world in self.base_world.items():
                T_base_obstacle = np.linalg.inv(base_world) @ pose_from_xyz(center_world)
                quat = matrix_to_quat_wxyz(T_base_obstacle[:3, :3])
                obstacles[side].append(
                    Cuboid(
                        name=f"{side}_{spec.name}",
                        dims=np.round(dims, 6).tolist(),
                        pose=np.round(np.r_[T_base_obstacle[:3, 3], quat], 9).tolist(),
                    )
                )
        midline_report = {
            "enabled": True,
            "body_midline_world_x": BODY_MIDLINE_WORLD_X,
            "wall_model": "side_specific_forbidden_halfspace_cuboid",
            "left_forbidden_world_x_range": [
                BODY_MIDLINE_WORLD_X - MIDLINE_WALL_X_THICKNESS_M,
                BODY_MIDLINE_WORLD_X,
            ],
            "right_forbidden_world_x_range": [
                BODY_MIDLINE_WORLD_X,
                BODY_MIDLINE_WORLD_X + MIDLINE_WALL_X_THICKNESS_M,
            ],
            "wall_dims_world_xyz": [
                MIDLINE_WALL_X_THICKNESS_M,
                MIDLINE_WALL_Y_SIZE_M,
                MIDLINE_WALL_Z_SIZE_M,
            ],
        }
        for side, center_x in (
            ("left", BODY_MIDLINE_WORLD_X - MIDLINE_WALL_X_THICKNESS_M * 0.5),
            ("right", BODY_MIDLINE_WORLD_X + MIDLINE_WALL_X_THICKNESS_M * 0.5),
        ):
            center_world = np.array([center_x, MIDLINE_WALL_CENTER_Y_M, MIDLINE_WALL_CENTER_Z_M], dtype=float)
            dims = np.array([MIDLINE_WALL_X_THICKNESS_M, MIDLINE_WALL_Y_SIZE_M, MIDLINE_WALL_Z_SIZE_M], dtype=float)
            T_base_obstacle = np.linalg.inv(self.base_world[side]) @ pose_from_xyz(center_world)
            quat = matrix_to_quat_wxyz(T_base_obstacle[:3, :3])
            obstacles[side].append(
                Cuboid(
                    name=f"{side}_hard_body_midline_wall",
                    dims=np.round(dims, 6).tolist(),
                    pose=np.round(np.r_[T_base_obstacle[:3, 3], quat], 9).tolist(),
                )
            )
        report["hard_body_midline_constraint"] = midline_report
        return obstacles, report

    def _init_curobo_planner(self, obstacles=None):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Scene

        obstacles = obstacles or []
        cfg = MotionPlannerCfg.create(
            robot=self.curobo_cfg,
            scene_model=Scene(cuboid=obstacles),
            collision_cache={"cuboid": max(8, len(obstacles))},
            optimizer_collision_activation_distance=0.02,
        )
        planner = MotionPlanner(cfg)
        planner.warmup(enable_graph=True, num_warmup_iterations=2)
        return planner

    def _curobo_tool_pose(self, q_arm: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        qmap = dict(zip(CUROBO_JOINTS, q_arm.tolist()))
        qmap["4C2_Joint1"] = 0.0
        T = fk(self.curobo_chain, qmap)
        return T[:3, 3], matrix_to_quat_wxyz(T[:3, :3])

    def plan_arm(
        self,
        request: ArmPlanRequest,
        obstacles_by_side: Optional[Dict[Side, List[object]]] = None,
    ) -> ArmPlanResult:
        from curobo.types import JointState

        t0 = time.perf_counter()
        start = _joint_np(request.start_joint_position_rad, "start_joint_position_rad")
        goal, goal_diag = self.solve_goal_joint(request)

        source = "fallback_joint_smoothstep"
        path = fallback_path(start, goal, count=max(2, request.sample_count))
        planner_diag: Dict[str, object] = {}
        if request.use_curobo:
            try:
                obstacles = (obstacles_by_side or {}).get(request.side, [])
                planner = self._init_curobo_planner(obstacles)
                start_q7 = np.r_[start, 0.0]
                cur = JointState.from_position(
                    torch.tensor(start_q7.reshape(1, -1), device="cuda:0", dtype=torch.float32),
                    joint_names=planner.default_joint_state.joint_names,
                )
                pos, quat = self._curobo_tool_pose(goal)
                max_attempts = max(1, int(request.curobo_max_attempts))
                result = planner.plan_pose(make_goal(pos, quat), cur, max_attempts=max_attempts)
                if result is not None:
                    plan = result.get_interpolated_plan().position.cpu().numpy()[0, 0, :, :6]
                    correction_count = max(8, int(np.linalg.norm(plan[-1] - goal) * 35))
                    correction = fallback_path(plan[-1], goal, count=correction_count)
                    path = np.vstack([plan, correction[1:]])
                    path = _resample_path(path, max(2, request.sample_count))
                    source = "curobo_plan_pose_plus_joint_correction"
                    planner_diag = {
                        "curobo_success": True,
                        "max_attempts": max_attempts,
                        "raw_sample_count": int(plan.shape[0]),
                        "target_q_error_norm_before_correction": float(np.linalg.norm(plan[-1] - goal)),
                        "correction_sample_count": int(correction.shape[0]),
                    }
                else:
                    planner_diag = {
                        "curobo_success": False,
                        "max_attempts": max_attempts,
                        "reason": "plan_pose returned None",
                    }
            except Exception as exc:
                planner_diag = {"curobo_success": False, "reason": repr(exc)}

        midline_diag = self.validate_midline_constraint(request.side, path)
        if not midline_diag["ok"]:
            source = f"{source}_midline_violation"
        elapsed = time.perf_counter() - t0
        return ArmPlanResult(
            side=request.side,
            success=bool(midline_diag["ok"]),
            source=source,
            joint_names=ARM_JOINTS,
            duration_sec=float(request.duration_sec),
            sample_count=int(len(path)),
            start_joint_position_rad=np.round(start, 9).tolist(),
            goal_joint_position_rad=np.round(goal, 9).tolist(),
            trajectory=_timestamped_points(path, request.duration_sec),
            diagnostics={
                "elapsed_sec": elapsed,
                "goal": goal_diag,
                "planner": planner_diag,
                "midline_constraint": midline_diag,
                "final_q_error_norm": float(np.linalg.norm(path[-1] - goal)),
            },
        )

    def plan_dual_arm(
        self,
        left: Optional[ArmPlanRequest],
        right: Optional[ArmPlanRequest],
        obstacles_by_side: Optional[Dict[Side, List[object]]] = None,
    ) -> DualArmPlanResult:
        t0 = time.perf_counter()
        left_result = self.plan_arm(left, obstacles_by_side) if left is not None else None
        right_result = self.plan_arm(right, obstacles_by_side) if right is not None else None
        duration = max(
            left_result.duration_sec if left_result else 0.0,
            right_result.duration_sec if right_result else 0.0,
        )
        return DualArmPlanResult(
            success=bool((left_result is None or left_result.success) and (right_result is None or right_result.success)),
            left=left_result,
            right=right_result,
            synchronized_duration_sec=duration,
            diagnostics={"elapsed_sec": time.perf_counter() - t0, "planning_mode": "sequential_per_arm"},
        )


def default_task_obstacles() -> List[CuboidObstacleSpec]:
    return [
        CuboidObstacleSpec("table", usd_path="/World/Table", inflation_xyz=[0.04, 0.04, 0.03]),
        CuboidObstacleSpec("loading_equip", usd_path="/World/LoadingEquip", inflation_xyz=[0.04, 0.04, 0.03]),
        CuboidObstacleSpec("dryer", usd_path="/World/Dryer", inflation_xyz=[0.05, 0.05, 0.05]),
    ]


def export_result_json(result, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_requests(path: Path) -> Tuple[Optional[ArmPlanRequest], Optional[ArmPlanRequest], List[CuboidObstacleSpec]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    left = ArmPlanRequest(**data["left"]) if data.get("left") else None
    right = ArmPlanRequest(**data["right"]) if data.get("right") else None
    obstacles = [CuboidObstacleSpec(**item) for item in data.get("obstacles", [])]
    return left, right, obstacles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--request-json", required=True)
    parser.add_argument("--out", default=str(PLAYGROUND_ROOT / "runtime/dual_arm_api_plan.json"))
    parser.add_argument("--default-obstacles", action="store_true")
    args = parser.parse_args()

    context = DualArmPlanningContext(scene_path=args.scene)
    left, right, obstacles = _load_requests(Path(args.request_json))
    if args.default_obstacles:
        obstacles = default_task_obstacles() + obstacles
    obstacles_by_side, obstacle_report = context.build_curobo_obstacles(obstacles)
    result = context.plan_dual_arm(left, right, obstacles_by_side)
    data = asdict(result)
    data["obstacles"] = obstacle_report
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({"success": result.success, "elapsed_sec": result.diagnostics["elapsed_sec"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
