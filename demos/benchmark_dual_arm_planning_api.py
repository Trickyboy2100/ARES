#!/usr/bin/env python3
"""Benchmark the reusable dual-arm planning API."""


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
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from dual_arm_planning_api import (
    ArmPlanRequest,
    CuboidObstacleSpec,
    DualArmPlanningContext,
    default_task_obstacles,
)


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]


def ms(values):
    return {
        "min_ms": round(min(values) * 1000.0, 3),
        "mean_ms": round(statistics.mean(values) * 1000.0, 3),
        "max_ms": round(max(values) * 1000.0, 3),
        "runs": len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default=str(PLAYGROUND_ROOT / "reports/dual_arm_planning_api_benchmark.json"))
    parser.add_argument("--no-curobo", action="store_true", help="Benchmark fallback interpolation only.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="cuRobo plan_pose max_attempts per arm request.",
    )
    parser.add_argument(
        "--obstacle-preset",
        choices=("default", "single-cube", "none"),
        default="default",
        help="Obstacle set used for cuRobo scene construction.",
    )
    args = parser.parse_args()

    context_start = time.perf_counter()
    context = DualArmPlanningContext()
    if args.obstacle_preset == "default":
        obstacle_specs = default_task_obstacles()
    elif args.obstacle_preset == "single-cube":
        obstacle_specs = [
            CuboidObstacleSpec(
                name="single_cube",
                center_world_xyz=[0.0, 0.5, 0.85],
                dims_world_xyz=[0.5, 0.5, 0.5],
                inflation_xyz=[0.0, 0.0, 0.0],
            )
        ]
    else:
        obstacle_specs = []
    obstacles_by_side, obstacle_report = context.build_curobo_obstacles(obstacle_specs)
    context_elapsed = time.perf_counter() - context_start

    use_curobo = not args.no_curobo
    left_request = ArmPlanRequest(
        side="left",
        start_joint_position_rad=[0, 0, 0, 0, 0, 0],
        goal_type="pad_position",
        goal_pad_world_xyz=[0.2725, 0.35, 1.183],
        duration_sec=2.0,
        sample_count=121,
        use_curobo=use_curobo,
        curobo_max_attempts=args.max_attempts,
    )
    right_request = ArmPlanRequest(
        side="right",
        start_joint_position_rad=[0, 0, 0, 0, 0, 0],
        goal_type="pad_position",
        goal_pad_world_xyz=[-0.0775, 0.65, 1.2],
        duration_sec=2.0,
        sample_count=121,
        use_curobo=use_curobo,
        curobo_max_attempts=args.max_attempts,
    )

    left_times = []
    right_times = []
    dual_times = []
    last_results = {}
    for _ in range(max(1, args.runs)):
        t0 = time.perf_counter()
        left_result = context.plan_arm(left_request, obstacles_by_side)
        left_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        right_result = context.plan_arm(right_request, obstacles_by_side)
        right_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        dual_result = context.plan_dual_arm(left_request, right_request, obstacles_by_side)
        dual_times.append(time.perf_counter() - t0)

        last_results = {
            "left": asdict(left_result),
            "right": asdict(right_result),
            "dual": asdict(dual_result),
        }

    report = {
        "runs": max(1, args.runs),
        "use_curobo": use_curobo,
        "max_attempts": args.max_attempts,
        "obstacle_preset": args.obstacle_preset,
        "context_init_sec": context_elapsed,
        "obstacles": obstacle_report,
        "timing": {
            "single_left": ms(left_times),
            "single_right": ms(right_times),
            "dual_sequential_left_plus_right": ms(dual_times),
        },
        "requests": {
            "left": asdict(left_request),
            "right": asdict(right_request),
        },
        "last_results_summary": {
            "left_source": last_results["left"]["source"],
            "right_source": last_results["right"]["source"],
            "dual_success": last_results["dual"]["success"],
            "left_goal": last_results["left"]["diagnostics"]["goal"],
            "right_goal": last_results["right"]["diagnostics"]["goal"],
            "left_planner": last_results["left"]["diagnostics"]["planner"],
            "right_planner": last_results["right"]["diagnostics"]["planner"],
            "left_midline": last_results["left"]["diagnostics"]["midline_constraint"],
            "right_midline": last_results["right"]["diagnostics"]["midline_constraint"],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["timing"], indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
