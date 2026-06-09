# Dual Arm Planning API

Created: 2026-06-09 CST

This note documents the reusable planning wrapper added after the tray handoff demo reached a stable visual baseline.

## Files

```text
scripts/dual_arm_planning_api.py
scripts/benchmark_dual_arm_planning_api.py
examples/dual_arm_plan_request.json
reports/dual_arm_planning_api_benchmark_fallback.json
reports/dual_arm_planning_api_benchmark_curobo.json
reports/dual_arm_planning_api_benchmark_attempts8.json
reports/dual_arm_planning_api_benchmark_attempts4.json
reports/dual_arm_planning_api_benchmark_attempts2.json
reports/dual_arm_planning_api_benchmark_midline_wall_attempts8.json
reports/dual_arm_planning_api_benchmark_single_cube.json
reports/dual_arm_planning_api_benchmark_no_obstacles.json
runtime/dual_arm_api_plan_example.json
```

## What It Encapsulates

`dual_arm_planning_api.py` turns external task-module inputs into joint trajectories:

1. Load scene and arm/gripper calibration.
2. Accept obstacle inputs before planning.
3. Accept start/end conditions per arm.
4. Solve goal IK if the endpoint is specified in pad/task space.
5. Optionally call cuRobo for obstacle-aware motion.
6. Enforce the hard body-midline separation rule.
7. Return timestamped joint trajectory points.

The output is intentionally close to what a later JAKA mini 2 SDK adapter needs:

```json
{
  "time_from_start_sec": 1.25,
  "joint_position_rad": [0, 0, 0, 0, 0, 0],
  "joint_position_deg": [0, 0, 0, 0, 0, 0]
}
```

Velocity/acceleration are not generated yet. A real SDK bridge should add retiming, speed limits, blending, and controller-specific command conversion.

## Hard Body-Midline Constraint

The dual-arm robot does not need arm-crossing motions or real-time avoidance
between the two arms. The planning API therefore hard-codes the robot body
midline as `world X = 0.0`:

```text
left arm gripper pads and pad midpoint:  world X >= 0
right arm gripper pads and pad midpoint: world X <= 0
```

This is enforced in three places:

1. `build_curobo_obstacles()` always appends a side-specific cuboid wall to the
   cuRobo obstacle list. The left arm receives a forbidden wall covering the
   robot-right side, and the right arm receives the mirrored wall.
2. `solve_goal_joint()` rejects pad-space targets that are already on the wrong
   side of the body midline.
3. `plan_arm()` validates every output trajectory sample for `left_pad`,
   `right_pad`, and `pad_midpoint`. If any gripper point crosses the midline,
   the returned `ArmPlanResult.success` is false and the violation samples are
   written into `diagnostics.midline_constraint`.

The midline walls are not optional task obstacles; they are part of the robot
planning policy for this project.

## Python API

```python
from dual_arm_planning_api import (
    ArmPlanRequest,
    CuboidObstacleSpec,
    DualArmPlanningContext,
    default_task_obstacles,
)

context = DualArmPlanningContext()
obstacles_by_side, obstacle_report = context.build_curobo_obstacles(
    default_task_obstacles()
    + [
        CuboidObstacleSpec(
            name="extra_box",
            center_world_xyz=[0.0, 0.5, 0.9],
            dims_world_xyz=[0.2, 0.2, 0.2],
            inflation_xyz=[0.02, 0.02, 0.02],
        )
    ]
)

left = ArmPlanRequest(
    side="left",
    start_joint_position_rad=[0, 0, 0, 0, 0, 0],
    goal_type="pad_position",
    goal_pad_world_xyz=[0.2725, 0.35, 1.183],
    duration_sec=2.0,
    sample_count=121,
    use_curobo=True,
    curobo_max_attempts=8,
)

result = context.plan_arm(left, obstacles_by_side)
```

Dual-arm call:

```python
dual = context.plan_dual_arm(left_request, right_request, obstacles_by_side)
```

`plan_dual_arm` currently plans each arm sequentially. It does not yet do coupled dual-arm collision checking or time-parameterized synchronization beyond common duration metadata.

## CLI API

```bash
cd /home/andyee/Developer/PG-JY/isaac_sim/playground_dual_arm_control
python3 scripts/dual_arm_planning_api.py \
  --request-json examples/dual_arm_plan_request.json \
  --default-obstacles \
  --out runtime/dual_arm_api_plan_example.json
```

Request format:

```json
{
  "left": {
    "side": "left",
    "start_joint_position_rad": [0, 0, 0, 0, 0, 0],
    "goal_type": "pad_position",
    "goal_pad_world_xyz": [0.2725, 0.35, 1.183],
    "duration_sec": 2.0,
    "sample_count": 121,
    "use_curobo": true
  },
  "right": {
    "side": "right",
    "start_joint_position_rad": [0, 0, 0, 0, 0, 0],
    "goal_type": "pad_pose",
    "goal_pad_world_xyz": [-0.0775, 0.65, 1.2],
    "goal_pad_up_world": [0, 0, 1],
    "goal_pad_forward_world": [0, -1, 0],
    "duration_sec": 2.0,
    "sample_count": 121,
    "use_curobo": true
  },
  "obstacles": []
}
```

Supported goal types:

```text
joint
pad_position
pad_pose
```

Obstacle inputs:

```json
{
  "name": "explicit_box",
  "center_world_xyz": [0.0, 0.5, 0.9],
  "dims_world_xyz": [0.2, 0.2, 0.2],
  "inflation_xyz": [0.02, 0.02, 0.02]
}
```

or:

```json
{
  "name": "table",
  "usd_path": "/World/Table",
  "inflation_xyz": [0.04, 0.04, 0.03]
}
```

## Current Benchmark

Measured on 2026-06-09 with the current implementation.

Fallback interpolation, no cuRobo:

```text
single_left mean ~= 64.55 ms
single_right mean ~= 71.69 ms
dual sequential mean ~= 122.76 ms
```

cuRobo enabled, one run:

```text
max_attempts=20
single_left ~= 5201 ms
single_right ~= 3186 ms
dual sequential ~= 6404 ms
```

cuRobo enabled after exposing and lowering `ArmPlanRequest.curobo_max_attempts`:

```text
max_attempts=8
single_left ~= 3841 ms
single_right ~= 1581 ms
dual sequential ~= 3296 ms

max_attempts=4
single_left ~= 3889 ms
single_right ~= 1594 ms
dual sequential ~= 3324 ms

max_attempts=2
single_left ~= 3948 ms
single_right ~= 1597 ms
dual sequential ~= 3325 ms
```

All three lowered-attempts runs still used
`curobo_plan_pose_plus_joint_correction` for both arms. Lowering from `20` to
`8` is a useful first optimization for the current API and roughly halves the
dual sequential wall time for this target pair. Lowering below `8` did not
materially improve this benchmark, so `8` is the current default.

cuRobo enabled with hard body-midline walls, `max_attempts=8`:

```text
single_left ~= 3804 ms
single_right ~= 1713 ms
dual sequential ~= 3340 ms
left worst midline signed margin ~= 0.255 m
right worst midline signed margin ~= 0.067 m
midline violations = 0
source = curobo_plan_pose_plus_joint_correction
```

cuRobo enabled with the three task obstacles replaced by one explicit cube at
`center_world_xyz=[0.0, 0.5, 0.85]`, `dims_world_xyz=[0.5, 0.5, 0.5]`:

```text
single_left ~= 5367 ms
single_right ~= 2910 ms
dual sequential ~= 5544 ms
```

cuRobo enabled with no obstacles:

```text
single_left ~= 5214 ms
single_right ~= 2676 ms
dual sequential ~= 5236 ms
```

The single-cube and no-obstacle runs still used
`curobo_plan_pose_plus_joint_correction`, so these are not fallback timings.
The current result is that simplifying the obstacle set does not produce a
large speedup. The dominant cost is planner construction/warmup and the
pose-goal solve/correction path inside each API call. If the real task cannot
reuse cached planners because every request changes the planning problem, the
next useful performance steps are lower `curobo_max_attempts`, parallel left
and right planning when the task permits, and reducing the post-cuRobo joint
correction path by choosing cleaner joint or pose goals.

These cuRobo timings include planner construction and warmup inside each `plan_arm` call. For a real workflow where every request changes enough that planner reuse is not available, this timing is closer to the relevant end-to-end planning cost than a cached-query-only benchmark.

## Current Limitations

- Dual-arm planning is sequential per arm, not coupled two-arm collision checking.
- Output is position-only joint trajectory; no velocity/acceleration/jerk retiming yet.
- cuRobo planners are initialized per call, so benchmark timing is conservative.
- The scene remains a kinematic visual-control scene, not a dynamics-ready articulation.
- JAKA mini 2 SDK execution is not implemented yet; the API output is shaped for a later adapter.
