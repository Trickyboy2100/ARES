# Kinematics Strategy

Created: 2026-06-08

## Current Decision

Use the URDF files as the kinematic source of truth, and use the USD scene as the source of world placement and mounted tool geometry.

This is based on the first probe:

- `Link_6` in USD agrees with JAKA URDF zero-pose FK to about `1.4e-7 m`.
- The current USD has `Link_1` through `Link_6` flattened as sibling prims under each arm root.
- That flattened scene can look visually correct, but it is not a clean parent-child articulation for PhysX control.

## Frame Names

- Arm base:
  `/World/robot/jaka_minicobo_<side>/Link_0`

- Bare arm wrist/flange:
  `/World/robot/jaka_minicobo_<side>/Link_6`

- Bare arm TCP:
  `/World/robot/jaka_minicobo_<side>/Link_6/dummy_tcp`

  In the current JAKA URDF this is coincident with `Link_6`.

- Mounted gripper root:
  `/World/robot/jaka_minicobo_<side>/Link_6/CAM_Mount/force_sensor/gripper_flange/Inspire_EG2_4C2`

- Approximate current USD gripper functional point:
  midpoint of `left_pad` and `right_pad`

- cuRobo tool frame:
  `4C2_Link5` from `jaka_minicobo_gripper.urdf`

## Cross-Validation Loop

1. Read current USD world transforms for each arm base, bare wrist/TCP, gripper root, and pad midpoint.
2. Compute URDF FK from `Link_0`/`base_link` to the same intended frame.
3. Compare `T_world_base @ T_base_tip_fk` against USD.
4. Treat mismatches as either:
   - frame naming/calibration mistakes, or
   - invalid USD hierarchy/articulation setup.
5. Only after the kinematic comparison is clean, re-enable dynamics/articulation control.

## Immediate Next Steps

1. Promote `scripts/kinematics_probe.py` into the standard sanity check before every control experiment.
2. Add IK over the JAKA six-axis URDF chain, initially independent of PhysX.
3. Calibrate the cuRobo `4C2_Link5` tool frame against the actual USD gripper pad midpoint.
4. Build a clean nested articulation scene or regenerate the robot USD from URDF instead of trying to drive the flattened sibling-link import.
5. Add a live logger in Isaac that records target joint values, measured link poses, FK predictions, and tool-frame deltas per frame.
