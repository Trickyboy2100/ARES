# System Architecture — Left-Arm Tray Pick-to-Chest Demo

## Overview

Isaac Sim demo: left arm picks tray by ear, carries to chest height.
No fake grasp (no FixedJoint). Physical contact via kinematic collision boxes on gripper pads.

## Key Files

| File | Role |
|------|------|
| `isaac_sim/playground_dual_arm_control/scripts/gui_left_arm_tray_pick_to_chest_demo.py` | Main entry point, animation loop, scene setup |
| `isaac_sim/playground_dual_arm_control/scripts/make_tray_handoff_curobo_demo.py` | IK solver, path planning functions |
| `isaac_sim/playground_dual_arm_control/scripts/kinematics_probe.py` | FK utilities, URDF loading |
| `isaac_sim/playground_dual_arm_control/scripts/gui_tray_handoff_demo.py` | Gripper FK helpers |
| `/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd` | Scene USD |

## Grasp Mechanism

```
Gripper open  →  arm approaches ear from +Y  →  gripper stays open during approach
             →  arm reaches pick position     →  gripper closes slowly (--close-steps frames)
             →  lift (+Z)                     →  gripper fully closed
             →  carry to chest                →  gripper fully closed
```

Contact geometry:
- `left_pad`: kinematic rigid body, PadContactBox at cx=+0.0035, cz=0.020 (local pad frame)
- `right_pad`: kinematic rigid body, PadContactBox at cx=-0.0035, cz=0.020

Jaw orientation invariant: **EG2 X = world -Z** throughout all motion phases.
Achieved by keeping EG2 up and forward vectors in the world XY plane.

## IK Pipeline

`solve_pad_pose_ik` (scipy `least_squares`):
- Position residual: `T_pad[:3,3] - target_xyz` (3 terms)
- Up-axis residual: `axis_weight * (T_pad_Y - target_up)` (3 terms)
- Forward-axis residual: `forward_weight * (T_pad_Z - target_forward)` (3 terms)
- Continuity: `continuity_weight * (q - reference_q)` (6 terms)

Path building functions in `make_tray_handoff_curobo_demo.py`:
- `constrained_pose_path`: fixed orientation throughout a segment
- `constrained_pose_ramp_path`: interpolates both up and forward axes (used for carry phase)

## Phase Bounds

| Phase | Frames | Gripper |
|-------|--------|---------|
| move_to_pregrasp | 91 | open |
| approach | ~30 | open |
| slow_close | --close-steps (default 60) | 0→1 |
| lift | ~30 | closed |
| carry_to_chest | ~60 | closed |

## Launch Command

```bash
cd /home/andyee/Developer/PG-JY/isaac_sim/playground_dual_arm_control
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL -u CONDA_EXE -u CONDA_PYTHON_EXE \
  /home/andyee/isaacsim/python.sh scripts/gui_left_arm_tray_pick_to_chest_demo.py --hold-open

# Auto-play (no manual play button needed):
  ... --auto-play --hold-open

# Slow gripper close (120 frames = 2s at 60fps):
  ... --close-steps 120

# Force-based gripper stop (stop if tray moves >3mm during close):
  ... --force-threshold-m 0.003

# Auto-loop demo:
  ... --loop

# Pull camera back 40cm:
  ... --cam-pullback-m 0.4
```

## Design Constraints

- **No fake grasp**: FixedJoint binding tray to gripper is strictly forbidden
- **Vertical jaw**: jaw closes in world Z direction (EG2 X = world -Z always)
- **No milestone edits**: milestones/ directories are read-only archives
