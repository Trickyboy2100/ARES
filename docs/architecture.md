# System Architecture — tray_grasp_cycle (Dual-Arm Handoff)

## Overview

Isaac Sim 5.1.0 demo: dual-arm tray handoff cycle.
Left arm picks tray from table → carries to handoff → right arm takes over → delivers to dryer.

## Key Files

| File | Role |
|------|------|
| `isaac_sim/simforge/demos/tray_grasp_cycle/demo.py` | Main demo: state machine, IK, physics, UI |
| `isaac_sim/simforge/demos/tray_grasp_cycle/launch.sh` | Launch script |
| `isaac_sim/simforge/demos/tray_grasp_cycle/_curobo_worker.py` | Persistent cuRobo subprocess (Python 3.13 miniconda) |
| `isaac_sim/simforge/core/kinematics.py` | FK/IK, arm chains, gripper constants |
| `isaac_sim/simforge/core/scene_utils.py` | Carrier, grasp lock, physics helpers |
| `isaac_sim/simforge/core/planning.py` | cuRobo interface, IK solvers |
| `isaac_sim/simforge/scenes/main.usd` | Main scene USD (~16 MB binary) |

## State Machine (per cycle)

```
TO_PRE_L → APPROACH_L → CLOSE_GRIP_L → LIFT_L → CARRY_L
  → R_TO_NEAR → R_APPROACH → CLOSE_GRIP_R
  → RELEASE_L → RETRACT_L → CARRY_DRYER → HOLD_DRYER
  → RELEASE_R → HOME_R → RESET_SCENE → PAUSE → [repeat]
```

- **Left arm phases**: TO_PRE_L … CARRY_L (pick tray, carry to handoff)
- **Right arm phases**: R_TO_NEAR … HOME_R (grasp tray, deliver to dryer). Right arm starts moving in parallel during TO_PRE_L (pre-moves to handoff).
- **R_TO_NEAR / R_APPROACH**: skipped (1 frame each) — right arm already at handoff via parallel path.

## Grasp Mechanism

- **Left arm**: FixedJoint from `RIGHT_PAD_L_PATH` (kinematic pad with RigidBodyAPI) → tray. Force-based contact detection during APPROACH_L.
- **Right arm**: Carrier-based grasp. Hidden kinematic cube (`_GraspCarrier_R/Carrier`) tracks right arm pad pose. FixedJoint from carrier → tray. Needed because `RIGHT_PAD_R_PATH` lacks RigidBodyAPI.

## Path Planning

- **cuRobo**: GPU-accelerated trajectory optimization, runs as persistent subprocess (miniconda python3, avoids Isaac Sim's Warp 1.8.2 conflict).
- **Right arm**: single cuRobo plan `q_zero → q_handoff_R` (81 steps), replaces old 3-segment chain.
- **Left arm approach**: IK-based Y-linear Cartesian path with force detection.
- **SG filter**: 5-point quadratic Savitzky-Golay filter smooths cuRobo path velocity jitter.
- **Path stitching**: none — each cuRobo path planned independently.

## Key Geometry

```
ROBOT_SHIFT_X = -0.15 m
HANDOFF_Z_ABS = 1.20 m (120 cm above floor)
HANDOFF_Y_OFFSET = -0.30 m
TRAY_HALF_X_HANDOFF = 0.090 m (18 cm tray Y at rest → world X at handoff)
```

## Launch

```bash
pkill -f "isaacsim/kit/kit" 2>/dev/null; sleep 2
cd /home/andyee/Developer/PG-JY/isaac_sim/simforge
bash demos/tray_grasp_cycle/launch.sh > /tmp/tgc.log 2>&1 &
```

## Design Constraints

- Only one Isaac Sim window at a time (kill before launching)
- No fake grasp: FixedJoint created only after physical contact
- cuRobo subprocess must use miniconda python3 (not Isaac Sim's bundled python)
- milestones/ directories are read-only archives
