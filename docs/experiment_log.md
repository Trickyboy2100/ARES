# Experiment Log — Left-Arm Tray Pick-to-Chest Demo

## 2026-06-09 — Initial IK + Vertical Jaw

- Established vertical jaw constraint: EG2 X = world -Z (jaw opens in world Z)
- First successful IK solution for approach/pick/lift phases
- Pad contact boxes placed at cz=0.020 in local pad frame
- Grasp failure: tray pushed sideways during approach because gripper was closing simultaneously

## 2026-06-10 — Horizontal Constraint + Chest Retarget (Milestone)

- Added `constrained_pose_ramp_path` for carry phase; up+forward both stay in world XY → tray horizontal
- Chest target now read from `/World/leftarmterminal` scene prim
- Tray start position now read dynamically from stage (not hardcoded)
- Manual play: demo waits for user to click Play before planning/executing
- Left arm initial pose fixed: stale `tray_demo_fk` xformOps reset to identity in USD

**Root cause of grasp failure** (from validation report run3):
| t=2.42s | Boxes enter ear Y range, tray_Y drops ~6mm (boxes pushing tray) |
| t=2.67s | Tray_Y jump −36mm: box inner face hitting ear top/side during approach+close |
| t=2.83s | Tray falls |

Root cause: approach + gripper close happen simultaneously. Pads push tray sideways instead of gripping.

## 2026-06-10 — Separate Slow-Close Phase (Current Work)

**Changes implemented**:
1. Approach phase now keeps gripper OPEN; gripper only closes after arm reaches pick position
2. Dedicated slow-close phase: `--close-steps` frames (default 60 = 1s at 60fps)
3. Force-based stop: `--force-threshold-m` — if tray moves more than threshold during close, stop
4. IK speed: counts 81→30, 151→60; tolerances 1e-10→1e-6 (faster planning)
5. `--auto-play` flag to skip manual play
6. `--loop` flag to auto-reset and repeat demo
7. `--cam-pullback-m` to pull back overhead camera
8. Debug logging: per-waypoint JSON to `logs/debug/YYYYMMDD_HHMMSS/waypoints.jsonl`

**Pending**: run and verify grasp success with separated approach/close phases.

---

## Test Protocol

For each run, record:
- `--close-steps` value used
- Whether force threshold was hit and at which step
- Tray Z at end of animation (should be > initial Z by at least 2cm)
- Chest center error (should be < 3cm)
- Validation report status (pass/fail)

Command:
```bash
pkill -9 -f "kit_"; sleep 3
cd /home/andyee/Developer/PG-JY/isaac_sim/playground_dual_arm_control
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL -u CONDA_EXE -u CONDA_PYTHON_EXE \
  /home/andyee/isaacsim/python.sh scripts/gui_left_arm_tray_pick_to_chest_demo.py \
  --close-steps 90 --force-threshold-m 0.003 --hold-open
```
