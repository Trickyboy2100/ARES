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

---

## 2026-06-23 — tray_grasp_cycle: Persistent Worker + Path Smoothing + Parallel Arms

### Persistent cuRobo Worker
- Worker (`_curobo_worker.py`) now runs in loop mode: reads JSON batches from stdin, writes results to stdout
- Eliminates ~2s startup/warmup overhead per planning call
- Client (`_cu_batch`) communicates via `subprocess.Popen` pipes

### Path Smoothing
- 5-point quadratic Savitzky-Golay filter (`_smooth_path`) applied to all cuRobo paths
- Reduces high-frequency velocity jitter inherent in cuRobo's `get_interpolated_plan()`
- No endpoint pinning — cuRobo may choose different elbow config than IK solution

### Parallel Right Arm
- Right arm starts moving to handoff during left arm's TO_PRE_L (not after CARRY_L)
- Single cuRobo plan `q_zero → q_handoff_R` replaces old pre→near→approach 3-segment chain
- R_TO_NEAR and R_APPROACH phases skip instantly (1 frame each)

### Carrier Orientation Fix
- `ensure_hidden_carrier`: `AddTranslateOp` → `AddTransformOp` (full 6-DOF)
- `set_carrier` now accepts full 4×4 matrix (position + orientation)
- `create_grasp_lock` rotation constraint: `localRot1 = tray⁻¹ @ carrier` (was `tray⁻¹`, assumed carrier at identity)

### Known Issues
- ~5 state-transition jumps remain (cuRobo plans independently per segment)
- Retract/home paths snap to q_zero at end
- Isaac Sim occasionally OOM-killed (memory pressure from persistent worker + simulation)
