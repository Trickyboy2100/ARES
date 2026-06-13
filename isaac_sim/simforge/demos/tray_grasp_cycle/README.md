# tray_grasp_cycle — Tray Ear Grasp + Lift Loop Demo

Left-arm picks the tray ear tab, lifts +30 cm, lowers, releases, retracts, and repeats — indefinitely. Real-time dual-arm data panel via `omni.ui`.

---

## Quick Start

```bash
# From repo root (auto-kills any existing Isaac Sim):
bash isaac_sim/simforge/demos/tray_grasp_cycle/launch.sh

# With full log capture and auto-generated report:
bash isaac_sim/simforge/demos/tray_grasp_cycle/record.sh
```

---

## Demo Overview

```
SETTLE (120 fr)
  └─ physics settle; both arms held at q=0
READ tray_grasp_point
  └─ reads /World/Tray/tray_grasp_point world pos (settled)
IK PLAN
  └─ solve 4 waypoints: pre / near / pick / lift
CYCLE (repeats):
  TO_PRE    → arm moves to pre-grasp (Y+0.125 from ear face, no contact)
  TO_NEAR   → arm moves to near     (Y+0.040, no contact)
  APPROACH  → arm advances toward pick, monitors friction each frame
              ⚡ FORCE STOP: halts when F_friction ≥ 3 N
  CLOSE_GRIP → gripper closes 0.1945 → 0.0 rad (visual hold)
  LIFT      → arm lifts +0.30 m; tray kinematically tracks pad ΔZ
  HOLD      → 60 frames at peak height
  LOWER     → arm returns to pick height
  RELEASE   → gripper opens
  RETRACT   → arm returns to pre-grasp
  HOME      → arm returns to q=0
  PAUSE     → 90 frames, then next cycle
```

---

## Contact Force Model

The ear tab extends in world **+Y** from the tray body (horizontal, parallel to ground). The gripper approaches from **+Y** along a line parallel to the Y axis passing through `tray_grasp_point`.

```
Pad face presses on ear face (Y axis):
  contact_y = ear_face_y + PAD_FACE_DEPTH_M   (= ear_y + 0.028 m)
  
  When pad_mid_y < contact_y:
    press_depth = contact_y − pad_mid_y
    actual_y    = pad_mid_y + K_ear/(K_arm + K_ear) · press_depth
    F_normal    = K_arm · (actual_y − pad_mid_y)   [per pad]
    F_friction  = 2 · μ · F_normal                  [total, both pads]

Constants:
  K_arm = 300 N/m    K_ear = 3000 N/m    μ = 1.5
  PAD_FACE_DEPTH_M = 0.028 m
  GRIP_FORCE_STOP_N = 3.0 N
```

Friction in the XZ plane (perpendicular to the Y normal) supports the tray vertically via the Z component.

---

## Tray Kinematic Lift

During LIFT/HOLD/LOWER, the tray becomes a kinematic body and tracks the left-pad Z displacement:

```python
delta_z = pad_z_now − pad_z_at_grasp
tray_translate_z = tray_z_at_grasp + delta_z
```

The tray is released back to dynamic physics at RELEASE.

---

## GUI — Dual Arm Monitor

Window: **"Dual Arm Grasp Monitor"** (600 × 680 px)

| Section | Contents |
|---------|---------|
| **Left arm** | Pad EE world XYZ · contact indicator · gripper angle + gap · friction force bar + history chart · pad-Y approach curve · lift height bar |
| **Right arm** | Pad EE world XYZ at home · "HOLDING HOME q=[0,0,0,0,0,0]" status |
| **Footer** | K values · force-stop threshold · lift target |

---

## Key USD Prims

| Path | Role |
|------|------|
| `/World/robot/jaka_minicobo_left` | Left arm root |
| `/World/robot/jaka_minicobo_right` | Right arm root (held at q=0) |
| `/World/Tray` | Tray rigid body (becomes kinematic during lift) |
| `/World/Tray/tray_grasp_point` | **Calibrated grasp reference** — Xform prim at ear face position |

---

## Verified Results (2026-06-13)

| Metric | Value |
|--------|-------|
| IK pre-grasp pos_err | 0.0 mm |
| Force-stop trigger | F = 3.21 N at pad_y = 0.4191 |
| Lift achieved | 29.6 cm |
| Hold force | ≈ 10 N |
| Loop stability | 5+ cycles confirmed |

---

## Design Constraints

- **No PhysX pads** — pure analytical contact model (two-spring Y-axis)
- **No FixedJoint** — kinematic tray translate tracking only
- **Both arms FK every frame** — right arm at q=0 to prevent physics chaos
- **UI created BEFORE `timeline.play()`** — ordering is critical
- **Read `tray_grasp_point` AFTER settle** — so tray has reached rest position
