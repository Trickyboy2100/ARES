# SimForge — Dual-Arm Robot Simulation Lab

Structured simulation codebase for the JAKA MiniCobo dual-arm + Inspire EG2-4C2 gripper system running in **Isaac Sim 5.1.0**.

---

## Directory layout

```
simforge/
├── config.py              ← centralised path configuration (read this first)
├── core/                  ← shared Python modules
│   ├── kinematics.py      ← URDF loading, FK, joint chain helpers
│   ├── kinematics_probe.py← backward-compat alias of kinematics.py
│   ├── planning.py        ← cuRobo IK solver, path planning
│   ├── gripper.py         ← EG2-4C2 xform/drive control, FK, pad geometry
│   ├── scene_utils.py     ← USD helpers (bbox, xform ops, friction, joints)
│   └── ik_sanity.py       ← joint-limit extraction from URDF chain
├── demos/
│   ├── open_scene.py      ← open scene in GUI, no motion
│   ├── dual_arm_draw.py   ← right arm circles, left arm squares (integration test)
│   ├── ear_grasp_lift.py  ← left arm picks up tray by ears, lifts
│   ├── pick_to_chest.py   ← left arm pick → carry to chest handoff position
│   └── tray_handoff.py    ← full dual-arm tray handoff sequence
├── tools/                 ← one-shot USD setup / conversion utilities
├── scenes/
│   ├── main.usd           ← tracked scene snapshot (see note below)
│   └── checkpoint.sh      ← copy active playground scene → main.usd + git commit
└── milestones/
    └── INDEX.md           ← index of archived milestone results
```

---

## Prerequisites

| Requirement | Version / notes |
|-------------|----------------|
| **Isaac Sim** | 5.1.0-rc.19 (tested), installed at `~/isaacsim` |
| **CUDA** | 12.1 (for `LD_PRELOAD` workaround below) |
| **Python packages** | `numpy`, `scipy` — available inside Isaac Sim's Python |
| **cuRobo** | installed into Isaac Sim's Python env |
| **JAKA minicobo URDF** | `jaka_ros2/src/jaka_description/urdf/jaka_minicobo.urdf` |
| **cuRobo YAML** | `jinyu_ros_pkg/nodes/simulation/jaka_minicobo_curobo.yml` |
| **Scene USD** | `2026061100_main.usd` in the Isaac Sim playground directory |

---

## Setup — configure paths

All external paths are set in `simforge/config.py`.  
Override any of them with environment variables (e.g. in `~/.bashrc`):

```bash
# Where Isaac Sim is installed (default: ~/isaacsim)
export ISAACSIM_ROOT=~/isaacsim

# The main scene USD file
export SIMFORGE_SCENE=~/isaacsim/playground/2026061100_main.usd

# Directory containing jaka_minicobo.urdf and jaka_minicobo_gripper.urdf
export SIMFORGE_URDF_DIR=~/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf

# cuRobo YAML config for the arm
export SIMFORGE_CUROBO_CFG=~/Developer/PG-JY/jinyu_ros_pkg/nodes/simulation/jaka_minicobo_curobo.yml
```

Check everything is wired up:

```bash
python3 isaac_sim/simforge/config.py
```

---

## Running demos

**Always launch via `isaac-sim.sh --exec`** (not `python.sh` — that triggers the Storm renderer and breaks physics).

```bash
# Integration test — dual-arm drawing (circle + square)
LD_PRELOAD=~/isaacsim/extscache/omni.isaac.lula-*/lib/libcusparse.so.12 \
  ~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/dual_arm_draw.py

# Left arm tray ear-grasp + lift
LD_PRELOAD=~/isaacsim/extscache/omni.isaac.lula-*/lib/libcusparse.so.12 \
  ~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/ear_grasp_lift.py

# Open scene only (no motion)
~/isaacsim/isaac-sim.sh --exec \
  isaac_sim/simforge/demos/open_scene.py
```

After the scene loads and IK pre-computation finishes, the console prints:
```
[DRAW] Ready — press Play to start the drawing loop.
```
Press **▶ Play** in the GUI to start the motion.

> **LD_PRELOAD note**: The exact path to `libcusparse.so.12` may differ.  
> Try: `find ~/isaacsim -name "libcusparse.so.12" 2>/dev/null | head -1`  
> Or use: `LD_PRELOAD=/usr/local/cuda-12.1/targets/x86_64-linux/lib/libcusparse.so.12`

---

## Scene USD notes

`simforge/scenes/main.usd` is a tracked snapshot for reference.  
**It has broken relative USD references** when run from the simforge directory, because the original scene uses paths relative to `~/isaacsim/playground/`.

Use the original scene file for running demos (set via `SIMFORGE_SCENE`).  
Use `scenes/checkpoint.sh` to update the snapshot after scene changes:

```bash
cd isaac_sim/simforge/scenes
./checkpoint.sh "feat(scene): description of change"
```

---

## Key design constraints

1. **No fake grasps** — `FixedJoint` is only created after `force_stop_step is not None` (physical contact confirmed).
2. **Gripper orientation** — EG2 X-axis = world -Z direction; jaw closes in Z.
3. **Milestones are read-only** — `milestones/INDEX.md` lists all archived runs.
4. **One Isaac Sim window at a time** — kill any existing instance before launching.

---

## Module quick-reference

### `core/kinematics.py`
```python
from kinematics import load_joints, chain_to_link, fk, ARM_JOINTS, DEFAULT_ARM_URDF, GRIPPER_ROOT_SUFFIX
arm_joints  = load_joints(Path(DEFAULT_ARM_URDF))
link6_chain = chain_to_link(arm_joints, "Link_0", "Link_6")
T_world_link6 = base_world @ fk(link6_chain, {"joint_1": 0.5, ...})
```

### `core/gripper.py`
```python
from gripper import gripper_link_transform, setup_gripper_xform_ops, pad_separation_m
T_link = gripper_link_transform("left_outer_link", joint_angle_rad)
gap_m  = pad_separation_m(0.1945)   # → 0.05 m
```

### `core/planning.py`
```python
from planning import selected_pad_midpoint, solve_pad_pose_ik
base_world, pad_world, link6_to_pad = selected_pad_midpoint(stage, cache, "left")
q, pos_err, up_err, fwd_err, ok, msg = solve_pad_pose_ik(
    link6_chain, lower, upper, base_world, link6_to_pad,
    target_xyz, up_world, forward_world, seeds)
```

### `core/scene_utils.py`
```python
from scene_utils import gf_matrix_from_column_transform, create_grasp_lock
op.Set(gf_matrix_from_column_transform(T_4x4))   # set USD xform from numpy matrix
joint_path = create_grasp_lock(stage, tray_path, carrier_path, joint_path)
```
