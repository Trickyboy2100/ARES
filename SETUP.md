# Quick Setup & Run Guide

> **Goal**: Run the dual-arm tray handover demo on a fresh machine.  
> Expected time: 30–60 min (mostly Isaac Sim installation).

---

## Step 1 — Clone the repo

```bash
git clone git@github.com:Trickyboy2100/ARES.git ~/simforge
cd ~/simforge
```

---

## Step 2 — Install Isaac Sim 5.1.0

Isaac Sim is the only external dependency not included in this repo.

**Option A: Omniverse Launcher (recommended)**
1. Download the Launcher from https://developer.nvidia.com/isaac/sim
2. Install **Isaac Sim 5.1.0** from the Launcher
3. Default install path: `~/isaacsim/`

**Option B: tar install**
```bash
tar -xzf isaac-sim-5.1.0-linux-x86_64.tar.gz -C ~/
mv isaac-sim-5.1.0 ~/isaacsim
```

Verify:
```bash
ls ~/isaacsim/isaac-sim.sh   # must exist
```

If Isaac Sim is installed somewhere other than `~/isaacsim/`, set:
```bash
export ISAACSIM_ROOT=/path/to/your/isaacsim
```

---

## Step 3 — Install cuRobo (for motion planning)

cuRobo runs in a **separate Python environment** from Isaac Sim.  
The demo works without cuRobo (falls back to direct joint jumps), but motion will be jerky.

```bash
# Using Miniconda (recommended)
conda create -n curobo python=3.13 -y
conda activate curobo
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install curobo

# Tell the demo where to find this Python:
export CUROBO_PYTHON=~/miniconda3/envs/curobo/bin/python3
```

Add `export CUROBO_PYTHON=...` to your `~/.bashrc` so it persists.

---

## Step 4 — Run the demo

```bash
bash ~/simforge/demos/tray_grasp_cycle/launch.sh
```

Isaac Sim will open. The **first launch** compiles shader caches (~3–5 min).  
Subsequent launches take ~30–60 seconds.

Once loaded you will see:
- The dual-arm lab scene with table, tray, and dryer
- A **Tray Grasp Cycle** monitor panel on the right (force & position curves)
- The left arm begins moving automatically toward the tray

### Demo phases

```
PAUSE → TO_PRE_L → APPROACH_L → CLOSE_GRIP_L → LIFT_L
      → CARRY_L  → HANDOFF    → CARRY_DRYER  → RESET_SCENE → [repeat]
```

The cycle repeats automatically. Each cycle takes roughly 20–40 seconds.

---

## Step 5 — Stop the demo

Close the Isaac Sim window, or from a terminal:

```bash
pkill -f "isaacsim/kit/kit"
```

---

## Environment variables (all optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `ISAACSIM_ROOT` | `~/isaacsim` | Isaac Sim installation directory |
| `CUROBO_PYTHON` | `~/miniconda3/bin/python3` | Python with cuRobo installed |
| `SIMFORGE_SCENE` | `scenes/main.usd` | Override scene file |
| `SIMFORGE_URDF_DIR` | `robot/` | Override robot URDF directory |

---

## Troubleshooting

**Semaphore hang on startup**
```bash
rm -f /dev/shm/sem.carbonite-sharedmemory
# then relaunch
```

**`ld.so Assertion` crash**  
You are launching Isaac Sim directly instead of via `launch.sh`.  
Always use `bash demos/tray_grasp_cycle/launch.sh`.

**Scene renders as white background + black silhouettes**  
The `scenes/main.usd` in this repo is fully self-contained (all assets in `assets/`).  
If you see this, the file may have been replaced with an unpatched copy.  
Run: `bash scenes/checkpoint.sh "restore"` from the repo root to fix it.

**cuRobo subprocess slow on first run**  
cuRobo compiles CUDA kernels on first use (~2–5 min). Subsequent runs use cache.

**Demo starts but arms don't move**  
Check the terminal for `[TGC] ERROR`. Common causes:
- Scene not loaded: verify `scenes/main.usd` exists
- URDF not found: run `ls robot/jaka_minicobo.urdf`
