#!/usr/bin/env python3
"""SimForge integration test — dual-arm geometric drawing.

Right arm: end-effector continuously traces a circle (R=0.08 m, XZ plane).
Left  arm: end-effector continuously traces a square (side=0.12 m, XZ plane).
Left  gripper: fully closed  (0.0 rad → 35.4 mm gap).
Right gripper: fully open    (0.6241 rad → 85.4 mm gap).

Exercises the full simforge/core/ stack:
  kinematics.py   URDF chain loading, FK
  planning.py     cuRobo IK for shape waypoints
  gripper.py      xform-based gripper control, gripper_link_transform
  scene_utils.py  gf_matrix_from_column_transform, xform_world_xyz

Usage:
  LD_PRELOAD=.../libcusparse.so.12 isaac-sim.sh --exec dual_arm_draw.py
  Then press Play in the GUI.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

_CORE_DIR = str(Path(__file__).resolve().parents[1] / "core")
if _CORE_DIR not in sys.path:
    sys.path.insert(0, _CORE_DIR)

# ── Scene and robot paths ──────────────────────────────────────────────────────
try:
    import config as _cfg
    DEFAULT_SCENE = _cfg.SCENE_USD
except Exception:
    DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026061100_main.usd"

LEFT_ROOT  = "/World/robot/jaka_minicobo_left"
RIGHT_ROOT = "/World/robot/jaka_minicobo_right"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]

# ── Drawing parameters ────────────────────────────────────────────────────────

# Circle (right arm): radius in the XZ plane, centred on seed EE position
CIRCLE_RADIUS = 0.08    # 8 cm
CIRCLE_N      = 32      # waypoints per revolution
CIRCLE_AXIS1  = np.array([1.0, 0.0, 0.0], dtype=float)   # world X
CIRCLE_AXIS2  = np.array([0.0, 0.0, 1.0], dtype=float)   # world Z

# Square (left arm): side=0.12 m in the XZ plane, centred on seed EE position
SQUARE_HALF       = 0.06   # half-side (→ 12 cm × 12 cm square)
SQUARE_STEPS_SIDE = 16     # interpolation steps per side (4 sides → 64 total)

# Playback speed (frames per waypoint step)
FRAMES_PER_STEP = 2   # slower = smoother visual, 2 ≈ 1 s per circle revolution

# ── Gripper angles for this demo ──────────────────────────────────────────────

LEFT_GRIPPER_ANGLE  = 0.0      # fully closed  → 35.4 mm gap
RIGHT_GRIPPER_ANGLE = 0.6241   # fully open    → 85.4 mm gap

# ── IK orientation constraints (same as ear-grasp demo: vertical jaw) ─────────
#   EG2 Y = world +X  (TARGET_UP_WORLD)
#   EG2 Z = world -Y  (approach direction)
UP_WORLD      = np.array([1.0,  0.0, 0.0], dtype=float)
FORWARD_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)
ORIENT_WEIGHT = 0.20

# ── IK seeds (multiple per arm for better coverage) ──────────────────────────
# Same bank as ear-grasp demo — starting points only, NOT target configs
LEFT_SEEDS = [
    np.array([1.0,  0.5,  0.4, -1.1, -0.2, -0.4], dtype=float),
    np.array([1.0,  0.1,  1.0, -1.5, -0.3, -0.4], dtype=float),
    np.array([1.0,  0.2,  0.8, -1.4, -0.2, -0.5], dtype=float),
    np.array([1.0,  0.5,  0.6, -1.3, -0.3, -0.6], dtype=float),
]
RIGHT_SEEDS = [
    np.array([-1.0,  0.5,  0.4, -1.1,  0.2,  0.4], dtype=float),
    np.array([-1.0,  0.1,  1.0, -1.5,  0.3,  0.4], dtype=float),
    np.array([-1.0,  0.2,  0.8, -1.4,  0.2,  0.5], dtype=float),
    np.array([-1.0,  0.5,  0.6, -1.3,  0.3,  0.6], dtype=float),
]

# Drawing target height — shapes are centred on the q=0 pad world positions,
# lifted to at least DRAW_MIN_Z so they stay above the table surface.
DRAW_MIN_Z = 1.05   # m — minimum centre height

# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_arm(ops: dict, chains: dict, q: np.ndarray):
    from kinematics import ARM_JOINTS, fk
    from scene_utils import gf_matrix_from_column_transform
    q_map = dict(zip(ARM_JOINTS, q.tolist()))
    for link, op in ops.items():
        op.Set(gf_matrix_from_column_transform(fk(chains[link], q_map)))


def _set_gripper(gripper_ops: dict, angle_rad: float):
    from gripper import gripper_link_transform
    from scene_utils import gf_matrix_from_column_transform
    for link_name, op in gripper_ops.items():
        op.Set(gf_matrix_from_column_transform(gripper_link_transform(link_name, angle_rad)))


def _setup_arm_ops(stage, root: str, suffix: str) -> dict:
    from pxr import UsdGeom
    ops = {}
    for link in LINK_NAMES:
        prim = stage.GetPrimAtPath(f"{root}/{link}")
        if not prim or not prim.IsValid():
            print(f"[DRAW] WARNING: {root}/{link} not found", flush=True)
            continue
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        ops[link] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    return ops


def _setup_gripper_ops(stage, gripper_root: str, suffix: str) -> dict:
    from pxr import UsdGeom
    from gripper import GRIPPER_LINK_JOINT_CHAINS
    ops = {}
    for link_name in GRIPPER_LINK_JOINT_CHAINS:
        prim = stage.GetPrimAtPath(f"{gripper_root}/{link_name}")
        if not prim or not prim.IsValid():
            continue
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        ops[link_name] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    return ops


def _solve_ik(link6_chain, lower, upper, base_world, link6_to_pad,
              target_xyz: np.ndarray, seeds: list):
    """Solve IK for target_xyz; return (q, pos_err_m). Falls back to seeds[0] on failure."""
    from planning import solve_pad_pose_ik
    try:
        q_sol, pos_err, _up_err, _fwd_err, _ok, _msg = solve_pad_pose_ik(
            link6_chain, lower, upper, base_world, link6_to_pad,
            target_xyz, UP_WORLD, FORWARD_WORLD, seeds,
        )
        return q_sol, float(pos_err)
    except Exception as e:
        print(f"[DRAW] IK warning: {e}", flush=True)
        return seeds[0].copy(), 999.0


# ── Path pre-computation ──────────────────────────────────────────────────────

def _compute_circle_path(link6_chain, lower, upper, base_world, link6_to_pad,
                         center: np.ndarray, init_seeds: list) -> list:
    """Compute CIRCLE_N joint configs tracing a circle around center (XZ plane)."""
    path = []
    # Start IK from all init_seeds; once first solution found, use it as warm start
    q_curr: np.ndarray | None = None
    thetas = np.linspace(0, 2 * math.pi, CIRCLE_N, endpoint=False)
    print(f"[DRAW] Computing circle: center={np.round(center,3)}, R={CIRCLE_RADIUS}m, N={CIRCLE_N}", flush=True)
    for k, theta in enumerate(thetas):
        target = center + CIRCLE_RADIUS * (math.cos(theta) * CIRCLE_AXIS1
                                           + math.sin(theta) * CIRCLE_AXIS2)
        seeds = ([q_curr] + init_seeds) if q_curr is not None else init_seeds
        q_sol, err = _solve_ik(link6_chain, lower, upper, base_world, link6_to_pad,
                                target, seeds)
        if err < 0.05:
            q_curr = q_sol
        else:
            q_sol = (q_curr if q_curr is not None else init_seeds[0]).copy()
            print(f"[DRAW] Circle IK failed at θ={math.degrees(theta):.0f}°, err={err:.4f} — reusing prev", flush=True)
        path.append(q_sol.copy())
        if (k + 1) % 8 == 0:
            print(f"[DRAW]   circle {k+1}/{CIRCLE_N} solved", flush=True)
    return path


def _compute_square_path(link6_chain, lower, upper, base_world, link6_to_pad,
                          center: np.ndarray, init_seeds: list) -> list:
    """Compute joint-space path tracing a square around center (XZ plane)."""
    # 4 corners in order: TR → TL → BL → BR → back to TR
    corners_offsets = [
        np.array([ SQUARE_HALF, 0.0,  SQUARE_HALF], dtype=float),  # TR
        np.array([-SQUARE_HALF, 0.0,  SQUARE_HALF], dtype=float),  # TL
        np.array([-SQUARE_HALF, 0.0, -SQUARE_HALF], dtype=float),  # BL
        np.array([ SQUARE_HALF, 0.0, -SQUARE_HALF], dtype=float),  # BR
    ]
    corner_labels = ["TR", "TL", "BL", "BR"]
    print(f"[DRAW] Computing square: center={np.round(center,3)}, half={SQUARE_HALF}m", flush=True)

    # Solve IK for each corner
    corner_q = []
    q_curr: np.ndarray | None = None
    for i, (offset, label) in enumerate(zip(corners_offsets, corner_labels)):
        target = center + offset
        seeds = ([q_curr] + init_seeds) if q_curr is not None else init_seeds
        q_sol, err = _solve_ik(link6_chain, lower, upper, base_world, link6_to_pad,
                                target, seeds)
        if err >= 0.05:
            print(f"[DRAW] Square corner {label} IK err={err:.4f} — reusing prev", flush=True)
            q_sol = (q_curr if q_curr is not None else init_seeds[0]).copy()
        else:
            q_curr = q_sol
        corner_q.append(q_sol.copy())
        print(f"[DRAW]   corner {label}: err={err:.4f}", flush=True)

    # Build full path by interpolating between consecutive corners (closed loop)
    path = []
    n = len(corner_q)
    for i in range(n):
        q_start = corner_q[i]
        q_end   = corner_q[(i + 1) % n]
        for step in range(SQUARE_STEPS_SIDE):
            t = step / SQUARE_STEPS_SIDE
            path.append(((1.0 - t) * q_start + t * q_end).copy())
    return path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    # Always open the scene — in --exec mode there is already an empty stage
    # so `ctx.get_stage()` is truthy even before the real scene is loaded.
    print(f"[DRAW] Opening scene: {DEFAULT_SCENE}", flush=True)
    ctx.open_stage(DEFAULT_SCENE)

    # Wait until the robot prim is present (up to ~300 frames ≈ 5 s)
    for i in range(300):
        app.update()
        s = ctx.get_stage()
        if s and s.GetPrimAtPath("/World/robot/jaka_minicobo_left").IsValid():
            print(f"[DRAW] Scene loaded in {i+1} frames", flush=True)
            break
    else:
        print("[DRAW] ERROR: scene did not load — robot prim missing!", flush=True)
        return

    run(app, ctx.get_stage())


def run(app, stage):
    import omni.timeline
    from pxr import Usd, UsdGeom
    from kinematics import (ARM_JOINTS, DEFAULT_ARM_URDF, GRIPPER_ROOT_SUFFIX,
                            chain_to_link, fk, load_joints)
    from planning import selected_pad_midpoint
    from ik_sanity import joint_limits as _ik_joint_limits

    print("[DRAW] Loading URDF kinematics…", flush=True)
    arm_joints = load_joints(Path(DEFAULT_ARM_URDF))
    chains = {n: chain_to_link(arm_joints, "Link_0", n) for n in LINK_NAMES}
    link6_chain = chains["Link_6"]
    lower, upper = _ik_joint_limits(link6_chain)  # takes chain, not path

    print("[DRAW] Reading arm world transforms…", flush=True)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    left_base,  left_pad_world,  left_l6_to_pad  = selected_pad_midpoint(stage, cache, "left")
    right_base, right_pad_world, right_l6_to_pad = selected_pad_midpoint(stage, cache, "right")

    # Drawing centres = actual pad positions at the scene's initial q=0,
    # lifted to at least DRAW_MIN_Z so shapes are above the table.
    def _centre(pad_world):
        c = pad_world[:3, 3].copy()
        if c[2] < DRAW_MIN_Z:
            c[2] = DRAW_MIN_Z
        return c

    left_center  = _centre(left_pad_world)
    right_center = _centre(right_pad_world)
    print(f"[DRAW] Left  arm draw centre:  {np.round(left_center, 3)}", flush=True)
    print(f"[DRAW] Right arm draw centre:  {np.round(right_center, 3)}", flush=True)

    # ── Pre-compute drawing paths (before Play) ────────────────────────────────
    print("[DRAW] Pre-computing paths via cuRobo IK (this takes ~10–30 s)…", flush=True)

    circle_path = _compute_circle_path(
        link6_chain, lower, upper, right_base, right_l6_to_pad,
        right_center, RIGHT_SEEDS,
    )
    square_path = _compute_square_path(
        link6_chain, lower, upper, left_base, left_l6_to_pad,
        left_center, LEFT_SEEDS,
    )
    print(f"[DRAW] Paths ready — circle: {len(circle_path)} pts, "
          f"square: {len(square_path)} pts", flush=True)

    # ── Setup arm FK xform ops ─────────────────────────────────────────────────
    left_ops  = _setup_arm_ops(stage, LEFT_ROOT,  "draw_fk")
    right_ops = _setup_arm_ops(stage, RIGHT_ROOT, "draw_fk")

    # Snap arms to first seed position before Play
    _set_arm(left_ops,  chains, LEFT_SEEDS[0])
    _set_arm(right_ops, chains, RIGHT_SEEDS[0])

    # ── Setup gripper xform ops ────────────────────────────────────────────────
    left_gripper_root  = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
    right_gripper_root = f"{RIGHT_ROOT}/{GRIPPER_ROOT_SUFFIX}"

    left_g_ops  = _setup_gripper_ops(stage, left_gripper_root,  "draw_gripper_fk")
    right_g_ops = _setup_gripper_ops(stage, right_gripper_root, "draw_gripper_fk")

    print(f"[DRAW] Left  gripper ops: {len(left_g_ops)} links", flush=True)
    print(f"[DRAW] Right gripper ops: {len(right_g_ops)} links", flush=True)

    _set_gripper(left_g_ops,  LEFT_GRIPPER_ANGLE)   # fully closed
    _set_gripper(right_g_ops, RIGHT_GRIPPER_ANGLE)  # fully open

    # ── Wait for Play ──────────────────────────────────────────────────────────
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    print("[DRAW] Ready — press Play to start the drawing loop.", flush=True)
    print(f"[DRAW] Left  arm:  SQUARE  (side={SQUARE_HALF*2*100:.0f} cm, "
          f"{len(square_path)} pts, {FRAMES_PER_STEP} fps/pt)", flush=True)
    print(f"[DRAW] Right arm:  CIRCLE  (R={CIRCLE_RADIUS*100:.0f} cm, "
          f"{len(circle_path)} pts, {FRAMES_PER_STEP} fps/pt)", flush=True)
    print(f"[DRAW] Left  gripper: CLOSED  ({LEFT_GRIPPER_ANGLE:.4f} rad → 35.4 mm)", flush=True)
    print(f"[DRAW] Right gripper: OPEN    ({RIGHT_GRIPPER_ANGLE:.4f} rad → 85.4 mm)", flush=True)

    while not timeline.is_playing():
        app.update()
        time.sleep(0.02)

    # Settle physics (60 frames = 1 s)
    for _ in range(60):
        app.update()

    # ── Animation loop ─────────────────────────────────────────────────────────
    left_step  = 0
    right_step = 0
    left_loops = 0
    right_loops = 0
    frame = 0

    print("[DRAW] Drawing loop started. Stop in the GUI to exit.", flush=True)
    while timeline.is_playing():
        # Advance one waypoint every FRAMES_PER_STEP frames
        if frame % FRAMES_PER_STEP == 0:
            li = left_step  % len(square_path)
            ri = right_step % len(circle_path)

            _set_arm(left_ops,  chains, square_path[li])
            _set_arm(right_ops, chains, circle_path[ri])

            # Grippers stay fixed each frame
            _set_gripper(left_g_ops,  LEFT_GRIPPER_ANGLE)
            _set_gripper(right_g_ops, RIGHT_GRIPPER_ANGLE)

            left_step  += 1
            right_step += 1

            # Log on each full loop completion
            if left_step % len(square_path) == 0:
                left_loops += 1
                print(f"[DRAW] Square loop {left_loops} complete", flush=True)
            if right_step % len(circle_path) == 0:
                right_loops += 1
                if right_loops % 4 == 0:
                    print(f"[DRAW] Circle loop {right_loops} complete", flush=True)

        app.update()
        frame += 1

    print("[DRAW] Stopped.", flush=True)


if __name__ == "__main__":
    main()
