#!/usr/bin/env python3
"""gripper_force_demo.py — Gripper soft-contact force demo with live omni.ui chart.

Physics model
─────────────
  Two-spring equilibrium at contact:

    K_drive × (actual − drive) = K_sphere × (contact − actual)
    ──────────────────────────────────────────────────────────
    actual = (K_drive·drive + K_sphere·contact) / (K_drive + K_sphere)

  • Before contact (drive ≥ contact): actual = drive, F = 0
  • During contact: actual is pulled toward contact_angle by sphere spring;
    gripper visually penetrates the sphere slightly; force rises with drive error.
  • Max penetration ≈ K_drive/(K_drive+K_sphere) × contact_angle_rad → ~2 mm

Contact geometry (EG2-4C2)
──────────────────────────
  pad_separation_m(a) = 2 × |−0.04 + 0.0223·cos(a) − 0.035591·sin(a)|
  PAD_INNER_FACE_M = 0.0177 m  (mesh X_max from EG2_4C2_base.usd vertex data)
  contact_angle = angle_for_pad_separation(2 × (SPHERE_RADIUS + PAD_INNER_FACE_M))

Architecture (xform FK — scene does not support EG2 physics drive)
──────────────────────────────────────────────────────────────────
  • ARM links (Link_1-6): xform FK at q=0 every frame
  • GRIPPER finger links: xform FK at actual_deg every frame
  • Force / penetration: soft two-spring model (no PhysX contact needed)

Launch:
  CUSPARSE=~/isaacsim/extscache/omni.isaac.lula-*/lib/libcusparse.so.12
  LD_PRELOAD=$CUSPARSE ~/isaacsim/isaac-sim.sh --exec \\
    isaac_sim/simforge/demos/gripper_force_demo.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# ── sys.path ──────────────────────────────────────────────────────────────────
_SIMFORGE = str(Path(__file__).resolve().parents[1])
_CORE     = str(Path(__file__).resolve().parents[1] / "core")
for _p in (_SIMFORGE, _CORE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import config as _cfg
    DEFAULT_SCENE    = _cfg.SCENE_USD
    DEFAULT_ARM_URDF = str(_cfg.ARM_URDF)
except Exception:
    DEFAULT_SCENE    = "/home/andyee/isaacsim/playground/2026061100_main.usd"
    DEFAULT_ARM_URDF = ""

if not DEFAULT_ARM_URDF or not Path(DEFAULT_ARM_URDF).is_file():
    _fb = Path.home() / "Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo.urdf"
    if _fb.is_file():
        DEFAULT_ARM_URDF = str(_fb)

from kinematics import GRIPPER_ROOT_SUFFIX, ARM_JOINTS, chain_to_link, load_joints, fk
from gripper import (
    angle_for_pad_separation_m,
    gripper_link_transform,
    setup_gripper_xform_ops,
    pad_separation_m,
)
from planning import selected_pad_midpoint
from scene_utils import gf_matrix_from_column_transform

# ── Robot prim roots ──────────────────────────────────────────────────────────
LEFT_ROOT  = "/World/robot/jaka_minicobo_left"
RIGHT_ROOT = "/World/robot/jaka_minicobo_right"
LEFT_GR    = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
RIGHT_GR   = f"{RIGHT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]

# ── Gripper & contact parameters ──────────────────────────────────────────────
GRIPPER_DRIVE_STIFFNESS = 3.0    # N·m/rad — motor drive spring
GRIPPER_DRIVE_MAX_FORCE = 10.0   # N·m     — motor torque limit
GRIPPER_OPEN_DEG        = 35.65  # full open → 85.25 mm pad-origin gap
LEVER_ARM_M             = 0.055  # joint → pad contact point [m]

# Pad geometry (from EG2_4C2_base.usd mesh vertex bounds)
PAD_INNER_FACE_M = 0.0177        # pad mesh X_max in pad-local frame [m]
                                  # = depth of contact surface from prim origin

# Sphere parameters
SPHERE_RADIUS_M  = 0.012         # 24 mm sphere; contact at ~50% of stroke

# Sphere contact spring — controls how much gripper penetrates
# Penetration at full-close ≈ K_drive/(K_drive+K_sphere) × contact_angle_rad
# K_sphere = 15 → ~3° angular penetration → ~2 mm physical penetration
SPHERE_STIFFNESS = 15.0          # N·m/rad

# Simulation timing
SETTLE_FRAMES         = 60       # 1 s  — arm snaps to q=0
CLOSE_RAMP_FRAMES     = 150      # 2.5 s — ramp from open to closed
CONTACT_DETECT_MARGIN = 0.3      # °     — enter CONTACT phase this much past contact

# UI
HIST_LEN    = 350
UI_EVERY    = 3
FORCE_MAX_N = 50.0
PENET_MAX_MM = 5.0               # full-scale penetration gauge


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _contact_angle_deg() -> float:
    """Joint angle where pad inner face just touches sphere surface (zero penetration)."""
    return math.degrees(
        angle_for_pad_separation_m(2.0 * (SPHERE_RADIUS_M + PAD_INNER_FACE_M))
    )


def _x_pad_abs(angle_rad: float) -> float:
    """Absolute EG2-X position of pad prim origin (= half of pad_separation_m)."""
    return abs(-0.04 + 0.0223 * math.cos(angle_rad) - 0.035591 * math.sin(angle_rad))


def _solve_contact(drive_deg: float, contact_deg: float) -> tuple[float, float, float]:
    """Two-spring soft-contact equilibrium.

    Returns (actual_deg, penetration_mm, force_N).

    actual_deg      — where the gripper is set (may be < contact_deg, causing
                      slight visual penetration into the sphere)
    penetration_mm  — depth of pad inner face inside sphere surface [mm]
    force_N         — net gripping force on sphere [N]
    """
    if drive_deg >= contact_deg:
        return drive_deg, 0.0, 0.0

    drive_rad   = math.radians(drive_deg)
    contact_rad = math.radians(contact_deg)
    K_d = GRIPPER_DRIVE_STIFFNESS
    K_s = SPHERE_STIFFNESS

    # Equilibrium: K_d*(actual-drive) = K_s*(contact-actual)
    actual_rad = (K_d * drive_rad + K_s * contact_rad) / (K_d + K_s)
    actual_deg = math.degrees(actual_rad)

    # Gripper torque (capped at motor limit)
    torque  = min(K_d * (actual_rad - drive_rad), GRIPPER_DRIVE_MAX_FORCE)
    force_N = 2.0 * torque / LEVER_ARM_M

    # Physical penetration: how far pad inner face has entered sphere
    inner_face_abs = _x_pad_abs(actual_rad) - PAD_INNER_FACE_M
    penetration_mm = max(0.0, (SPHERE_RADIUS_M - inner_face_abs) * 1000.0)

    return actual_deg, penetration_mm, force_N


# ─────────────────────────────────────────────────────────────────────────────
# Arm xform FK
# ─────────────────────────────────────────────────────────────────────────────

def _setup_arm_ops(stage, root, suffix):
    from pxr import UsdGeom
    ops = {}
    for link in LINK_NAMES:
        prim = stage.GetPrimAtPath(f"{root}/{link}")
        if prim and prim.IsValid():
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            ops[link] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    return ops


def _set_arm_q(ops, chains, q_map):
    for link, op in ops.items():
        op.Set(gf_matrix_from_column_transform(fk(chains[link], q_map)))


# ─────────────────────────────────────────────────────────────────────────────
# Gripper xform FK
# ─────────────────────────────────────────────────────────────────────────────

def _set_gripper_angle(ops, angle_deg: float):
    from pxr import Gf
    angle_rad = math.radians(angle_deg)
    for link_name, op in ops.items():
        T = gripper_link_transform(link_name, angle_rad)
        op.Set(Gf.Matrix4d(*sum(np.asarray(T, dtype=float).T.tolist(), [])))


# ─────────────────────────────────────────────────────────────────────────────
# Sphere
# ─────────────────────────────────────────────────────────────────────────────

def _spawn_sphere(stage, path, center):
    from pxr import UsdGeom, UsdPhysics, Gf
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(SPHERE_RADIUS_M)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.55, 0.1)])
    xf = UsdGeom.Xformable(sphere.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*center.tolist()))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    print(f"[FORCE] Sphere at {np.round(center, 3)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# omni.ui panel
# ─────────────────────────────────────────────────────────────────────────────

def _bar_color_force(frac: float) -> int:
    """Green → yellow → red as frac goes 0 → 1 (ABGR)."""
    frac = max(0.0, min(1.0, frac))
    r = min(255, int(frac * 2 * 255))
    g = min(255, int((1.0 - frac) * 2 * 255))
    return 0xFF000000 | (g << 8) | r


def _bar_color_penet(frac: float) -> int:
    """Cyan → yellow — penetration indicator (ABGR)."""
    frac = max(0.0, min(1.0, frac))
    r = int(frac * 255)
    g = 200
    b = int((1.0 - frac) * 255)
    return 0xFF000000 | (b << 16) | (g << 8) | r


class ForceMonitorUI:
    def __init__(self, contact_deg: float, stall_force: float):
        import omni.ui as ui
        self._ui = ui
        self._contact_deg = contact_deg
        self._stall_force  = stall_force
        self._l_hist = [0.0] * HIST_LEN
        self._r_hist = [0.0] * HIST_LEN
        self._l_deg  = GRIPPER_OPEN_DEG
        self._r_deg  = GRIPPER_OPEN_DEG
        self._l_fn   = 0.0
        self._r_fn   = 0.0
        self._l_pen  = 0.0   # penetration [mm]
        self._r_pen  = 0.0
        self._phase  = "SETTLE"
        self._tick   = 0
        self._win = ui.Window(
            "Gripper Force Monitor", width=520, height=510,
            flags=(ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE),
        )
        self._rebuild()

    def push(self, l_deg, r_deg, l_fn, r_fn, l_pen, r_pen, phase):
        self._l_hist.append(l_fn)
        self._l_hist = self._l_hist[-HIST_LEN:]
        self._r_hist.append(r_fn)
        self._r_hist = self._r_hist[-HIST_LEN:]
        self._l_deg, self._r_deg = l_deg, r_deg
        self._l_fn,  self._r_fn  = l_fn,  r_fn
        self._l_pen, self._r_pen = l_pen, r_pen
        self._phase = phase
        self._tick += 1
        if self._tick % UI_EVERY == 0:
            self._rebuild()

    def _rebuild(self):
        ui = self._ui
        self._win.frame.clear()
        with self._win.frame:
            with ui.VStack(spacing=5):
                # ── Title & phase ─────────────────────────────────────────
                with ui.HStack(height=28):
                    ui.Label("GRIPPER FORCE MONITOR",
                             style={"font_size": 15, "color": 0xFFFFFFFF})
                    ui.Spacer()
                    col = {"SETTLE":  0xFF888888,
                           "CLOSING": 0xFF00AAFF,
                           "CONTACT": 0xFF44FF44}.get(self._phase, 0xFFAAAAAA)
                    ui.Label(f"● {self._phase}",
                             style={"font_size": 13, "color": col})
                ui.Separator(height=2)

                # ── Per-gripper panels ────────────────────────────────────
                for label, hist, deg, fn, pen, line_col in [
                    ("LEFT  GRIPPER", self._l_hist,
                     self._l_deg, self._l_fn, self._l_pen, 0xFF22AAFF),
                    ("RIGHT GRIPPER", self._r_hist,
                     self._r_deg, self._r_fn, self._r_pen, 0xFFFF7722),
                ]:
                    in_contact = (self._phase == "CONTACT")

                    # Angle + force label
                    with ui.HStack(height=20):
                        ui.Label(
                            f"{label}   {deg:.1f}°  │  F = {fn:.1f} N",
                            style={"font_size": 13})
                        if in_contact:
                            ui.Label(
                                f"  ↓{pen:.2f}mm",
                                style={"font_size": 12,
                                       "color": 0xFFFFDD44 if pen > 0.5 else 0xFF44FF88})

                    # Force bar
                    with ui.ZStack(height=14):
                        ui.Rectangle(style={"background_color": 0xFF222222,
                                            "border_radius": 3})
                        pct_f = min(100.0, fn / FORCE_MAX_N * 100.0)
                        if pct_f > 0:
                            with ui.HStack():
                                ui.Rectangle(
                                    width=ui.Percent(pct_f),
                                    style={"background_color": _bar_color_force(fn / FORCE_MAX_N),
                                           "border_radius": 3})
                                ui.Spacer()

                    # Penetration bar (shown when in contact)
                    if in_contact and pen > 0:
                        with ui.ZStack(height=8):
                            ui.Rectangle(style={"background_color": 0xFF111111,
                                                "border_radius": 2})
                            pct_p = min(100.0, pen / PENET_MAX_MM * 100.0)
                            with ui.HStack():
                                ui.Rectangle(
                                    width=ui.Percent(pct_p),
                                    style={"background_color": _bar_color_penet(pen / PENET_MAX_MM),
                                           "border_radius": 2})
                                ui.Spacer()
                    else:
                        ui.Spacer(height=8)

                    # Force history chart
                    ui.Plot(
                        ui.Type.LINE, 0.0, FORCE_MAX_N, *hist,
                        height=95,
                        style={"color": line_col,
                               "background_color": 0xFF05050F},
                    )
                    ui.Spacer(height=3)

                # ── Info bar ──────────────────────────────────────────────
                ui.Separator(height=2)
                ui.Label(
                    f"sphere ø={SPHERE_RADIUS_M * 2000:.0f}mm  │  "
                    f"contact≈{self._contact_deg:.1f}°  │  "
                    f"K_drive={GRIPPER_DRIVE_STIFFNESS:.0f}  K_sphere={SPHERE_STIFFNESS:.0f} N·m/rad  │  "
                    f"max F≈{self._stall_force:.1f}N  │  soft-contact xform FK",
                    style={"font_size": 9, "color": 0xFF666666}, height=14,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import omni.kit.app, omni.usd, omni.timeline
    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    print(f"[FORCE] Opening: {DEFAULT_SCENE}", flush=True)
    ctx.open_stage(DEFAULT_SCENE)

    # Wait for scene geometry to be available (mirror ear_grasp_lift pattern)
    for i in range(200):
        app.update()
        s = ctx.get_stage()
        if s and s.GetPrimAtPath(LEFT_GR + "/left_pad").IsValid():
            print(f"[FORCE] Scene ready ({i+1} frames)", flush=True)
            break
    else:
        print("[FORCE] ERROR: EG2 pads not found after 200 frames", flush=True)
        return

    _run(app, ctx.get_stage())


def _run(app, stage):
    import omni.timeline
    from pxr import Usd, UsdGeom

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.set_end_time(9999.0)    # prevent scene's built-in end-time from stopping the demo
    timeline.set_looping(False)

    arm_jts = load_joints(Path(DEFAULT_ARM_URDF))
    chains  = {n: chain_to_link(arm_jts, "Link_0", n) for n in LINK_NAMES}
    q_zero  = {j: 0.0 for j in ARM_JOINTS}

    # Pre-compute contact geometry
    contact_deg = _contact_angle_deg()
    _, max_penet, stall_force = _solve_contact(0.0, contact_deg)

    print(f"[FORCE] sphere ø={SPHERE_RADIUS_M*2000:.0f}mm  "
          f"contact={contact_deg:.2f}°  "
          f"K_drive={GRIPPER_DRIVE_STIFFNESS}  K_sphere={SPHERE_STIFFNESS}  "
          f"stallF={stall_force:.1f}N  maxPenet={max_penet:.2f}mm",
          flush=True)

    # ── Arm + gripper xform FK setup (before play) ───────────────────────────
    l_ops = _setup_arm_ops(stage, LEFT_ROOT,  "force_demo_arm")
    r_ops = _setup_arm_ops(stage, RIGHT_ROOT, "force_demo_arm")
    _set_arm_q(l_ops, chains, q_zero)
    _set_arm_q(r_ops, chains, q_zero)

    l_gr_ops = setup_gripper_xform_ops(stage, LEFT_GR,  "force_demo_gr")
    r_gr_ops = setup_gripper_xform_ops(stage, RIGHT_GR, "force_demo_gr")
    _set_gripper_angle(l_gr_ops, GRIPPER_OPEN_DEG)
    _set_gripper_angle(r_gr_ops, GRIPPER_OPEN_DEG)
    app.update()

    # Spawn spheres at pad midpoints (arm at q=0, gripper fully open)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    try:
        _, l_pad_w, _ = selected_pad_midpoint(stage, cache, "left")
        _, r_pad_w, _ = selected_pad_midpoint(stage, cache, "right")
        l_center = l_pad_w[:3, 3].copy()
        r_center = r_pad_w[:3, 3].copy()
    except RuntimeError as e:
        print(f"[FORCE] Pad midpoint error: {e} — using fallback positions", flush=True)
        l_center = np.array([0.85,  0.55, 1.19])
        r_center = np.array([-0.85, 0.55, 1.19])

    print(f"[FORCE] Left pad:  {np.round(l_center, 3)}", flush=True)
    print(f"[FORCE] Right pad: {np.round(r_center, 3)}", flush=True)
    _spawn_sphere(stage, "/World/ForceDemo_SphereL", l_center)
    _spawn_sphere(stage, "/World/ForceDemo_SphereR", r_center)

    # Build UI and auto-start physics — no manual Play press needed
    ui_mon = ForceMonitorUI(contact_deg, stall_force)
    app.update()

    print("[FORCE] Auto-starting simulation…", flush=True)
    timeline.play()
    app.update()

    if not timeline.is_playing():
        print("[FORCE] WARNING: timeline did not start — check scene settings", flush=True)

    print("[FORCE] Simulation running.", flush=True)

    frame      = 0
    phase      = "SETTLE"
    drive_deg  = GRIPPER_OPEN_DEG
    actual_deg = GRIPPER_OPEN_DEG
    penet_mm   = 0.0
    fn         = 0.0

    while app.is_running():

        # ── Phase state machine ───────────────────────────────────────────────
        if phase == "SETTLE" and frame >= SETTLE_FRAMES:
            phase = "CLOSING"
            print(f"[FORCE] Phase → CLOSING (fr {frame})", flush=True)

        if phase in ("CLOSING", "CONTACT"):
            closing_frame = max(0, frame - SETTLE_FRAMES)
            t = min(1.0, closing_frame / CLOSE_RAMP_FRAMES)
            drive_deg = GRIPPER_OPEN_DEG * (1.0 - t)

            # Soft-contact equilibrium
            actual_deg, penet_mm, fn = _solve_contact(drive_deg, contact_deg)

            if phase == "CLOSING" and drive_deg < contact_deg - CONTACT_DETECT_MARGIN:
                phase = "CONTACT"
                print(f"[FORCE] Phase → CONTACT  drive={drive_deg:.2f}°  "
                      f"actual={actual_deg:.2f}°  pen={penet_mm:.2f}mm  "
                      f"F={fn:.1f}N  (fr {frame})", flush=True)
        else:
            actual_deg = GRIPPER_OPEN_DEG
            penet_mm   = 0.0
            fn         = 0.0

        # ── Set kinematics ────────────────────────────────────────────────────
        _set_arm_q(l_ops, chains, q_zero)
        _set_arm_q(r_ops, chains, q_zero)
        _set_gripper_angle(l_gr_ops, actual_deg)
        _set_gripper_angle(r_gr_ops, actual_deg)

        # ── UI update ─────────────────────────────────────────────────────────
        ui_mon.push(actual_deg, actual_deg, fn, fn, penet_mm, penet_mm, phase)

        if frame % (20 if phase == "CLOSING" else 60 if phase == "SETTLE" else 30) == 0:
            print(f"[FORCE] fr={frame:5d}  {phase:8s}  "
                  f"drive={drive_deg:5.1f}°  actual={actual_deg:5.1f}°  "
                  f"pen={penet_mm:5.2f}mm  F={fn:5.1f}N",
                  flush=True)

        app.update()
        frame += 1

    print("[FORCE] Stopped.", flush=True)


if __name__ == "__main__":
    main()
