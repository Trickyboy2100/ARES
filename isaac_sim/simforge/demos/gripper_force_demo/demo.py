#!/usr/bin/env python3
"""gripper_force_demo — Soft-contact gripper force visualisation in Isaac Sim.

Physics model (two-spring equilibrium)
───────────────────────────────────────
The EG2-4C2 drive spring and sphere contact spring share the joint:

    K_drive × (actual − drive) = K_sphere × (contact − actual)
    ──────────────────────────────────────────────────────────
    actual = (K_drive·drive + K_sphere·contact) / (K_drive + K_sphere)

  • drive ≥ contact_angle : no contact, actual = drive, F = 0
  • drive < contact_angle : joint is pulled open by the sphere spring;
    pad inner face slightly overlaps the sphere (visible penetration);
    gripping force rises as drive continues to close.

Contact geometry (EG2-4C2, from mesh vertex data)
──────────────────────────────────────────────────
  pad_separation(a) = 2 × |−0.04 + 0.0223·cos(a) − 0.035591·sin(a)|   [m]
  PAD_INNER_FACE = 0.0177 m  — depth of pad contact surface from prim origin
  contact_angle  = angle_for_pad_separation(2 × (R_sphere + PAD_INNER_FACE))

  Full-stroke range: 35.4 mm (closed, a=0) → 85.4 mm (open, a=35.65°)
  Max holdable sphere diameter: 50 mm  (= 85.4 − 35.4)

Launch
──────
  cd /path/to/PG-JY
  bash isaac_sim/simforge/demos/gripper_force_demo/launch.sh
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ── sys.path: allow imports from simforge/core ───────────────────────────────
_DEMO_DIR  = Path(__file__).resolve().parent
_SIMFORGE  = _DEMO_DIR.parents[1]          # simforge/
_CORE      = _SIMFORGE / "core"
for _p in (str(_SIMFORGE), str(_CORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Scene path: prefer scene.usd next to this file ───────────────────────────
DEFAULT_SCENE = str(_DEMO_DIR / "scene.usd")
if not Path(DEFAULT_SCENE).exists():
    try:
        import config as _cfg
        DEFAULT_SCENE = _cfg.SCENE_USD
    except Exception:
        DEFAULT_SCENE = str(Path.home() / "isaacsim/playground/2026061100_main.usd")

# ── URDF fallback ─────────────────────────────────────────────────────────────
try:
    import config as _cfg
    DEFAULT_ARM_URDF = str(_cfg.ARM_URDF)
except Exception:
    DEFAULT_ARM_URDF = ""
if not DEFAULT_ARM_URDF or not Path(DEFAULT_ARM_URDF).is_file():
    _fb = Path.home() / "Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo.urdf"
    if _fb.is_file():
        DEFAULT_ARM_URDF = str(_fb)

from kinematics import GRIPPER_ROOT_SUFFIX, ARM_JOINTS, chain_to_link, load_joints, fk
from gripper import angle_for_pad_separation_m, gripper_link_transform, setup_gripper_xform_ops, pad_separation_m
from planning import selected_pad_midpoint
from scene_utils import gf_matrix_from_column_transform

# ── Robot prim paths ──────────────────────────────────────────────────────────
LEFT_ROOT  = "/World/robot/jaka_minicobo_left"
RIGHT_ROOT = "/World/robot/jaka_minicobo_right"
LEFT_GR    = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
RIGHT_GR   = f"{RIGHT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]

# ── Gripper / contact parameters ─────────────────────────────────────────────
GRIPPER_OPEN_DEG        = 35.65   # full open → 85.25 mm pad-origin gap
PAD_INNER_FACE_M        = 0.0177  # pad mesh contact-surface depth from prim origin [m]
SPHERE_RADIUS_M         = 0.012   # 24 mm sphere; contact at ~50 % of stroke
GRIPPER_DRIVE_STIFFNESS = 3.0     # N·m/rad — motor drive spring constant
GRIPPER_DRIVE_MAX_FORCE = 10.0    # N·m     — motor torque limit
SPHERE_STIFFNESS        = 15.0    # N·m/rad — sphere contact spring (5× drive → ~2 mm penetration)
LEVER_ARM_M             = 0.055   # joint → pad contact point [m]

# ── Simulation timing ─────────────────────────────────────────────────────────
SETTLE_FRAMES         = 60    # 1 s  — hold open so physics settles
CLOSE_RAMP_FRAMES     = 150   # 2.5 s — ramp from open to fully closed
CONTACT_DETECT_MARGIN = 0.3   # °    — hysteresis before entering CONTACT phase

# ── UI ────────────────────────────────────────────────────────────────────────
HIST_LEN     = 350
UI_EVERY     = 3
FORCE_MAX_N  = 50.0
PENET_MAX_MM = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _contact_angle_deg() -> float:
    return math.degrees(
        angle_for_pad_separation_m(2.0 * (SPHERE_RADIUS_M + PAD_INNER_FACE_M))
    )


def _x_pad_abs(angle_rad: float) -> float:
    return abs(-0.04 + 0.0223 * math.cos(angle_rad) - 0.035591 * math.sin(angle_rad))


def _solve_contact(drive_deg: float, contact_deg: float) -> tuple[float, float, float]:
    """Two-spring equilibrium → (actual_deg, penetration_mm, force_N)."""
    if drive_deg >= contact_deg:
        return drive_deg, 0.0, 0.0
    K_d, K_s = GRIPPER_DRIVE_STIFFNESS, SPHERE_STIFFNESS
    drive_rad, contact_rad = math.radians(drive_deg), math.radians(contact_deg)
    actual_rad = (K_d * drive_rad + K_s * contact_rad) / (K_d + K_s)
    torque     = min(K_d * (actual_rad - drive_rad), GRIPPER_DRIVE_MAX_FORCE)
    force_N    = 2.0 * torque / LEVER_ARM_M
    inner_abs  = _x_pad_abs(actual_rad) - PAD_INNER_FACE_M
    penet_mm   = max(0.0, (SPHERE_RADIUS_M - inner_abs) * 1000.0)
    return math.degrees(actual_rad), penet_mm, force_N


# ─────────────────────────────────────────────────────────────────────────────
# Arm / gripper xform FK
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


def _set_gripper(ops, angle_deg: float):
    from pxr import Gf
    a = math.radians(angle_deg)
    for name, op in ops.items():
        T = gripper_link_transform(name, a)
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
    UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim()).CreateKinematicEnabledAttr(True)
    print(f"[DEMO] Sphere at {np.round(center, 3)}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# omni.ui panel
# ─────────────────────────────────────────────────────────────────────────────

def _bar_force(frac: float) -> int:
    frac = max(0.0, min(1.0, frac))
    return 0xFF000000 | (min(255, int((1 - frac) * 2 * 255)) << 8) | min(255, int(frac * 2 * 255))


def _bar_penet(frac: float) -> int:
    frac = max(0.0, min(1.0, frac))
    return 0xFF000000 | (int((1 - frac) * 255) << 16) | (200 << 8) | int(frac * 255)


class ForceMonitorUI:
    def __init__(self, contact_deg: float, stall_force: float):
        import omni.ui as ui
        self._ui = ui
        self._contact_deg = contact_deg
        self._stall_force  = stall_force
        self._l_hist = [0.0] * HIST_LEN
        self._r_hist = [0.0] * HIST_LEN
        self._state  = dict(l_deg=GRIPPER_OPEN_DEG, r_deg=GRIPPER_OPEN_DEG,
                            l_fn=0.0, r_fn=0.0, l_pen=0.0, r_pen=0.0, phase="SETTLE")
        self._tick   = 0
        self._win = ui.Window(
            "Gripper Force Monitor", width=520, height=510,
            flags=(ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE),
        )
        self._rebuild()

    def push(self, l_deg, r_deg, l_fn, r_fn, l_pen, r_pen, phase):
        self._l_hist = (self._l_hist + [l_fn])[-HIST_LEN:]
        self._r_hist = (self._r_hist + [r_fn])[-HIST_LEN:]
        self._state.update(l_deg=l_deg, r_deg=r_deg, l_fn=l_fn, r_fn=r_fn,
                           l_pen=l_pen, r_pen=r_pen, phase=phase)
        self._tick += 1
        if self._tick % UI_EVERY == 0:
            self._rebuild()

    def _rebuild(self):
        ui  = self._ui
        s   = self._state
        self._win.frame.clear()
        with self._win.frame:
            with ui.VStack(spacing=5):
                with ui.HStack(height=28):
                    ui.Label("GRIPPER FORCE MONITOR",
                             style={"font_size": 15, "color": 0xFFFFFFFF})
                    ui.Spacer()
                    col = {"SETTLE": 0xFF888888, "CLOSING": 0xFF00AAFF,
                           "CONTACT": 0xFF44FF44}.get(s["phase"], 0xFFAAAAAA)
                    ui.Label(f"● {s['phase']}", style={"font_size": 13, "color": col})
                ui.Separator(height=2)

                for label, hist, deg, fn, pen, line_col in [
                    ("LEFT  GRIPPER", self._l_hist, s["l_deg"], s["l_fn"], s["l_pen"], 0xFF22AAFF),
                    ("RIGHT GRIPPER", self._r_hist, s["r_deg"], s["r_fn"], s["r_pen"], 0xFFFF7722),
                ]:
                    in_contact = (s["phase"] == "CONTACT")
                    with ui.HStack(height=20):
                        ui.Label(f"{label}   {deg:.1f}°  │  F = {fn:.1f} N",
                                 style={"font_size": 13})
                        if in_contact:
                            ui.Label(f"  ↓{pen:.2f}mm",
                                     style={"font_size": 12,
                                            "color": 0xFFFFDD44 if pen > 0.5 else 0xFF44FF88})
                    with ui.ZStack(height=14):
                        ui.Rectangle(style={"background_color": 0xFF222222, "border_radius": 3})
                        pct_f = min(100.0, fn / FORCE_MAX_N * 100.0)
                        if pct_f > 0:
                            with ui.HStack():
                                ui.Rectangle(width=ui.Percent(pct_f),
                                             style={"background_color": _bar_force(fn / FORCE_MAX_N),
                                                    "border_radius": 3})
                                ui.Spacer()
                    if in_contact and pen > 0:
                        with ui.ZStack(height=8):
                            ui.Rectangle(style={"background_color": 0xFF111111, "border_radius": 2})
                            pct_p = min(100.0, pen / PENET_MAX_MM * 100.0)
                            with ui.HStack():
                                ui.Rectangle(width=ui.Percent(pct_p),
                                             style={"background_color": _bar_penet(pen / PENET_MAX_MM),
                                                    "border_radius": 2})
                                ui.Spacer()
                    else:
                        ui.Spacer(height=8)
                    ui.Plot(ui.Type.LINE, 0.0, FORCE_MAX_N, *hist, height=95,
                            style={"color": line_col, "background_color": 0xFF05050F})
                    ui.Spacer(height=3)

                ui.Separator(height=2)
                ui.Label(
                    f"sphere ø={SPHERE_RADIUS_M*2000:.0f}mm  │  contact≈{self._contact_deg:.1f}°  │  "
                    f"K_drive={GRIPPER_DRIVE_STIFFNESS:.0f}  K_sphere={SPHERE_STIFFNESS:.0f} N·m/rad  │  "
                    f"stall F≈{self._stall_force:.1f} N",
                    style={"font_size": 9, "color": 0xFF666666}, height=14,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

def _run(app, stage):
    import omni.timeline
    from pxr import Usd, UsdGeom

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.set_end_time(9999.0)
    timeline.set_looping(False)

    arm_jts = load_joints(Path(DEFAULT_ARM_URDF))
    chains  = {n: chain_to_link(arm_jts, "Link_0", n) for n in LINK_NAMES}
    q_zero  = {j: 0.0 for j in ARM_JOINTS}

    contact_deg = _contact_angle_deg()
    _, max_penet, stall_force = _solve_contact(0.0, contact_deg)
    print(f"[DEMO] sphere ø={SPHERE_RADIUS_M*2000:.0f}mm  "
          f"contact={contact_deg:.2f}°  stallF={stall_force:.1f}N  "
          f"maxPenet={max_penet:.2f}mm", flush=True)

    l_ops    = _setup_arm_ops(stage, LEFT_ROOT,  "fd_arm")
    r_ops    = _setup_arm_ops(stage, RIGHT_ROOT, "fd_arm")
    l_gr_ops = setup_gripper_xform_ops(stage, LEFT_GR,  "fd_gr")
    r_gr_ops = setup_gripper_xform_ops(stage, RIGHT_GR, "fd_gr")
    _set_arm_q(l_ops, chains, q_zero)
    _set_arm_q(r_ops, chains, q_zero)
    _set_gripper(l_gr_ops, GRIPPER_OPEN_DEG)
    _set_gripper(r_gr_ops, GRIPPER_OPEN_DEG)
    app.update()

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    try:
        _, l_pad_w, _ = selected_pad_midpoint(stage, cache, "left")
        _, r_pad_w, _ = selected_pad_midpoint(stage, cache, "right")
        l_center = l_pad_w[:3, 3].copy()
        r_center = r_pad_w[:3, 3].copy()
    except RuntimeError:
        l_center = np.array([0.85,  0.55, 1.19])
        r_center = np.array([-0.85, 0.55, 1.19])

    _spawn_sphere(stage, "/World/FD_SphereL", l_center)
    _spawn_sphere(stage, "/World/FD_SphereR", r_center)

    ui_mon = ForceMonitorUI(contact_deg, stall_force)
    app.update()

    print("[DEMO] Auto-starting…", flush=True)
    timeline.play()
    app.update()

    frame     = 0
    phase     = "SETTLE"
    drive_deg = GRIPPER_OPEN_DEG
    actual_deg = GRIPPER_OPEN_DEG

    while app.is_running():
        if phase == "SETTLE" and frame >= SETTLE_FRAMES:
            phase = "CLOSING"
            print(f"[DEMO] → CLOSING (fr {frame})", flush=True)

        if phase in ("CLOSING", "CONTACT"):
            t = min(1.0, max(0, frame - SETTLE_FRAMES) / CLOSE_RAMP_FRAMES)
            drive_deg = GRIPPER_OPEN_DEG * (1.0 - t)
            actual_deg, penet_mm, fn = _solve_contact(drive_deg, contact_deg)
            if phase == "CLOSING" and drive_deg < contact_deg - CONTACT_DETECT_MARGIN:
                phase = "CONTACT"
                print(f"[DEMO] → CONTACT  drive={drive_deg:.2f}°  "
                      f"actual={actual_deg:.2f}°  pen={penet_mm:.2f}mm  F={fn:.1f}N  "
                      f"(fr {frame})", flush=True)
        else:
            actual_deg, penet_mm, fn = GRIPPER_OPEN_DEG, 0.0, 0.0

        _set_arm_q(l_ops, chains, q_zero)
        _set_arm_q(r_ops, chains, q_zero)
        _set_gripper(l_gr_ops, actual_deg)
        _set_gripper(r_gr_ops, actual_deg)
        ui_mon.push(actual_deg, actual_deg, fn, fn, penet_mm, penet_mm, phase)

        log_every = 20 if phase == "CLOSING" else 60 if phase == "SETTLE" else 30
        if frame % log_every == 0:
            print(f"[DEMO] fr={frame:5d}  {phase:8s}  "
                  f"drive={drive_deg:5.1f}°  actual={actual_deg:5.1f}°  "
                  f"pen={penet_mm:5.2f}mm  F={fn:5.1f}N", flush=True)

        app.update()
        frame += 1

    print("[DEMO] Stopped.", flush=True)


def main():
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    print(f"[DEMO] Scene: {DEFAULT_SCENE}", flush=True)
    ctx.open_stage(DEFAULT_SCENE)

    for i in range(200):
        app.update()
        s = ctx.get_stage()
        if s and s.GetPrimAtPath(LEFT_GR + "/left_pad").IsValid():
            print(f"[DEMO] Ready ({i+1} frames)", flush=True)
            break
    else:
        print("[DEMO] ERROR: EG2 pads not found after 200 frames", flush=True)
        return

    _run(app, ctx.get_stage())


if __name__ == "__main__":
    main()
