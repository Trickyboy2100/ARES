#!/usr/bin/env python3
"""tray_grasp_cycle — ear-grasp pick-lift-place loop.

Sphere-demo approach: pure xform FK, analytical Y-contact spring model,
kinematic tray lift via direct translate write. No PhysX pads, no FixedJoint.

Contact model (two springs in series, Y-axis)
─────────────────────────────────────────────
  contact_y = ear_face_y + PAD_FACE_DEPTH_M
  When drive_y < contact_y (pad past ear face):
    actual_y = drive_y + K_ear/(K_arm+K_ear) * (contact_y − drive_y)
    F_normal_per_pad = K_arm * (actual_y − drive_y)
    F_friction_total = 2 * μ * F_normal_per_pad

Approach direction: world +Y → −Y (parallel to Y axis through tray_grasp_init_point).
Force-stop: arm halts approach when friction_n ≥ GRIP_FORCE_STOP_N.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# ── sys.path ──────────────────────────────────────────────────────────────────
_DEMO_DIR = Path(__file__).resolve().parent
_SIMFORGE  = _DEMO_DIR.parents[1]
_CORE      = _SIMFORGE / "core"
for _p in (str(_SIMFORGE), str(_CORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Scene / URDF paths ────────────────────────────────────────────────────────
try:
    import config as _cfg
    DEFAULT_SCENE    = _cfg.SCENE_USD
    DEFAULT_ARM_URDF = str(_cfg.ARM_URDF)
except Exception:
    DEFAULT_SCENE    = str(Path.home() / "isaacsim/playground/2026061100_main.usd")
    DEFAULT_ARM_URDF = str(
        Path.home() / "Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo.urdf"
    )

from kinematics import (
    GRIPPER_ROOT_SUFFIX, ARM_JOINTS, chain_to_link, load_joints, fk, get_world_pose,
)
from gripper import (
    GRIPPER_OPEN_ANGLE_RAD,
    gripper_link_transform,
    pad_separation_m,
    setup_gripper_xform_ops,
)
from planning import (
    selected_pad_midpoint,
    solve_pad_pose_ik,
    fallback_path,
    pad_world_transform,
)
from ik_sanity import joint_limits
from scene_utils import gf_matrix_from_column_transform

# ── Robot prim paths ──────────────────────────────────────────────────────────
LEFT_ROOT  = "/World/robot/jaka_minicobo_left"
RIGHT_ROOT = "/World/robot/jaka_minicobo_right"
LEFT_GR    = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
RIGHT_GR   = f"{RIGHT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]
TRAY_PATH        = "/World/Tray"
GRASP_PRIM_PATH  = "/World/Tray/tray_grasp_point"   # user-defined Xform in USD

# Calibrated fallback (tray_grasp_init_point, measured in settled scene):
#   X=0.0721  Y=0.3942 (ear face world Y)  Z=1.0116 (ear tab world Z)
# Used only when USD prim reading returns zero or invalid.
TRAY_GRASP_INIT = np.array([0.0721, 0.3942, 1.0116])

# ── Grasp orientation (EG2-4C2 on left arm) ──────────────────────────────────
# EG2-Y = world +X (jaw height axis)
# EG2-Z = world −Y (approach direction: gripper moves from +Y toward −Y)
TARGET_UP_WORLD      = np.array([1.0,  0.0,  0.0])
TARGET_FORWARD_WORLD = np.array([0.0, -1.0,  0.0])

# ── Approach geometry (all along world-Y line through ear position) ────────────
PRE_Y_OFFSET  = 0.125   # m — pre-grasp, clear of ear
NEAR_Y_OFFSET = 0.040   # m — intermediate, still no contact
PICK_Y_OFFSET = 0.012   # m — maximum drive depth (force-stop usually triggers sooner)
LIFT_Z        = 0.300   # m — lift height

# ── Contact force model ───────────────────────────────────────────────────────
# Pad face is PAD_FACE_DEPTH_M ahead of pad_mid in approach direction (world −Y)
# Contact when pad_mid_y < ear_face_y + PAD_FACE_DEPTH_M
PAD_FACE_DEPTH_M  = 0.028   # m
K_ARM_Y           = 300.0   # N/m — arm approach spring
K_EAR_Y           = 3000.0  # N/m — ear surface spring
MU_STATIC         = 1.5     # rubber-on-metal static friction
GRIP_FORCE_STOP_N = 3.0     # N — approach halts when total friction ≥ this

# ── Timing ────────────────────────────────────────────────────────────────────
SETTLE_FRAMES = 120
MOTION_FRAMES = 90
FINE_FRAMES   = 60   # near → pick (shorter, contact zone)
GRIP_FRAMES   = 45
HOLD_FRAMES   = 60
PAUSE_FRAMES  = 90

# ── UI ────────────────────────────────────────────────────────────────────────
HIST_LEN     = 350
UI_EVERY     = 3
FORCE_MAX_N  = 20.0
PRESS_MAX_MM = 5.0
LIFT_MAX_M   = LIFT_Z + 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Contact model
# ─────────────────────────────────────────────────────────────────────────────

def _solve_ear_contact(drive_y: float, contact_y: float) -> tuple:
    """Two-spring equilibrium. Returns (actual_y, press_mm, friction_N)."""
    if drive_y >= contact_y:
        return drive_y, 0.0, 0.0
    press_depth = contact_y - drive_y
    K_d, K_s = K_ARM_Y, K_EAR_Y
    actual_y  = drive_y + K_s * press_depth / (K_d + K_s)
    press_mm  = (contact_y - actual_y) * 1000.0
    F_N       = K_d * (actual_y - drive_y)
    friction_n = 2.0 * MU_STATIC * F_N
    return actual_y, press_mm, friction_n


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


def _set_arm_q(ops, chains, q: np.ndarray):
    q_map = dict(zip(ARM_JOINTS, q.tolist()))
    for link, op in ops.items():
        op.Set(gf_matrix_from_column_transform(fk(chains[link], q_map)))


def _set_gripper(gr_ops, angle_rad: float):
    for name, op in gr_ops.items():
        T = gripper_link_transform(name, angle_rad)
        op.Set(gf_matrix_from_column_transform(T))


# ─────────────────────────────────────────────────────────────────────────────
# Tray kinematic control
# ─────────────────────────────────────────────────────────────────────────────

def _set_tray_kinematic(stage, enabled: bool):
    prim = stage.GetPrimAtPath(TRAY_PATH)
    if prim and prim.IsValid():
        prim.GetAttribute("physics:kinematicEnabled").Set(enabled)


def _get_tray_translate(stage) -> np.ndarray:
    v = stage.GetPrimAtPath(TRAY_PATH).GetAttribute("xformOp:translate").Get()
    return np.array([float(v[0]), float(v[1]), float(v[2])])


def _set_tray_z(stage, tray_xyz0: np.ndarray, delta_z: float):
    from pxr import Gf
    prim = stage.GetPrimAtPath(TRAY_PATH)
    prim.GetAttribute("xformOp:translate").Set(
        Gf.Vec3d(float(tray_xyz0[0]), float(tray_xyz0[1]), float(tray_xyz0[2]) + delta_z)
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI — Dual arm monitor
# ─────────────────────────────────────────────────────────────────────────────

def _bar_force(frac: float) -> int:
    frac = max(0.0, min(1.0, frac))
    r = min(255, int(frac * 2 * 255))
    g = min(255, int((1 - frac) * 2 * 255))
    return 0xFF000000 | r | (g << 8)


def _bar_press(frac: float) -> int:
    frac = max(0.0, min(1.0, frac))
    return 0xFF000000 | (int((1 - frac) * 255) << 16) | (200 << 8) | int(frac * 255)


class DualArmMonitorUI:
    def __init__(self, r_pad_xyz: np.ndarray):
        import omni.ui as ui
        self._ui       = ui
        self._hist_f   = [0.0] * HIST_LEN   # left arm friction force
        self._hist_py  = [0.0] * HIST_LEN   # left arm pad world Y (approach)
        self._r_xyz    = np.array(r_pad_xyz, dtype=float)
        self._s        = dict(
            phase="SETTLE", cycle=0,
            grip_frac=0.0, grip_angle=GRIPPER_OPEN_ANGLE_RAD,
            friction_n=0.0, press_mm=0.0, lift_m=0.0,
            pad_xyz=np.zeros(3), contact_y=0.0,
        )
        self._tick = 0
        self._win  = ui.Window(
            "Dual Arm Grasp Monitor", width=600, height=680,
            flags=(ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE),
        )
        self._rebuild()

    def push(self, phase, cycle, grip_frac, grip_angle, friction_n, press_mm, lift_m,
             pad_xyz=None, contact_y=0.0):
        self._hist_f  = (self._hist_f  + [friction_n])[-HIST_LEN:]
        if pad_xyz is not None:
            self._hist_py = (self._hist_py + [float(pad_xyz[1])])[-HIST_LEN:]
        self._s.update(
            phase=phase, cycle=cycle,
            grip_frac=grip_frac, grip_angle=grip_angle,
            friction_n=friction_n, press_mm=press_mm, lift_m=lift_m,
            pad_xyz=(np.array(pad_xyz) if pad_xyz is not None else self._s["pad_xyz"]),
            contact_y=contact_y,
        )
        self._tick += 1
        if self._tick % UI_EVERY == 0:
            self._rebuild()

    def _rebuild(self):
        ui    = self._ui
        s     = self._s
        PHASE_COLORS = {
            "SETTLE":    0xFF888888, "PLAN":      0xFFFFAA00,
            "TO_PRE":    0xFF00AAFF, "TO_NEAR":   0xFF22CCFF,
            "APPROACH":  0xFF44DDFF, "CLOSE_GRIP":0xFFFFFF44,
            "LIFT":      0xFF44FF44, "HOLD":      0xFF22BB22,
            "LOWER":     0xFF88FF88, "RELEASE":   0xFFFFAA44,
            "RETRACT":   0xFF0088FF, "HOME":      0xFF006688,
            "PAUSE":     0xFF444444,
        }
        phase_col  = PHASE_COLORS.get(s["phase"], 0xFFAAAAAA)
        in_contact = s["press_mm"] > 0.0
        gripped    = s["grip_frac"] > 0.8
        gap_mm     = pad_separation_m(s["grip_angle"]) * 1000.0
        px, py, pz = float(s["pad_xyz"][0]), float(s["pad_xyz"][1]), float(s["pad_xyz"][2])
        rx, ry, rz = float(self._r_xyz[0]), float(self._r_xyz[1]), float(self._r_xyz[2])
        ct         = float(s["contact_y"])
        ear_face_y = ct - PAD_FACE_DEPTH_M

        # Y history bounds: ear_face ± some margin
        y_lo = ear_face_y - 0.02
        y_hi = ear_face_y + PRE_Y_OFFSET + 0.02

        self._win.frame.clear()
        with self._win.frame:
            with ui.VStack(spacing=3):

                # ── Header ────────────────────────────────────────────────────
                with ui.HStack(height=28):
                    ui.Label("DUAL ARM — TRAY GRASP CYCLE",
                             style={"font_size": 15, "color": 0xFFFFFFFF})
                    ui.Spacer()
                    ui.Label(f"● {s['phase']}  #{s['cycle']}",
                             style={"font_size": 13, "color": phase_col})
                ui.Separator(height=2)

                # ── LEFT ARM ──────────────────────────────────────────────────
                ui.Label("LEFT ARM  (grasp / lift)",
                         style={"font_size": 12, "color": 0xFF88DDFF}, height=18)

                with ui.HStack(height=17):
                    ui.Label(
                        f"Pad EE  X={px:.4f}  Y={py:.4f}  Z={pz:.4f}",
                        style={"font_size": 11},
                    )
                    ui.Spacer()
                    if in_contact:
                        ui.Label("CONTACT", style={"font_size": 11, "color": 0xFFFFDD44})
                    else:
                        dist = py - ear_face_y
                        ui.Label(f"dist={dist*1000:.1f}mm",
                                 style={"font_size": 10, "color": 0xFF888888})

                # Gripper state
                with ui.HStack(height=17):
                    status = "CLOSED" if gripped else "OPEN  "
                    ui.Label(
                        f"Gripper  {status}  gap={gap_mm:.1f}mm  "
                        f"angle={math.degrees(s['grip_angle']):.1f}°",
                        style={"font_size": 11},
                    )
                with ui.ZStack(height=8):
                    ui.Rectangle(style={"background_color": 0xFF222222, "border_radius": 2})
                    pct = min(100.0, s["grip_frac"] * 100.0)
                    if pct > 0:
                        with ui.HStack():
                            col = 0xFF44FF44 if gripped else 0xFF44AAFF
                            ui.Rectangle(width=ui.Percent(pct),
                                         style={"background_color": col, "border_radius": 2})
                            ui.Spacer()
                ui.Spacer(height=2)

                # Contact force bar
                with ui.HStack(height=20):
                    ui.Label(f"Friction Force  F = {s['friction_n']:.2f} N",
                             style={"font_size": 13})
                    if in_contact:
                        col = 0xFFFFDD44 if s["press_mm"] > 1.0 else 0xFF44FF88
                        ui.Label(f"  press={s['press_mm']:.2f}mm",
                                 style={"font_size": 11, "color": col})
                with ui.ZStack(height=12):
                    ui.Rectangle(style={"background_color": 0xFF222222, "border_radius": 3})
                    pct_f = min(100.0, s["friction_n"] / FORCE_MAX_N * 100.0)
                    if pct_f > 0:
                        with ui.HStack():
                            ui.Rectangle(
                                width=ui.Percent(pct_f),
                                style={"background_color": _bar_force(s["friction_n"] / FORCE_MAX_N),
                                       "border_radius": 3},
                            )
                            ui.Spacer()
                if in_contact:
                    with ui.ZStack(height=6):
                        ui.Rectangle(style={"background_color": 0xFF111111, "border_radius": 2})
                        pct_p = min(100.0, s["press_mm"] / PRESS_MAX_MM * 100.0)
                        with ui.HStack():
                            ui.Rectangle(
                                width=ui.Percent(pct_p),
                                style={"background_color": _bar_press(s["press_mm"] / PRESS_MAX_MM),
                                       "border_radius": 2},
                            )
                            ui.Spacer()
                else:
                    ui.Spacer(height=6)

                # Force history chart
                ui.Label("Friction Force (N) — history",
                         style={"font_size": 9, "color": 0xFF666666}, height=13)
                ui.Plot(
                    ui.Type.LINE, 0.0, FORCE_MAX_N, *self._hist_f, height=75,
                    style={"color": 0xFF22AAFF, "background_color": 0xFF05050F},
                )

                # Pad Y approach chart — shows arm moving toward ear face
                ui.Label(
                    f"Pad-Y approach (m) — ear_face={ear_face_y:.4f}  contact@{ct:.4f}",
                    style={"font_size": 9, "color": 0xFF666666}, height=13,
                )
                ui.Plot(
                    ui.Type.LINE, y_lo, y_hi, *self._hist_py, height=65,
                    style={"color": 0xFFFF8822, "background_color": 0xFF080508},
                )

                # Lift bar
                with ui.HStack(height=17):
                    ui.Label(
                        f"Lift Height  {s['lift_m']*100:.1f} cm / {LIFT_Z*100:.0f} cm",
                        style={"font_size": 12},
                    )
                with ui.ZStack(height=8):
                    ui.Rectangle(style={"background_color": 0xFF222222, "border_radius": 2})
                    pct_l = min(100.0, s["lift_m"] / LIFT_MAX_M * 100.0)
                    if pct_l > 0:
                        with ui.HStack():
                            ui.Rectangle(
                                width=ui.Percent(pct_l),
                                style={"background_color": 0xFF44FF88, "border_radius": 2},
                            )
                            ui.Spacer()

                ui.Separator(height=2)

                # ── RIGHT ARM ─────────────────────────────────────────────────
                ui.Label("RIGHT ARM  (holding home — q = [0,0,0,0,0,0])",
                         style={"font_size": 12, "color": 0xFFFFAA44}, height=18)
                with ui.HStack(height=17):
                    ui.Label(
                        f"Pad EE  X={rx:.4f}  Y={ry:.4f}  Z={rz:.4f}",
                        style={"font_size": 11},
                    )
                    ui.Spacer()
                    ui.Label("FK maintained every frame",
                             style={"font_size": 10, "color": 0xFF556655})
                with ui.ZStack(height=8):
                    ui.Rectangle(style={"background_color": 0xFF222222, "border_radius": 2})
                    with ui.HStack():
                        ui.Rectangle(
                            width=ui.Percent(100.0),
                            style={"background_color": 0xFF224422, "border_radius": 2},
                        )
                with ui.HStack(height=14):
                    ui.Label("● HOLDING HOME", style={"font_size": 10, "color": 0xFF55AA55})
                    ui.Spacer()
                    ui.Label(f"phase={s['phase']}  cycle={s['cycle']}",
                             style={"font_size": 10, "color": 0xFF666666})

                ui.Separator(height=2)
                ui.Label(
                    f"K_arm={K_ARM_Y:.0f} N/m  K_ear={K_EAR_Y:.0f} N/m  "
                    f"μ={MU_STATIC}  stop@F≥{GRIP_FORCE_STOP_N:.1f}N  "
                    f"pad_depth={PAD_FACE_DEPTH_M*1000:.0f}mm  lift={LIFT_Z*100:.0f}cm",
                    style={"font_size": 9, "color": 0xFF555555}, height=14,
                )


# ─────────────────────────────────────────────────────────────────────────────
# Main run loop
# ─────────────────────────────────────────────────────────────────────────────

def _run(app, stage):
    import omni.timeline
    from pxr import Usd, UsdGeom

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.set_end_time(99999.0)
    timeline.set_looping(False)

    # ── Kinematic chains ──────────────────────────────────────────────────────
    print("[TGC] Loading URDF…", flush=True)
    arm_jts   = load_joints(Path(DEFAULT_ARM_URDF))
    chains    = {n: chain_to_link(arm_jts, "Link_0", n) for n in LINK_NAMES}
    arm_chain = chains["Link_6"]
    lower, upper = joint_limits(arm_chain)
    q_zero = np.zeros(6)
    print("[TGC] URDF loaded.", flush=True)

    # ── Setup xform ops (both arms, both grippers) ────────────────────────────
    l_ops    = _setup_arm_ops(stage, LEFT_ROOT,  "tgc_arm")
    r_ops    = _setup_arm_ops(stage, RIGHT_ROOT, "tgc_arm")
    l_gr_ops = setup_gripper_xform_ops(stage, LEFT_GR,  "tgc_gr")
    r_gr_ops = setup_gripper_xform_ops(stage, RIGHT_GR, "tgc_gr")
    print("[TGC] Xform ops set up.", flush=True)

    # ── Set q=0 + gripper open BEFORE reading pad midpoint ───────────────────
    _set_arm_q(l_ops, chains, q_zero)
    _set_arm_q(r_ops, chains, q_zero)
    _set_gripper(l_gr_ops, GRIPPER_OPEN_ANGLE_RAD)
    _set_gripper(r_gr_ops, GRIPPER_OPEN_ANGLE_RAD)
    app.update()

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    base_world, _pad_home, link6_to_pad = selected_pad_midpoint(stage, cache, "left")
    print(f"[TGC] Left pad home:  {np.round(_pad_home[:3,3], 4)}", flush=True)

    # Right arm pad position (constant at q=0 — shown in GUI)
    _, r_pad_home, _ = selected_pad_midpoint(stage, cache, "right")
    r_pad_xyz = r_pad_home[:3, 3].copy()
    print(f"[TGC] Right pad home: {np.round(r_pad_xyz, 4)}", flush=True)

    def _pad_world_q(q: np.ndarray) -> np.ndarray:
        return pad_world_transform(arm_chain, base_world, link6_to_pad, q)

    # ── Create UI BEFORE physics starts ──────────────────────────────────────
    ui_mon = DualArmMonitorUI(r_pad_xyz)
    print("[TGC] UI created.", flush=True)
    app.update()

    # ── Start physics ─────────────────────────────────────────────────────────
    print("[TGC] Starting timeline…", flush=True)
    timeline.play()
    app.update()
    print("[TGC] First frame done — entering settle.", flush=True)

    # ── SETTLE ────────────────────────────────────────────────────────────────
    frame = 0
    for frame in range(SETTLE_FRAMES):
        _set_arm_q(l_ops, chains, q_zero)
        _set_arm_q(r_ops, chains, q_zero)
        _set_gripper(l_gr_ops, GRIPPER_OPEN_ANGLE_RAD)
        _set_gripper(r_gr_ops, GRIPPER_OPEN_ANGLE_RAD)
        ui_mon.push("SETTLE", 0, 0.0, GRIPPER_OPEN_ANGLE_RAD, 0.0, 0.0, 0.0,
                    pad_xyz=q_zero)
        app.update()
        if frame % 20 == 0:
            print(f"[TGC] settle {frame}/{SETTLE_FRAMES}", flush=True)
    frame = SETTLE_FRAMES

    # ── Diagnose tray structure ───────────────────────────────────────────────
    tray_prim = stage.GetPrimAtPath(TRAY_PATH)
    if tray_prim.IsValid():
        children = [c.GetName() for c in tray_prim.GetChildren()]
        print(f"[TGC] Tray children: {children}", flush=True)
        tray_t = _get_tray_translate(stage)
        print(f"[TGC] Tray translate (settled): {np.round(tray_t, 4)}", flush=True)
    else:
        print(f"[TGC] WARNING: {TRAY_PATH} not found!", flush=True)

    # ── Read grasp point world position (settled) ─────────────────────────────
    # /World/Tray/tray_grasp_point is a user-defined Xform child of the Tray.
    # After physics settles, its world transform correctly tracks the tray.
    cache.Clear()
    ear_xyz = None

    gp_prim = stage.GetPrimAtPath(GRASP_PRIM_PATH)
    print(f"[TGC] tray_grasp_point valid: {gp_prim.IsValid()}", flush=True)
    if gp_prim.IsValid():
        gp_T = get_world_pose(stage, cache, GRASP_PRIM_PATH)
        if gp_T is not None:
            cand = gp_T[:3, 3].copy()
            print(f"[TGC] tray_grasp_point world pos: {np.round(cand, 4)}", flush=True)
            # Accept if norm > 0.05 (clearly not origin/identity)
            if np.linalg.norm(cand) > 0.05:
                ear_xyz = cand

    if ear_xyz is None:
        # Calibrated fallback: measured in settled scene.
        ear_xyz = TRAY_GRASP_INIT.copy()
        if tray_prim.IsValid():
            tray_t = _get_tray_translate(stage)
            TRAY_Z_CALIB = 1.4443
            delta = float(tray_t[2]) - TRAY_Z_CALIB
            if abs(delta) > 0.005:
                ear_xyz[2] += delta
                print(f"[TGC] Tray settle Δz={delta:.4f} applied to fallback.", flush=True)
        print(f"[TGC] Grasp pos → calibrated fallback: {np.round(ear_xyz, 4)}", flush=True)
    else:
        print(f"[TGC] Grasp pos → from tray_grasp_point prim: {np.round(ear_xyz, 4)}", flush=True)

    # Contact threshold: pad_mid_y where pad face first touches ear face
    contact_y = float(ear_xyz[1]) + PAD_FACE_DEPTH_M
    print(
        f"[TGC] contact_y={contact_y:.4f}  "
        f"(ear_y={ear_xyz[1]:.4f} + depth={PAD_FACE_DEPTH_M})",
        flush=True,
    )

    # ── PLAN ──────────────────────────────────────────────────────────────────
    ui_mon.push("PLAN", 0, 0.0, GRIPPER_OPEN_ANGLE_RAD, 0.0, 0.0, 0.0,
                pad_xyz=_pad_world_q(q_zero)[:3, 3], contact_y=contact_y)
    app.update()

    seeds = [
        np.zeros(6),
        np.array([ 0.0,  0.30, -1.00,  0.0,  0.50,  0.0]),
        np.array([ 0.3,  0.50, -1.50,  0.0,  1.00,  0.3]),
        np.array([-0.3,  0.50, -1.50,  0.0,  1.00, -0.3]),
        np.array([ 0.0,  1.00, -2.00,  0.0,  1.50,  0.0]),
        np.array([ 0.1,  0.80, -1.80,  0.1,  1.20,  0.1]),
    ]

    # Waypoints along world-Y line through ear position (X and Z fixed):
    pre_xyz  = np.array([ear_xyz[0], ear_xyz[1] + PRE_Y_OFFSET,  ear_xyz[2]])
    near_xyz = np.array([ear_xyz[0], ear_xyz[1] + NEAR_Y_OFFSET, ear_xyz[2]])
    pick_xyz = np.array([ear_xyz[0], ear_xyz[1] + PICK_Y_OFFSET, ear_xyz[2]])
    lift_xyz = np.array([ear_xyz[0], ear_xyz[1] + PICK_Y_OFFSET, ear_xyz[2] + LIFT_Z])

    def _ik(label, target, ref=None):
        seed_list = ([] if ref is None else [ref]) + seeds
        q, pe, ue, fe, ok, msg = solve_pad_pose_ik(
            arm_chain, lower, upper, base_world, link6_to_pad,
            target, TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
            seed_list,
            continuity_weight=0.01 if ref is not None else 0.0,
        )
        print(
            f"[TGC]   IK {label} {np.round(target, 4)}: "
            f"pos={pe*1000:.1f}mm  up={ue:.1f}°  fw={fe:.1f}°",
            flush=True,
        )
        return q

    print("[TGC] IK: pre-grasp…", flush=True)
    q_pre  = _ik("pre",  pre_xyz)
    print("[TGC] IK: near…",      flush=True)
    q_near = _ik("near", near_xyz, ref=q_pre)
    print("[TGC] IK: pick…",      flush=True)
    q_pick = _ik("pick", pick_xyz, ref=q_near)
    print("[TGC] IK: lift…",      flush=True)
    q_lift = _ik("lift", lift_xyz, ref=q_pick)
    print("[TGC] IK done.", flush=True)

    # Joint-space paths (smoothstep)
    path_to_pre   = fallback_path(q_zero, q_pre,  MOTION_FRAMES)
    path_to_near  = fallback_path(q_pre,  q_near, MOTION_FRAMES)
    path_approach = fallback_path(q_near, q_pick, FINE_FRAMES)    # contact zone
    path_lift     = fallback_path(q_pick, q_lift, MOTION_FRAMES)
    path_lower    = fallback_path(q_lift, q_pick, MOTION_FRAMES)
    path_retract  = fallback_path(q_pick, q_pre,  MOTION_FRAMES)
    path_home     = fallback_path(q_pre,  q_zero, MOTION_FRAMES)

    grip_close = np.linspace(GRIPPER_OPEN_ANGLE_RAD, 0.0,                   GRIP_FRAMES)
    grip_open  = np.linspace(0.0,                    GRIPPER_OPEN_ANGLE_RAD, GRIP_FRAMES)

    # Tray tracking state
    tray_xyz0 = None
    pad_z0    = 0.0

    # ── Cycle state ───────────────────────────────────────────────────────────
    current_q  = q_zero.copy()
    current_gr = GRIPPER_OPEN_ANGLE_RAD
    path_idx   = 0
    cycle      = 1
    phase      = "TO_PRE"

    def _enter(new_phase):
        nonlocal phase, path_idx
        phase    = new_phase
        path_idx = 0
        print(f"[TGC] → {new_phase}  (cycle {cycle}  fr {frame})", flush=True)

    _enter("TO_PRE")

    while app.is_running():

        # ── State machine ─────────────────────────────────────────────────────
        if phase == "TO_PRE":
            if path_idx < len(path_to_pre):
                current_q = path_to_pre[path_idx]; path_idx += 1
            else:
                _enter("TO_NEAR")

        elif phase == "TO_NEAR":
            if path_idx < len(path_to_near):
                current_q = path_to_near[path_idx]; path_idx += 1
            else:
                _enter("APPROACH")

        elif phase == "APPROACH":
            # Advance toward pick — stop early if friction ≥ threshold.
            if path_idx < len(path_approach):
                next_q  = path_approach[path_idx]
                test_pw = _pad_world_q(next_q)
                _, _, test_f = _solve_ear_contact(float(test_pw[1, 3]), contact_y)
                if test_f >= GRIP_FORCE_STOP_N and path_idx > 0:
                    print(
                        f"[TGC] FORCE STOP  F={test_f:.2f}N ≥ {GRIP_FORCE_STOP_N}N  "
                        f"idx={path_idx}  pad_y={test_pw[1,3]:.4f}",
                        flush=True,
                    )
                    _enter("CLOSE_GRIP")
                else:
                    current_q = next_q
                    path_idx += 1
            else:
                _enter("CLOSE_GRIP")

        elif phase == "CLOSE_GRIP":
            if path_idx < len(grip_close):
                current_gr = grip_close[path_idx]; path_idx += 1
            else:
                current_gr = 0.0
                tray_xyz0  = _get_tray_translate(stage)
                pad_z0     = float(_pad_world_q(current_q)[2, 3])
                _set_tray_kinematic(stage, True)
                print(
                    f"[TGC] Tray kinematic ON  tray_z={tray_xyz0[2]:.4f}  pad_z={pad_z0:.4f}",
                    flush=True,
                )
                _enter("LIFT")

        elif phase == "LIFT":
            if path_idx < len(path_lift):
                current_q = path_lift[path_idx]; path_idx += 1
            else:
                _enter("HOLD")

        elif phase == "HOLD":
            if path_idx < HOLD_FRAMES:
                path_idx += 1
            else:
                _enter("LOWER")

        elif phase == "LOWER":
            if path_idx < len(path_lower):
                current_q = path_lower[path_idx]; path_idx += 1
            else:
                _enter("RELEASE")

        elif phase == "RELEASE":
            if path_idx < len(grip_open):
                current_gr = grip_open[path_idx]; path_idx += 1
            else:
                current_gr = GRIPPER_OPEN_ANGLE_RAD
                _set_tray_kinematic(stage, False)
                tray_xyz0 = None
                print("[TGC] Tray released → dynamic", flush=True)
                _enter("RETRACT")

        elif phase == "RETRACT":
            if path_idx < len(path_retract):
                current_q = path_retract[path_idx]; path_idx += 1
            else:
                _enter("HOME")

        elif phase == "HOME":
            if path_idx < len(path_home):
                current_q = path_home[path_idx]; path_idx += 1
            else:
                _enter("PAUSE")

        elif phase == "PAUSE":
            if path_idx < PAUSE_FRAMES:
                path_idx += 1
            else:
                cycle += 1
                current_q = q_zero.copy()
                _enter("TO_PRE")

        # ── Apply FK ──────────────────────────────────────────────────────────
        _set_arm_q(l_ops, chains, current_q)
        _set_arm_q(r_ops, chains, q_zero)      # right arm stays at home
        _set_gripper(l_gr_ops, current_gr)
        _set_gripper(r_gr_ops, GRIPPER_OPEN_ANGLE_RAD)

        # ── Analytical contact force ──────────────────────────────────────────
        pad_world = _pad_world_q(current_q)
        drive_y   = float(pad_world[1, 3])
        pad_xyz   = pad_world[:3, 3]
        _, press_mm, friction_n = _solve_ear_contact(drive_y, contact_y)

        # ── Tray Z tracking during lift / hold / lower ────────────────────────
        lift_m = 0.0
        if tray_xyz0 is not None:
            pad_z_now = float(pad_world[2, 3])
            delta_z   = pad_z_now - pad_z0
            lift_m    = max(0.0, delta_z)
            _set_tray_z(stage, tray_xyz0, delta_z)

        # ── UI ────────────────────────────────────────────────────────────────
        grip_frac = 1.0 - current_gr / GRIPPER_OPEN_ANGLE_RAD
        ui_mon.push(phase, cycle, grip_frac, current_gr, friction_n, press_mm, lift_m,
                    pad_xyz=pad_xyz, contact_y=contact_y)

        if frame % 30 == 0:
            print(
                f"[TGC] fr={frame:5d}  {phase:12s}  cy={cycle}"
                f"  pad_y={drive_y:.4f}  cty={contact_y:.4f}  F={friction_n:.2f}N"
                f"  gr={math.degrees(current_gr):.1f}°  lift={lift_m*100:.1f}cm",
                flush=True,
            )

        app.update()
        frame += 1

    print("[TGC] Stopped.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    print(f"[TGC] Scene: {DEFAULT_SCENE}", flush=True)
    ctx.open_stage(DEFAULT_SCENE)

    for i in range(200):
        app.update()
        s = ctx.get_stage()
        if s and s.GetPrimAtPath(LEFT_GR + "/left_pad").IsValid():
            print(f"[TGC] Stage ready ({i+1} frames)", flush=True)
            break
    else:
        print("[TGC] ERROR: left_pad not found after 200 frames", flush=True)
        return

    _run(app, ctx.get_stage())


if __name__ == "__main__":
    main()
