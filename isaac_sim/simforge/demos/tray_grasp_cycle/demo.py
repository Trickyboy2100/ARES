#!/usr/bin/env python3
"""Tray grasp cycle demo — left arm picks 2 mm ear tab, lifts 30 cm, returns, loops.

Grasp physics
─────────────
  EG2 X = world -Z  (jaw opens / closes vertically)
  EG2 Y = world +X  (TARGET_UP_WORLD)
  EG2 Z = world -Y  (approach direction, TARGET_FORWARD_WORLD)

  Pads straddle the 2 mm horizontal ear tab in Z (one pad above, one below).
  Approach from +Y side; normal force on ear face is ∥ world Y.
  Friction force ∥ world Z resists gravity and lifts the tray.

Cycle
─────
  home → pre-grasp → approach (force-stop) → close (physics drive) →
  FixedJoint → lift +30 cm → lower → open → remove joint →
  retract → home → pause → repeat

Usage
─────
  cd /path/to/PG-JY
  bash isaac_sim/simforge/demos/tray_grasp_cycle/launch.sh
"""
from __future__ import annotations

import math
import sys
import time
from enum import Enum
from pathlib import Path

import numpy as np

# ── sys.path ──────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_SIMFORGE  = str(_HERE.parents[1])          # simforge/  (contains config.py)
_CORE      = str(_HERE.parents[1] / "core") # simforge/core/
for _d in (_SIMFORGE, _CORE):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Scene ─────────────────────────────────────────────────────────────────────
try:
    import config as _cfg
    DEFAULT_SCENE = _cfg.SCENE_USD
except Exception:
    DEFAULT_SCENE = str(Path.home() / "isaacsim/playground/2026061100_main.usd")

# ── USD paths ─────────────────────────────────────────────────────────────────
LEFT_ROOT  = "/World/robot/jaka_minicobo_left"
RIGHT_ROOT = "/World/robot/jaka_minicobo_right"
TRAY_PATH  = "/World/Tray"
EAR_FRAME  = "/World/Tray/GraspFrames/YPlusEar"
PROBE_ROOT = "/World/TrayGraspCycleRuntime"
LINK_NAMES = ["Link_1", "Link_2", "Link_3", "Link_4", "Link_5", "Link_6"]

# ── Motion ────────────────────────────────────────────────────────────────────
TARGET_UP_WORLD      = np.array([1.0,  0.0, 0.0], dtype=float)  # EG2 Y = world +X
TARGET_FORWARD_WORLD = np.array([0.0, -1.0, 0.0], dtype=float)  # EG2 Z = world -Y
ORIENT_W = 0.22   # orientation weight for lift / lower

PRE_Y_OFFSET  = 0.125   # pre-grasp: 125 mm in +Y from ear
NEAR_Y_OFFSET = 0.040   # near: 40 mm — switch to slow approach
PICK_Y_OFFSET = 0.012   # pick: 12 mm — contact threshold
LIFT_M        = 0.30    # lift height
CONTACT_Y_THR = 0.0003  # 0.3 mm tray displacement → contact confirmed

# Frames per segment (@ 60 fps physics)
FRAMES_TO_PRE   = 91
FRAMES_APPROACH_FAST = 61
FRAMES_APPROACH_SLOW = 91
FRAMES_LIFT     = 91
FRAMES_LOWER    = 91
FRAMES_RETRACT  = 61
FRAMES_HOME     = 61
FRAMES_PAUSE    = 120   # 2 s pause between cycles

# ── Gripper ───────────────────────────────────────────────────────────────────
GRIPPER_OPEN_ANGLE_RAD = 0.1945       # → 50 mm pad gap
GRIPPER_JOINT_SUFFIX   = "joints/gripper_joint"
CLOSE_PHYSICS_FRAMES   = 60
OPEN_PHYSICS_FRAMES    = 30
GRIPPER_DRIVE_K        = 3.0          # N·m/rad drive spring
LEVER_ARM_M            = 0.055        # joint → pad contact

# ── Physics material ──────────────────────────────────────────────────────────
PAD_FRICTION_STATIC   = 1.5
PAD_FRICTION_DYNAMIC  = 1.0
TRAY_FRICTION_STATIC  = 1.5
TRAY_FRICTION_DYNAMIC = 1.0

# ── Gripper FK chains (EG2-4C2) ───────────────────────────────────────────────
GRIPPER_LINK_JOINT_CHAINS = {
    "left_outer_link":  [((-0.04, -0.009, 0.079), (0.0, -1.0, 0.0))],
    "left_inner_link":  [((-0.03, -0.009, 0.081), (0.0, -1.0, 0.0))],
    "right_inner_link": [((0.03,  -0.009, 0.081), (0.0,  1.0, 0.0))],
    "right_outer_link": [((0.04,  -0.009, 0.079), (0.0,  1.0, 0.0))],
    "left_pad": [
        ((-0.04,  -0.009, 0.079),   (0.0, -1.0, 0.0)),
        ((0.0223,  0.003, 0.035591),(0.0,  1.0, 0.0)),
    ],
    "right_pad": [
        ((0.04,   -0.009, 0.079),   (0.0,  1.0, 0.0)),
        ((-0.0223, 0.003, 0.035591),(0.0, -1.0, 0.0)),
    ],
}

_PAD_CONTACT = {
    "left_pad":  {"cx": +0.0071, "cy": +0.006, "cz": 0.020,
                  "hx": 0.014,   "hy":  0.009, "hz": 0.008},
    "right_pad": {"cx": -0.0071, "cy": +0.006, "cz": 0.020,
                  "hx": 0.014,   "hy":  0.009, "hz": 0.008},
}

# ── Phase ─────────────────────────────────────────────────────────────────────

class Phase(Enum):
    SETTLE   = "SETTLE"
    TO_PRE   = "TO PRE-GRASP"
    APPROACH = "APPROACH"
    CLOSE    = "CLOSE GRIPPER"
    LIFT     = "LIFT"
    LOWER    = "LOWER"
    RELEASE  = "RELEASE"
    RETRACT  = "RETRACT"
    HOME     = "HOME"
    PAUSE    = "PAUSE"


_PHASE_COLOR = {
    Phase.SETTLE:   0xFF888888,
    Phase.TO_PRE:   0xFF4488FF,
    Phase.APPROACH: 0xFF00FFAA,
    Phase.CLOSE:    0xFFFF8844,
    Phase.LIFT:     0xFF44FF44,
    Phase.LOWER:    0xFF44AAFF,
    Phase.RELEASE:  0xFFFF4444,
    Phase.RETRACT:  0xFFFFFF44,
    Phase.HOME:     0xFF888888,
    Phase.PAUSE:    0xFF555555,
}


# ── Math / USD helpers ────────────────────────────────────────────────────────

def _gfm(T):
    from pxr import Gf
    return Gf.Matrix4d(*sum(np.asarray(T, dtype=float).T.tolist(), []))


def _translate(xyz):
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def _axis_angle(axis, rad):
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        return np.eye(4)
    x, y, z = axis / n
    c, s = math.cos(rad), math.sin(rad)
    C = 1.0 - c
    R = np.array([
        [c+x*x*C, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, c+z*z*C],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    return T


def _gripper_link_T(link_name: str, rad: float) -> np.ndarray:
    T = np.eye(4)
    for xyz, axis in GRIPPER_LINK_JOINT_CHAINS[link_name]:
        T = T @ _translate(xyz) @ _axis_angle(axis, rad)
    return T


def bbox_payload(stage, path: str):
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    box = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"])
        .ComputeWorldBound(prim).ComputeAlignedBox()
    )
    return {"min": list(box.GetMin()), "max": list(box.GetMax())}


def bbox_center(b):
    return np.array([(b["min"][i] + b["max"][i]) * 0.5 for i in range(3)], dtype=float)


def xform_world_xyz(stage, path: str) -> np.ndarray:
    from pxr import Usd, UsdGeom
    prim = stage.GetPrimAtPath(path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return np.array(cache.GetLocalToWorldTransform(prim).ExtractTranslation(), dtype=float)


def _apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


# ── Physics setup ─────────────────────────────────────────────────────────────

def _ensure_phys_mat(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade
    if stage.GetPrimAtPath(path):
        return UsdShade.Material(stage.GetPrimAtPath(path))
    mat = UsdShade.Material.Define(stage, path)
    pm  = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(static)
    pm.CreateDynamicFrictionAttr().Set(dynamic)
    pm.CreateRestitutionAttr().Set(0.0)
    return mat


def apply_tray_friction(stage):
    from pxr import Usd, UsdPhysics, UsdShade
    mat = _ensure_phys_mat(stage, f"{TRAY_PATH}/CycleFrictionMat",
                           TRAY_FRICTION_STATIC, TRAY_FRICTION_DYNAMIC)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(TRAY_PATH)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def setup_pad_contacts(stage, gripper_root: str):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    mat = _ensure_phys_mat(stage, f"{gripper_root}/CyclePadMat",
                           PAD_FRICTION_STATIC, PAD_FRICTION_DYNAMIC)
    for pad_name, g in _PAD_CONTACT.items():
        pad_prim = stage.GetPrimAtPath(f"{gripper_root}/{pad_name}")
        if not pad_prim or not pad_prim.IsValid():
            continue
        rb = UsdPhysics.RigidBodyAPI.Apply(pad_prim)
        rb.CreateRigidBodyEnabledAttr().Set(True)
        rb.CreateKinematicEnabledAttr().Set(True)
        _apply_once(UsdPhysics.MassAPI, pad_prim).CreateMassAttr().Set(0.02)
        box_path = f"{gripper_root}/{pad_name}/CycleContactBox"
        if stage.GetPrimAtPath(box_path):
            stage.RemovePrim(box_path)
        box = UsdGeom.Cube.Define(stage, box_path)
        box.CreateSizeAttr().Set(1.0)
        xf = UsdGeom.Xformable(box.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(g["cx"], g["cy"], g["cz"]))
        xf.AddScaleOp().Set(Gf.Vec3f(g["hx"]*2, g["hy"]*2, g["hz"]*2))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    print(f"[CYCLE] Pad contact boxes set up under {gripper_root}", flush=True)


def ensure_carrier(stage):
    from pxr import UsdGeom, UsdPhysics
    if stage.GetPrimAtPath(PROBE_ROOT):
        stage.RemovePrim(PROBE_ROOT)
    UsdGeom.Xform.Define(stage, PROBE_ROOT)
    cube = UsdGeom.Cube.Define(stage, f"{PROBE_ROOT}/Carrier")
    prim = cube.GetPrim()
    cube.CreateSizeAttr(0.01)
    cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rb = _apply_once(UsdPhysics.RigidBodyAPI, prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    rb.CreateKinematicEnabledAttr().Set(True)
    _apply_once(UsdPhysics.MassAPI, prim).CreateMassAttr().Set(0.05)
    return prim, op


def set_carrier_xyz(op, xyz: np.ndarray):
    from pxr import Gf
    op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def create_grasp_lock(stage, carrier_path: str) -> str:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    joint_path = f"{PROBE_ROOT}/GraspFixedJoint"
    if stage.GetPrimAtPath(joint_path):
        return joint_path
    tray    = stage.GetPrimAtPath(TRAY_PATH)
    carrier = stage.GetPrimAtPath(carrier_path)
    cache   = UsdGeom.XformCache(Usd.TimeCode.Default())
    tray_w    = cache.GetLocalToWorldTransform(tray)
    carrier_w = cache.GetLocalToWorldTransform(carrier)
    anchor = carrier_w.ExtractTranslation()
    local0 = carrier_w.GetInverse().Transform(anchor)
    local1 = tray_w.GetInverse().Transform(anchor)
    j = UsdPhysics.FixedJoint.Define(stage, joint_path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(carrier_path)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(TRAY_PATH)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in local0]))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in local1]))
    q_inv = tray_w.ExtractRotationQuat().GetInverse().GetNormalized()
    im    = q_inv.GetImaginary()
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(float(q_inv.GetReal()),
                                          float(im[0]), float(im[1]), float(im[2])))
    j.CreateJointEnabledAttr().Set(True)
    j.CreateCollisionEnabledAttr().Set(False)
    print(f"[CYCLE] FixedJoint created: {joint_path}", flush=True)
    return joint_path


def remove_grasp_lock(stage):
    jp = f"{PROBE_ROOT}/GraspFixedJoint"
    if stage.GetPrimAtPath(jp):
        stage.RemovePrim(jp)
        print("[CYCLE] FixedJoint removed", flush=True)


# ── Arm FK ────────────────────────────────────────────────────────────────────

def set_robot_links(link_ops, chains, q):
    from kinematics import ARM_JOINTS, fk
    q_map = dict(zip(ARM_JOINTS, np.asarray(q, dtype=float).tolist()))
    for link_name in LINK_NAMES:
        link_ops[link_name].Set(_gfm(fk(chains[link_name], q_map)))


def pad_world_xyz(base_world, link6_chain, link6_to_pad, q) -> np.ndarray:
    from kinematics import ARM_JOINTS, fk
    q_map = dict(zip(ARM_JOINTS, np.asarray(q, dtype=float).tolist()))
    return (base_world @ fk(link6_chain, q_map) @ link6_to_pad)[:3, 3]


# ── Gripper drive ─────────────────────────────────────────────────────────────

def clear_gripper_xform_overrides(stage, gripper_root: str):
    from pxr import UsdGeom
    for link in GRIPPER_LINK_JOINT_CHAINS:
        prim = stage.GetPrimAtPath(f"{gripper_root}/{link}")
        if not prim or not prim.IsValid():
            continue
        attr = prim.GetAttribute(UsdGeom.Tokens.xformOpOrder)
        if attr and attr.IsAuthored():
            attr.Clear()
        for sfx in ("ear_grasp_gripper_fk", "force_demo_gripper_fk",
                    "grasp_cycle_fk", "fd_gr"):
            prop = f"xformOp:transform:{sfx}"
            if prim.HasAttribute(prop) and prim.GetAttribute(prop).IsAuthored():
                prim.RemoveProperty(prop)
    print(f"[CYCLE] Gripper xform overrides cleared: {gripper_root}", flush=True)


def set_gripper_drive(stage, gripper_root: str, target_deg: float) -> bool:
    attr = stage.GetPrimAtPath(
        f"{gripper_root}/{GRIPPER_JOINT_SUFFIX}"
    ).GetAttribute("drive:angular:physics:targetPosition")
    if not attr:
        return False
    attr.Set(float(target_deg))
    return True


def read_gripper_deg(stage, gripper_root: str) -> float | None:
    prim = stage.GetPrimAtPath(f"{gripper_root}/{GRIPPER_JOINT_SUFFIX}")
    if not prim or not prim.IsValid():
        return None
    attr = prim.GetAttribute("state:angular:physics:position")
    v = attr.Get() if attr else None
    return float(v) if v is not None else None


def grip_force_n(curr_deg: float | None, target_deg: float = 0.0) -> float:
    if curr_deg is None:
        return 0.0
    torque = GRIPPER_DRIVE_K * abs(math.radians(curr_deg) - math.radians(target_deg))
    return 2.0 * torque / LEVER_ARM_M


def pad_gap_mm(deg: float | None) -> float:
    if deg is None:
        return 50.0
    a = math.radians(abs(deg))
    return abs(-0.04 + 0.0223*math.cos(a) - 0.035591*math.sin(a)) * 2000.0


# ── Path planning ─────────────────────────────────────────────────────────────

def load_arm_chains():
    """Parse URDF and build kinematic chains (no USD reads — safe to call anytime)."""
    from kinematics import chain_to_link, load_joints
    from ik_sanity import joint_limits

    try:
        import config as _cfg
        urdf = str(_cfg.ARM_URDF)
    except Exception:
        urdf = ""
    if not urdf or not Path(urdf).is_file():
        urdf = str(Path.home() / "Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo.urdf")

    arm_joints  = load_joints(Path(urdf))
    chains      = {n: chain_to_link(arm_joints, "Link_0", n) for n in LINK_NAMES}
    link6_chain = chains["Link_6"]
    lower, upper = joint_limits(link6_chain)
    return chains, link6_chain, lower, upper


def build_paths(stage, chains, link6_chain, lower, upper,
                base_world, link6_to_pad):
    """Plan all IK paths offline (call after arm is at q=0 and pad poses are valid)."""
    from planning import constrained_pose_path, fallback_path, solve_pad_pose_ik

    seed_bank = [
        np.zeros(6),
        np.array([1.0,  0.5,  0.4, -1.1, -0.2, -0.4]),
        np.array([1.4,  0.6,  0.4, -1.1, -0.2, -0.4]),
        np.array([0.8,  0.8,  0.2, -1.2,  0.1, -0.2]),
        np.array([0.9,  0.5,  0.8, -0.8,  1.4,  1.5]),
        np.array([1.1,  0.4,  0.7, -0.9,  1.5,  1.6]),
        np.array([0.7,  0.7,  0.6, -1.0,  1.3,  1.4]),
    ]

    settled_bbox   = bbox_payload(stage, TRAY_PATH)
    settled_center = bbox_center(settled_bbox)
    ear_center     = xform_world_xyz(stage, EAR_FRAME)

    ee_z = float(settled_center[2])
    ex   = float(settled_center[0])
    ey   = float(ear_center[1])

    pre  = np.array([ex, ey + PRE_Y_OFFSET,  ee_z], dtype=float)
    near = np.array([ex, ey + NEAR_Y_OFFSET, ee_z], dtype=float)
    pick = np.array([ex, ey + PICK_Y_OFFSET, ee_z], dtype=float)

    print(f"[PLAN] EE Z={ee_z:.4f}  ear_Y={ey:.4f}", flush=True)
    print(f"[PLAN] targets: pre={pre.round(4)}  near={near.round(4)}  pick={pick.round(4)}", flush=True)

    q0 = np.zeros(6)
    print("[PLAN] Solving IK: home → pre-grasp…", flush=True)
    q_pre, pos_err, *_ = solve_pad_pose_ik(
        link6_chain, lower, upper, base_world, link6_to_pad,
        pre, TARGET_UP_WORLD, TARGET_FORWARD_WORLD, [q0] + seed_bank,
    )
    print(f"[PLAN] q_pre={np.degrees(q_pre).round(1).tolist()}  err={pos_err:.4f}m", flush=True)

    path_home_to_pre = fallback_path(q0, q_pre, count=FRAMES_TO_PRE)

    print("[PLAN] Path: pre → near (fast)…", flush=True)
    path_fast, _ = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_home_to_pre[-1], pre, near,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [q_pre] + seed_bank, count=FRAMES_APPROACH_FAST,
    )

    print("[PLAN] Path: near → pick (slow)…", flush=True)
    path_slow, _ = constrained_pose_path(
        link6_chain, lower, upper, base_world, link6_to_pad,
        path_fast[-1], near, pick,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [path_fast[-1]] + seed_bank, count=FRAMES_APPROACH_SLOW,
    )

    path_approach = np.vstack([path_fast, path_slow[1:]])
    print(f"[PLAN] Approach: {len(path_approach)} steps.  Planning complete.", flush=True)

    return {
        "chains":         chains,
        "link6_chain":    link6_chain,
        "lower":          lower,
        "upper":          upper,
        "base_world":     base_world,
        "link6_to_pad":   link6_to_pad,
        "seed_bank":      seed_bank,
        "settled_center": settled_center,
        "ear_center":     ear_center,
        "q0":             q0,
        "q_pre":          q_pre,
        "path_home_to_pre": path_home_to_pre,
        "path_approach":    path_approach,
    }


# ── omni.ui panel ─────────────────────────────────────────────────────────────

HIST_LEN = 300
UI_EVERY  = 3
FORCE_MAX = 30.0


class CycleUI:
    """Real-time panel showing phase, joint data, force, and tray position."""

    def __init__(self):
        import omni.ui as ui
        self._ui      = ui
        self._hist    = [0.0] * HIST_LEN
        self._phase   = Phase.SETTLE
        self._cycle   = 0
        self._q       = np.zeros(6)
        self._g_deg   = math.degrees(GRIPPER_OPEN_ANGLE_RAD)
        self._force   = 0.0
        self._tray_dy = 0.0
        self._tray_dz = 0.0
        self._tick    = 0
        self._win = ui.Window(
            "Tray Grasp Cycle", width=480, height=560,
            flags=(ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_RESIZE),
        )
        self._rebuild()

    def push(self, phase: Phase, cycle: int, q: np.ndarray, g_deg: float | None,
             force: float, tray_dy: float, tray_dz: float):
        if g_deg is None:
            g_deg = math.degrees(GRIPPER_OPEN_ANGLE_RAD)
        self._hist  = (self._hist + [force])[-HIST_LEN:]
        self._phase = phase
        self._cycle = cycle
        self._q     = np.asarray(q, dtype=float)
        self._g_deg = g_deg
        self._force = force
        self._tray_dy = tray_dy
        self._tray_dz = tray_dz
        self._tick += 1
        if self._tick % UI_EVERY == 0:
            self._rebuild()

    def _rebuild(self):
        ui = self._ui
        ph, cy = self._phase, self._cycle
        q      = np.degrees(self._q)
        g      = self._g_deg
        fn     = self._force
        dy     = self._tray_dy
        dz     = self._tray_dz
        gap    = pad_gap_mm(g)
        col_ph = _PHASE_COLOR.get(ph, 0xFFCCCCCC)

        self._win.frame.clear()
        with self._win.frame:
            with ui.VStack(spacing=4, style={"margin": 6}):

                # ── Header ──────────────────────────────────────────────────
                with ui.HStack(height=32):
                    ui.Label("TRAY GRASP CYCLE",
                             style={"font_size": 16, "color": 0xFFFFFFFF})
                    ui.Spacer()
                    ui.Label(f"#{cy}",
                             style={"font_size": 20, "color": 0xFFCCCCCC},
                             alignment=ui.Alignment.RIGHT)
                with ui.ZStack(height=26):
                    ui.Rectangle(style={"background_color": col_ph | 0x40000000,
                                         "border_radius": 4})
                    ui.Label(f"  {ph.value}",
                             style={"font_size": 17, "color": col_ph})
                ui.Separator(height=2)

                # ── Left arm joints ──────────────────────────────────────────
                ui.Label("LEFT ARM  (joint angles °)",
                         style={"font_size": 12, "color": 0xFF88AAFF}, height=18)
                with ui.HStack(height=40):
                    for i in range(6):
                        with ui.VStack():
                            ui.Label(f"J{i+1}",
                                     style={"font_size": 10, "color": 0xFF888888},
                                     alignment=ui.Alignment.CENTER_BOTTOM)
                            ui.Label(f"{q[i]:+.1f}",
                                     style={"font_size": 13},
                                     alignment=ui.Alignment.CENTER_TOP)
                ui.Separator(height=2)

                # ── Gripper ──────────────────────────────────────────────────
                ui.Label("GRIPPER (left, physics drive)",
                         style={"font_size": 12, "color": 0xFF88AAFF}, height=18)
                with ui.HStack(height=36):
                    with ui.VStack():
                        ui.Label("Drive target / actual",
                                 style={"font_size": 9, "color": 0xFF888888})
                        tgt = 0.0 if ph in (Phase.CLOSE, Phase.LIFT, Phase.LOWER) \
                              else math.degrees(GRIPPER_OPEN_ANGLE_RAD)
                        ui.Label(f"{tgt:.1f}° → {g:.1f}°",
                                 style={"font_size": 14})
                    ui.Spacer(width=12)
                    with ui.VStack():
                        ui.Label("Pad gap",
                                 style={"font_size": 9, "color": 0xFF888888})
                        ui.Label(f"{gap:.1f} mm",
                                 style={"font_size": 14})
                    ui.Spacer(width=12)
                    with ui.VStack():
                        ui.Label("Est. force",
                                 style={"font_size": 9, "color": 0xFF888888})
                        c_fn = (0xFF44FF44 if fn < 5 else
                                0xFFFFAA44 if fn < 15 else 0xFFFF4444)
                        ui.Label(f"{fn:.1f} N",
                                 style={"font_size": 15, "color": c_fn})

                # Force bar
                with ui.ZStack(height=12):
                    ui.Rectangle(style={"background_color": 0xFF1A1A1A,
                                         "border_radius": 3})
                    pct = min(100.0, fn / FORCE_MAX * 100.0)
                    if pct > 0:
                        bar_col = (0xFF44FF44 if fn < 5 else
                                   0xFFFFAA44 if fn < 15 else 0xFFFF4444)
                        with ui.HStack():
                            ui.Rectangle(width=ui.Percent(pct),
                                         style={"background_color": bar_col,
                                                "border_radius": 3})
                            ui.Spacer()

                # Force history chart
                ui.Plot(ui.Type.LINE, 0.0, FORCE_MAX, *self._hist, height=90,
                        style={"color": 0xFF44FF88, "background_color": 0xFF05050F})
                ui.Separator(height=2)

                # ── Tray ────────────────────────────────────────────────────
                ui.Label("TRAY POSITION",
                         style={"font_size": 12, "color": 0xFF88AAFF}, height=18)
                with ui.HStack(height=36):
                    with ui.VStack():
                        ui.Label("ΔY (contact drift)",
                                 style={"font_size": 9, "color": 0xFF888888})
                        c_dy = 0xFFFF4444 if abs(dy) > 1.0 else 0xFF44FF44
                        ui.Label(f"{dy:+.1f} mm",
                                 style={"font_size": 15, "color": c_dy})
                    ui.Spacer(width=20)
                    with ui.VStack():
                        ui.Label("ΔZ (lift height)",
                                 style={"font_size": 9, "color": 0xFF888888})
                        c_dz = 0xFF44FF44 if dz > 50 else 0xFFCCCCCC
                        ui.Label(f"{dz:+.1f} mm",
                                 style={"font_size": 15, "color": c_dz})
                ui.Separator(height=2)

                # ── Right arm (static) ───────────────────────────────────────
                ui.Label("RIGHT ARM  (static, not moving)",
                         style={"font_size": 12, "color": 0xFF555555}, height=16)
                ui.Label("J1–J6:  0.0° / 0.0° / 0.0° / 0.0° / 0.0° / 0.0°",
                         style={"font_size": 11, "color": 0xFF444444}, height=16)
                ui.Separator(height=2)

                # Footer
                ui.Label(
                    f"contact_thr={CONTACT_Y_THR*1000:.1f}mm  "
                    f"lift={LIFT_M*100:.0f}cm  "
                    f"K_drive={GRIPPER_DRIVE_K:.0f} N·m/rad  "
                    f"lever={LEVER_ARM_M*1000:.0f}mm",
                    style={"font_size": 9, "color": 0xFF555555}, height=14,
                )


# ── Single cycle execution ────────────────────────────────────────────────────

def run_cycle(app, stage, link_ops, gripper_root, carrier_prim, carrier_op,
              tray_y0: float, tray_z0: float, info: dict, ui: CycleUI,
              cycle_num: int) -> bool:
    """Execute one full grasp-lift-return cycle.  Returns True if successful."""
    from planning import constrained_pose_path, fallback_path

    carrier_path  = str(carrier_prim.GetPath())
    _open_deg     = math.degrees(GRIPPER_OPEN_ANGLE_RAD)

    def _tray_state():
        b = bbox_payload(stage, TRAY_PATH)
        c = bbox_center(b)
        return float(c[1]), float(c[2])

    def _update(ph, q, g_target_closed: bool):
        g_deg = read_gripper_deg(stage, gripper_root)
        tray_y, tray_z = _tray_state()
        fn = grip_force_n(g_deg, 0.0 if g_target_closed else _open_deg)
        ui.push(ph, cycle_num, q,
                g_deg,
                fn,
                (tray_y - tray_y0) * 1000,
                (tray_z - tray_z0) * 1000)

    def _tick(ph, q, g_closed: bool):
        carrier_xyz = pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], q)
        set_robot_links(link_ops, info["chains"], q)
        set_carrier_xyz(carrier_op, carrier_xyz)
        _update(ph, q, g_closed)
        app.update()
        return carrier_xyz

    # ── 1. home → pre-grasp ──────────────────────────────────────────────────
    print(f"[CYCLE {cycle_num}] Phase: home → pre-grasp", flush=True)
    for q in info["path_home_to_pre"]:
        _tick(Phase.TO_PRE, q, False)

    # Re-baseline tray Y after home→pre (arm may disturb table slightly)
    tray_y_base, _ = _tray_state()

    # ── 2. Approach with force-stop detection ────────────────────────────────
    print(f"[CYCLE {cycle_num}] Phase: approach (force-stop active)", flush=True)
    force_stop_idx = None
    q_stop   = None
    xyz_stop = None

    for i, q in enumerate(info["path_approach"]):
        xyz = _tick(Phase.APPROACH, q, False)
        tray_y, _ = _tray_state()
        if abs(tray_y - tray_y_base) > CONTACT_Y_THR:
            force_stop_idx = i
            q_stop   = q.copy()
            xyz_stop = xyz.copy()
            print(f"[CYCLE {cycle_num}] Contact at approach step {i}: "
                  f"tray ΔY={(tray_y-tray_y_base)*1000:+.2f}mm", flush=True)
            break

    if force_stop_idx is None:
        print(f"[CYCLE {cycle_num}] No contact detected — aborting cycle", flush=True)
        return False

    # ── 3. Close gripper (physics drive, force-limited) ──────────────────────
    print(f"[CYCLE {cycle_num}] Phase: close gripper ({CLOSE_PHYSICS_FRAMES} frames)", flush=True)
    set_gripper_drive(stage, gripper_root, 0.0)
    for _ in range(CLOSE_PHYSICS_FRAMES):
        set_robot_links(link_ops, info["chains"], q_stop)
        set_carrier_xyz(carrier_op, xyz_stop)
        _update(Phase.CLOSE, q_stop, True)
        app.update()

    g_final = read_gripper_deg(stage, gripper_root)
    print(f"[CYCLE {cycle_num}] Gripper stalled at {g_final:.2f}°  "
          f"gap={pad_gap_mm(g_final):.1f}mm  "
          f"force≈{grip_force_n(g_final):.1f}N", flush=True)

    # ── 4. FixedJoint (CLAUDE.md: only after force_stop confirmed) ───────────
    joint_path = create_grasp_lock(stage, carrier_path)
    for _ in range(10):
        app.update()

    # ── 5. Lift +LIFT_M in Z ─────────────────────────────────────────────────
    print(f"[CYCLE {cycle_num}] Phase: lift +{LIFT_M*100:.0f}cm", flush=True)
    lift_target = np.array([xyz_stop[0], xyz_stop[1], xyz_stop[2] + LIFT_M], dtype=float)
    path_lift, _ = constrained_pose_path(
        info["link6_chain"], info["lower"], info["upper"],
        info["base_world"], info["link6_to_pad"],
        q_stop, xyz_stop, lift_target,
        TARGET_UP_WORLD, TARGET_FORWARD_WORLD,
        [q_stop] + info["seed_bank"],
        count=FRAMES_LIFT,
        axis_weight=ORIENT_W, forward_weight=ORIENT_W,
    )
    q_lift = path_lift[-1].copy()
    for q in path_lift:
        _tick(Phase.LIFT, q, True)

    # ── 6. Lower (reverse lift, FixedJoint still active) ────────────────────
    print(f"[CYCLE {cycle_num}] Phase: lower", flush=True)
    for q in path_lift[::-1]:
        _tick(Phase.LOWER, q, True)

    # ── 7. Open gripper + remove joint ──────────────────────────────────────
    print(f"[CYCLE {cycle_num}] Phase: release", flush=True)
    set_gripper_drive(stage, gripper_root, _open_deg)
    remove_grasp_lock(stage)
    for _ in range(OPEN_PHYSICS_FRAMES):
        set_robot_links(link_ops, info["chains"], q_stop)
        set_carrier_xyz(carrier_op, xyz_stop)
        _update(Phase.RELEASE, q_stop, False)
        app.update()

    # ── 8. Retract: reverse the approach path back to pre-grasp ─────────────
    print(f"[CYCLE {cycle_num}] Phase: retract", flush=True)
    path_retract = info["path_approach"][:force_stop_idx+1][::-1]
    for q in path_retract:
        _tick(Phase.RETRACT, q, False)

    # ── 9. Home (joint-space) ────────────────────────────────────────────────
    print(f"[CYCLE {cycle_num}] Phase: home", flush=True)
    path_to_home = fallback_path(info["q_pre"], info["q0"], count=FRAMES_HOME)
    for q in path_to_home:
        _tick(Phase.HOME, q, False)

    # ── 10. Pause ────────────────────────────────────────────────────────────
    print(f"[CYCLE {cycle_num}] Pause ({FRAMES_PAUSE} frames)…", flush=True)
    for _ in range(FRAMES_PAUSE):
        set_robot_links(link_ops, info["chains"], info["q0"])
        set_carrier_xyz(carrier_op, pad_world_xyz(info["base_world"], info["link6_chain"], info["link6_to_pad"], info["q0"]))
        _update(Phase.PAUSE, info["q0"], False)
        app.update()

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from pxr import Sdf, Usd, UsdGeom

    from kinematics import GRIPPER_ROOT_SUFFIX

    app      = omni.kit.app.get_app()
    ctx      = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()

    # ── Load scene ────────────────────────────────────────────────────────────
    print(f"[CYCLE] Opening scene: {DEFAULT_SCENE}", flush=True)
    ctx.open_stage(DEFAULT_SCENE)
    for _ in range(200):
        app.update()
        time.sleep(0.005)
    stage = ctx.get_stage()
    if stage is None or not stage.GetPrimAtPath("/World").IsValid():
        print("[CYCLE] ERROR: stage not loaded!", flush=True)
        return
    print("[CYCLE] Stage loaded.", flush=True)

    # ── Purge stale runtime prims from a previous saved session ──────────────
    stale = stage.GetPrimAtPath(PROBE_ROOT)
    if stale and stale.IsValid():
        stage.RemovePrim(Sdf.Path(PROBE_ROOT))
        print(f"[CYCLE] Purged stale prim: {PROBE_ROOT}", flush=True)

    # ── Physics setup ─────────────────────────────────────────────────────────
    gripper_root = f"{LEFT_ROOT}/{GRIPPER_ROOT_SUFFIX}"
    setup_pad_contacts(stage, gripper_root)
    apply_tray_friction(stage)
    carrier_prim, carrier_op = ensure_carrier(stage)

    # ── Wait for Play ─────────────────────────────────────────────────────────
    timeline.set_current_time(0.0)
    timeline.set_end_time(999999.0)
    timeline.set_looping(False)
    print("[CYCLE] Ready. Press Play in the Isaac Sim GUI to start.", flush=True)
    while not timeline.is_playing():
        app.update()
        time.sleep(0.02)

    # ── Physics settle ────────────────────────────────────────────────────────
    settle_frames = 120
    print(f"[CYCLE] Settling physics ({settle_frames} frames)…", flush=True)
    for _ in range(settle_frames):
        app.update()

    # ── Build kinematic chains from URDF (pure parse, no stage reads) ───────────
    print("[CYCLE] Building arm chains from URDF…", flush=True)
    chains, link6_chain, lower, upper = load_arm_chains()

    # ── Setup arm FK ops ──────────────────────────────────────────────────────
    from pxr import UsdGeom as _UG
    link_ops = {}
    for link_name in LINK_NAMES:
        prim = stage.GetPrimAtPath(f"{LEFT_ROOT}/{link_name}")
        xf   = _UG.Xformable(prim)
        xf.ClearXformOpOrder()
        link_ops[link_name] = xf.AddTransformOp(_UG.XformOp.PrecisionDouble, "cycle_fk")

    # Set arm to q=0 IMMEDIATELY so USD prim world poses are valid for IK reads
    q0 = np.zeros(6)
    set_robot_links(link_ops, chains, q0)
    for _ in range(10):
        app.update()

    # Switch gripper to physics drive (clear any FK overrides from prior runs)
    clear_gripper_xform_overrides(stage, gripper_root)
    _open_deg = math.degrees(GRIPPER_OPEN_ANGLE_RAD)
    ok = set_gripper_drive(stage, gripper_root, _open_deg)
    if not ok:
        print("[CYCLE] WARNING: gripper drive not available — will abort", flush=True)
        return
    print(f"[CYCLE] Gripper drive set to open: {_open_deg:.2f}°", flush=True)

    # ── Read pad midpoint (arm is at q=0, xform ops are set correctly) ─────────
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    from planning import selected_pad_midpoint
    base_world, _, link6_to_pad = selected_pad_midpoint(stage, cache, "left")
    print(f"[CYCLE] Arm base: {base_world[:3,3].round(4)}", flush=True)

    # ── Read settled tray state ───────────────────────────────────────────────
    tray_bb = bbox_payload(stage, TRAY_PATH)
    tray_c  = bbox_center(tray_bb)
    tray_y0 = float(tray_c[1])
    tray_z0 = float(tray_c[2])
    print(f"[CYCLE] Tray settled: Y={tray_y0:.4f}  Z={tray_z0:.4f}", flush=True)

    # ── Path planning (offline IK solve — 30–90 s) ────────────────────────────
    print("[CYCLE] Planning arm paths (IK solving — may take 30–90 s)…", flush=True)
    info = build_paths(stage, chains, link6_chain, lower, upper,
                       base_world, link6_to_pad)

    # ── Build UI ──────────────────────────────────────────────────────────────
    panel = CycleUI()
    panel.push(Phase.SETTLE, 0, info["q0"], _open_deg, 0.0, 0.0, 0.0)
    app.update()

    # ── Main cycle loop ───────────────────────────────────────────────────────
    cycle = 0
    print("[CYCLE] Starting grasp cycle loop. Close window or press Stop to quit.", flush=True)
    while app.is_running() and timeline.is_playing():
        cycle += 1
        print(f"\n[CYCLE] ═══ Cycle {cycle} start ═══", flush=True)
        success = run_cycle(
            app, stage, link_ops, gripper_root, carrier_prim, carrier_op,
            tray_y0, tray_z0, info, panel, cycle,
        )
        if not success:
            print(f"[CYCLE] Cycle {cycle} failed — retrying after pause", flush=True)
            for _ in range(FRAMES_PAUSE * 2):
                app.update()

    print("[CYCLE] Loop ended.", flush=True)


if __name__ == "__main__":
    main()
