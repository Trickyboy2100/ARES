#!/usr/bin/env python3
"""Open the task USD in Isaac Sim GUI without running any motion playback.

Usage (--exec mode, recommended):
  CUDALIB=~/isaacsim/exts/omni.isaac.ml_archive/pip_prebundle
  LD_LIBRARY_PATH=$CUDALIB/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH \\
    ~/isaacsim/isaac-sim.sh --exec isaac_sim/simforge/demos/open_scene.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np

_SIMFORGE_ROOT = Path(__file__).resolve().parents[1]
_CORE = _SIMFORGE_ROOT / "core"
for _path in (str(_SIMFORGE_ROOT), str(_CORE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
DEFAULT_SCENE = os.environ.get("SIMFORGE_SCENE") or str(_SIMFORGE_ROOT / "scenes" / "main.usd")

from kinematics import get_world_pose
from planning import selected_pad_midpoint
from demos.tray_grasp_cycle.demo import (
    GRASP_PRIM_L,
    GRASP_PRIM_R,
    DRYER_PLACEMENT_TARGET,
    HANDOFF_EAR_HALF,
    HANDOFF_RIGHT_PAD_WORLD_OFFSET,
    LEFT_GR,
    LEFT_PICK_WORLD_OFFSET,
    PAD_FACE_DEPTH_M,
    TRAY_GRASP_INIT_L,
    _compute_handoff_center,
    _get_dryer_world_pos,
    apply_home_pose,
)

DEBUG_ROOT = "/World/TGC_DebugTargets"


def _make_material(stage, path, color):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _add_marker(stage, parent, name, xyz, radius, material):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    path = f"{parent}/{name}"
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(float(radius))
    sphere.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(material)

    label = UsdGeom.Scope.Define(stage, f"{path}_label")
    label.GetPrim().CreateAttribute("tgc:label", Sdf.ValueTypeNames.String).Set(name)
    label.GetPrim().CreateAttribute("tgc:xyz", Sdf.ValueTypeNames.Double3).Set(
        Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2]))
    )


def add_target_markers(stage):
    from pxr import Sdf, Usd, UsdGeom

    if stage.GetPrimAtPath(DEBUG_ROOT):
        stage.RemovePrim(DEBUG_ROOT)
    UsdGeom.Scope.Define(stage, DEBUG_ROOT)

    mats = {
        "grasp": _make_material(stage, f"{DEBUG_ROOT}/mat_grasp", (0.05, 0.55, 1.0)),
        "handoff": _make_material(stage, f"{DEBUG_ROOT}/mat_handoff", (1.0, 0.72, 0.08)),
        "dryer": _make_material(stage, f"{DEBUG_ROOT}/mat_dryer", (0.75, 0.18, 1.0)),
        "contact": _make_material(stage, f"{DEBUG_ROOT}/mat_contact", (0.1, 1.0, 0.25)),
    }

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    l_base_world, _, _ = selected_pad_midpoint(stage, cache, "left")
    r_base_world, _, _ = selected_pad_midpoint(stage, cache, "right")

    ear_xyz_L = None
    gp_T = get_world_pose(stage, cache, GRASP_PRIM_L)
    if gp_T is not None and np.linalg.norm(gp_T[:3, 3]) > 0.05:
        ear_xyz_L = gp_T[:3, 3].copy()
    if ear_xyz_L is None:
        ear_xyz_L = TRAY_GRASP_INIT_L.copy()

    ear_xyz_R = None
    gp_R = get_world_pose(stage, cache, GRASP_PRIM_R)
    if gp_R is not None and np.linalg.norm(gp_R[:3, 3]) > 0.05:
        ear_xyz_R = gp_R[:3, 3].copy()
    if ear_xyz_R is None:
        ear_xyz_R = ear_xyz_L.copy()
        ear_xyz_R[0] -= 2.0 * HANDOFF_EAR_HALF

    tray_center_x = (float(ear_xyz_L[0]) + float(ear_xyz_R[0])) / 2.0
    pick_xyz_L = ear_xyz_L.copy() + LEFT_PICK_WORLD_OFFSET
    contact_xyz_L = np.array([tray_center_x, ear_xyz_L[1] + PAD_FACE_DEPTH_M, ear_xyz_L[2]])

    handoff_center = _compute_handoff_center(l_base_world, r_base_world)
    handoff_pad_L = handoff_center + np.array([HANDOFF_EAR_HALF, 0.0, 0.0])
    handoff_pad_R = handoff_center + np.array([-HANDOFF_EAR_HALF, 0.0, 0.0]) + HANDOFF_RIGHT_PAD_WORLD_OFFSET

    dryer_pos = _get_dryer_world_pos(stage)
    dryer_target = DRYER_PLACEMENT_TARGET.copy()

    markers = [
        ("left_ear_grasp_prim", ear_xyz_L, 0.018, "grasp"),
        ("right_ear_grasp_prim", ear_xyz_R, 0.018, "grasp"),
        ("left_pick_grasp", pick_xyz_L, 0.025, "contact"),
        ("left_contact_y", contact_xyz_L, 0.016, "contact"),
        ("handoff_center", handoff_center, 0.025, "handoff"),
        ("handoff_left_pad", handoff_pad_L, 0.022, "handoff"),
        ("handoff_right_pad", handoff_pad_R, 0.022, "handoff"),
        ("dryer_world_pos", dryer_pos, 0.024, "dryer") if dryer_pos is not None else None,
        ("dryer_placement_target", dryer_target, 0.03, "dryer"),
    ]
    for item in markers:
        if item is None:
            continue
        name, xyz, radius, mat_key = item
        _add_marker(stage, DEBUG_ROOT, name, xyz, radius, mats[mat_key])

    stage.GetPrimAtPath(DEBUG_ROOT).CreateAttribute("tgc:note", Sdf.ValueTypeNames.String).Set(
        "Blue=tray grasp prims, green=left pick/contact, yellow=handoff, purple=dryer target."
    )


def main():
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    print(f"[SCENE] Opening: {DEFAULT_SCENE}", flush=True)
    if not Path(DEFAULT_SCENE).exists():
        raise RuntimeError(f"Scene USD not found: {DEFAULT_SCENE}")
    ctx.open_stage(DEFAULT_SCENE)

    for i in range(300):
        app.update()
        s = ctx.get_stage()
        if s and s.GetPrimAtPath(LEFT_GR + "/left_pad").IsValid():
            apply_home_pose(s)
            add_target_markers(s)
            app.update()
            print(f"[SCENE] Loaded ({i+1} frames). GUI is open — close window to quit.", flush=True)
            break

    while app.is_running():
        app.update()

    print("[SCENE] Window closed.", flush=True)


if __name__ == "__main__":
    main()
