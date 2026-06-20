#!/usr/bin/env python3
"""Build a clean kinematic task scene for cuRobo/FK tray handoff demos."""


# ── SimForge path injection ───────────────────────────────────────────────────
import sys as _sys
from pathlib import Path as _Path
_HERE = str(_Path(__file__).resolve().parent)
_SIMFORGE = str(_Path(__file__).resolve().parents[1])
_CORE = str(_Path(__file__).resolve().parents[1] / "core")
for _p in (_HERE, _SIMFORGE, _CORE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from install_gemini2_wrist_cameras import SIDES, get_reference_gemini_rel_to_link6, install_side


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "/home/andyee/isaacsim/playground/2026060721_control_fixed.usd"
DEFAULT_OUT = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_REPORT = PLAYGROUND_ROOT / "reports/curobo_task_scene_fix_report.json"
DEMO_TRAY_CENTER_WORLD = (0.35, 0.35, 1.015)
DEFAULT_GEMINI2_REFERENCE_SCENE = "/home/andyee/Documents/isc/202606060821.usd"


PHYSICS_APIS = [
    UsdPhysics.RigidBodyAPI,
    UsdPhysics.MassAPI,
    UsdPhysics.CollisionAPI,
    UsdPhysics.ArticulationRootAPI,
]


def remove_physics_apis(prim):
    removed = []
    for api in PHYSICS_APIS:
        try:
            if prim.RemoveAPI(api):
                removed.append(api.__name__)
        except Exception:
            pass
    return removed


def clear_xform(xf: UsdGeom.Xformable):
    xf.ClearXformOpOrder()
    return xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)


def make_material(stage, path: str, color):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PBR")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.58)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def build_demo_tray(stage, path="/World/DemoTray"):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)

    tray = UsdGeom.Xform.Define(stage, path)
    op = clear_xform(tray)
    # LoadingEquip top max z is about 1.006; this center puts the tray base
    # just above it while keeping the tray horizontal at rest.
    op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(*DEMO_TRAY_CENTER_WORLD)))

    base = UsdGeom.Cube.Define(stage, f"{path}/Base")
    base.CreateSizeAttr(1.0)
    base_xf = UsdGeom.Xformable(base.GetPrim())
    base_op = clear_xform(base_xf)
    base_op.Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(0.34, 0.22, 0.018))
    )

    rim_specs = [
        ("RimFront", (0.0, -0.115, 0.025), (0.36, 0.018, 0.045)),
        ("RimBack", (0.0, 0.115, 0.025), (0.36, 0.018, 0.045)),
        ("RimLeft", (-0.18, 0.0, 0.025), (0.018, 0.24, 0.045)),
        ("RimRight", (0.18, 0.0, 0.025), (0.018, 0.24, 0.045)),
    ]
    for name, translate, scale in rim_specs:
        rim = UsdGeom.Cube.Define(stage, f"{path}/{name}")
        rim.CreateSizeAttr(1.0)
        rim_xf = UsdGeom.Xformable(rim.GetPrim())
        rim_op = clear_xform(rim_xf)
        rim_op.Set(
            Gf.Matrix4d().SetScale(Gf.Vec3d(*scale))
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*translate))
        )

    mat = make_material(stage, "/World/MAT_DemoTray", (0.75, 0.82, 0.86))
    for prim in stage.GetPrimAtPath(path).GetChildren():
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def block_asset_arcs(prim):
    if not prim.IsValid():
        return False
    changed = False
    try:
        prim.GetReferences().ClearReferences()
        changed = True
    except Exception:
        pass
    try:
        prim.GetPayloads().ClearPayloads()
        changed = True
    except Exception:
        pass
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--gemini2-reference-scene", default=DEFAULT_GEMINI2_REFERENCE_SCENE)
    parser.add_argument("--no-gemini2-cameras", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    report_path = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, out)

    stage = Usd.Stage.Open(str(out))
    if stage is None:
        raise RuntimeError(f"Could not open copied scene: {out}")

    removed_count = 0
    removed_by_path = {}
    for prim in stage.TraverseAll():
        removed = remove_physics_apis(prim)
        if removed:
            removed_by_path[str(prim.GetPath())] = removed
            removed_count += len(removed)

    blocked_asset_paths = []
    inactive_paths = []
    for path in [
        "/World/Tray",
        "/World/Tray_01",
        "/World/robot/jaka_minicobo_left/joints",
        "/World/robot/jaka_minicobo_right/joints",
        "/World/robot/jaka_minicobo_left/root_joint",
        "/World/robot/jaka_minicobo_right/root_joint",
        "/World/robot/jaka_minicobo_left/world/visuals",
        "/World/robot/jaka_minicobo_right/world/visuals",
        "/World/robot/jaka_minicobo_left/Link_6/CAM_Mount/Gemini2",
        "/World/robot/jaka_minicobo_right/Link_6/CAM_Mount/Gemini2",
    ]:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            if block_asset_arcs(prim):
                blocked_asset_paths.append(path)
            prim.SetActive(False)
            inactive_paths.append(path)

    build_demo_tray(stage)
    gemini2_report = None
    if not args.no_gemini2_cameras:
        gemini2_report = {
            "reference_scene_read_only": args.gemini2_reference_scene,
            "sides": {},
        }
        for side in SIDES:
            rel = get_reference_gemini_rel_to_link6(args.gemini2_reference_scene, side)
            gemini2_report["sides"][side] = install_side(stage, side, rel)

    stage.GetRootLayer().Save()

    report = {
        "source": str(source),
        "out": str(out),
        "removed_physics_api_count": removed_count,
        "removed_physics_api_paths": removed_by_path,
        "inactive_paths": inactive_paths,
        "blocked_asset_paths": blocked_asset_paths,
        "demo_tray": {
            "path": "/World/DemoTray",
            "center_world_xyz": list(DEMO_TRAY_CENTER_WORLD),
            "size_xyz_approx": [0.36, 0.24, 0.07],
        },
        "gemini2_wrist_cameras": gemini2_report,
        "control_model": "kinematic_fk_playback_no_physx",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {report_path}")
    print(f"removed physics APIs: {removed_count}; inactive prims: {len(inactive_paths)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
