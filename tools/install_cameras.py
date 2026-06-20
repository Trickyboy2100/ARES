#!/usr/bin/env python3
"""Install Gemini2 wrist RGB-D cameras into the task scene.

The source USD is read-only and used only to recover the manually placed
Gemini2 transform relative to each arm's Link_6. The active task scene receives
local Camera prims for RGB/depth rendering plus the official Isaac Orbbec
Gemini2 visual model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE_SCENE = "/home/andyee/Documents/isc/202606060821.usd"
DEFAULT_TARGET_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_REPORT = PLAYGROUND_ROOT / "reports/gemini2_wrist_camera_install_report.json"
ORBBEC_GEMINI2_MODEL_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Sensors/Orbbec/Gemini2/orbbec_gemini2_v1.0.usd"
)
SIDES = ("left", "right")

RGB_RESOLUTION = [1280, 720]
DEPTH_RESOLUTION = [640, 400]
RGB_HFOV_DEG = 86.0
RGB_VFOV_DEG = 55.0
DEPTH_HFOV_DEG = 91.0
DEPTH_VFOV_DEG = 66.0
DEPTH_RANGE_M = [0.15, 10.0]
CAMERA_MODEL_SCALE = 1.0
CAMERA_MODEL_FORWARD_FLIP_DEG = 180.0
CAMERA_IMAGE_ROLL_DEG = 180.0
CAMERA_OPTICAL_OFFSET_M = -0.08
DEFAULT_FORWARD_FLIP_DEG = 180.0


def clear_xform(prim):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    return xf


def set_matrix_xform(prim, matrix: Gf.Matrix4d):
    xf = clear_xform(prim)
    xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(matrix)


def set_identity_xform(prim):
    set_matrix_xform(prim, Gf.Matrix4d(1.0))


def set_camera_image_xform(prim):
    xf = clear_xform(prim)
    # Put the render camera at the physical lens/front plane instead of at the
    # sensor body's origin, otherwise the wrist/camera housing occludes RGB.
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CAMERA_OPTICAL_OFFSET_M))
    # Keep the optical forward axis unchanged, but make image up point upward
    # in the robot/world frame instead of rendering upside down.
    xf.AddRotateZOp().Set(CAMERA_IMAGE_ROLL_DEG)
    return xf


def add_camera_model_reference(stage: Usd.Stage, path: str):
    model = UsdGeom.Xform.Define(stage, path)
    xf = clear_xform(model.GetPrim())
    xf.AddRotateYOp().Set(CAMERA_MODEL_FORWARD_FLIP_DEG)
    xf.AddScaleOp().Set(Gf.Vec3d(CAMERA_MODEL_SCALE, CAMERA_MODEL_SCALE, CAMERA_MODEL_SCALE))
    model.GetPrim().GetReferences().AddReference(ORBBEC_GEMINI2_MODEL_USD)
    model.GetPrim().CreateAttribute("gemini2:visual_model_source", Sdf.ValueTypeNames.String).Set(
        ORBBEC_GEMINI2_MODEL_USD
    )
    model.GetPrim().CreateAttribute("gemini2:visual_model_scale", Sdf.ValueTypeNames.Float).Set(
        CAMERA_MODEL_SCALE
    )
    model.GetPrim().CreateAttribute("gemini2:visual_model_forward_flip_deg", Sdf.ValueTypeNames.Float).Set(
        CAMERA_MODEL_FORWARD_FLIP_DEG
    )
    return model


def apply_camera_forward_flip(matrix: Gf.Matrix4d, flip_deg: float) -> Gf.Matrix4d:
    # USD cameras look along local -Z. Rotating around the camera's local Y axis
    # flips the forward/back sign without rolling the camera upside down.
    flip = Gf.Matrix4d(1.0)
    flip.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), float(flip_deg)))
    corrected = Gf.Matrix4d(matrix * flip)
    corrected.SetTranslateOnly(matrix.ExtractTranslation())
    return corrected


def camera_focal_from_fov(aperture_mm: float, fov_deg: float) -> float:
    return aperture_mm / (2.0 * math.tan(math.radians(fov_deg) * 0.5))


def configure_camera(
    cam: UsdGeom.Camera,
    *,
    resolution,
    hfov_deg: float,
    vfov_deg: float,
    clipping_range,
):
    horizontal_aperture_mm = 20.0
    vertical_aperture_mm = horizontal_aperture_mm * float(resolution[1]) / float(resolution[0])
    focal_mm = camera_focal_from_fov(horizontal_aperture_mm, hfov_deg)

    cam.CreateHorizontalApertureAttr().Set(horizontal_aperture_mm)
    cam.CreateVerticalApertureAttr().Set(vertical_aperture_mm)
    cam.CreateFocalLengthAttr().Set(focal_mm)
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(float(clipping_range[0]), float(clipping_range[1])))
    cam.CreateFStopAttr().Set(0.0)   # pinhole — disables DoF blur for sensor simulation

    prim = cam.GetPrim()
    prim.CreateAttribute("gemini2:resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(*resolution))
    prim.CreateAttribute("gemini2:hfov_deg", Sdf.ValueTypeNames.Float).Set(float(hfov_deg))
    prim.CreateAttribute("gemini2:vfov_deg", Sdf.ValueTypeNames.Float).Set(float(vfov_deg))
    prim.CreateAttribute("gemini2:render_annotator", Sdf.ValueTypeNames.String).Set("rgb")
    return {
        "resolution": list(resolution),
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "horizontal_aperture_mm": horizontal_aperture_mm,
        "vertical_aperture_mm": vertical_aperture_mm,
        "focal_length_mm": focal_mm,
        "clipping_range_m": list(clipping_range),
    }


def get_reference_gemini_rel_to_link6(reference_scene: str, side: str) -> Gf.Matrix4d:
    stage = Usd.Stage.Open(reference_scene)
    if stage is None:
        raise RuntimeError(f"Could not open reference scene: {reference_scene}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    link6 = stage.GetPrimAtPath(f"/World/robot/jaka_minicobo_{side}/Link_6")
    gemini = stage.GetPrimAtPath(f"/World/robot/jaka_minicobo_{side}/Link_6/CAM_Mount/Gemini2")
    if not link6.IsValid():
        raise RuntimeError(f"Missing reference Link_6 for {side}")
    if not gemini.IsValid():
        raise RuntimeError(f"Missing reference Gemini2 prim for {side}")
    return cache.ComputeRelativeTransform(gemini, link6)[0]


def matrix_to_rows(matrix: Gf.Matrix4d):
    return [[round(float(matrix[i][j]), 9) for j in range(4)] for i in range(4)]


def install_side(
    stage: Usd.Stage,
    side: str,
    rel_link6_gemini: Gf.Matrix4d,
    forward_flip_deg: float = DEFAULT_FORWARD_FLIP_DEG,
):
    link6_path = f"/World/robot/jaka_minicobo_{side}/Link_6"
    link6 = stage.GetPrimAtPath(link6_path)
    if not link6.IsValid():
        raise RuntimeError(f"Missing target Link_6 for {side}: {link6_path}")

    root_path = f"{link6_path}/Gemini2Sim"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)

    corrected_rel_link6_gemini = apply_camera_forward_flip(rel_link6_gemini, forward_flip_deg)

    rig = UsdGeom.Xform.Define(stage, root_path)
    set_matrix_xform(rig.GetPrim(), corrected_rel_link6_gemini)
    rig.GetPrim().CreateAttribute("gemini2:reference_source", Sdf.ValueTypeNames.String).Set(
        "manual Gemini2 transform from /home/andyee/Documents/isc/202606060821.usd"
    )
    rig.GetPrim().CreateAttribute("gemini2:note", Sdf.ValueTypeNames.String).Set(
        "USD cameras are local lightweight simulation stand-ins for RGB-D capture."
    )
    rig.GetPrim().CreateAttribute("gemini2:forward_flip_deg", Sdf.ValueTypeNames.Float).Set(
        float(forward_flip_deg)
    )

    add_camera_model_reference(stage, f"{root_path}/CameraModel")

    rgb_cam = UsdGeom.Camera.Define(stage, f"{root_path}/RgbCamera")
    set_camera_image_xform(rgb_cam.GetPrim())
    rgb_info = configure_camera(
        rgb_cam,
        resolution=RGB_RESOLUTION,
        hfov_deg=RGB_HFOV_DEG,
        vfov_deg=RGB_VFOV_DEG,
        clipping_range=DEPTH_RANGE_M,
    )

    depth_cam = UsdGeom.Camera.Define(stage, f"{root_path}/DepthCamera")
    set_camera_image_xform(depth_cam.GetPrim())
    depth_info = configure_camera(
        depth_cam,
        resolution=DEPTH_RESOLUTION,
        hfov_deg=DEPTH_HFOV_DEG,
        vfov_deg=DEPTH_VFOV_DEG,
        clipping_range=DEPTH_RANGE_M,
    )
    depth_cam.GetPrim().GetAttribute("gemini2:render_annotator").Set("distance_to_camera")
    depth_cam.GetPrim().CreateAttribute("gemini2:range_min_m", Sdf.ValueTypeNames.Float).Set(DEPTH_RANGE_M[0])
    depth_cam.GetPrim().CreateAttribute("gemini2:range_max_m", Sdf.ValueTypeNames.Float).Set(DEPTH_RANGE_M[1])
    depth_cam.GetPrim().CreateAttribute("gemini2:noise_sigma_m", Sdf.ValueTypeNames.Float).Set(0.003)
    depth_cam.GetPrim().CreateAttribute("gemini2:noise_proportional_coeff", Sdf.ValueTypeNames.Float).Set(0.0005)
    depth_cam.GetPrim().CreateAttribute("gemini2:invalid_ratio", Sdf.ValueTypeNames.Float).Set(0.002)

    return {
        "rig_path": root_path,
        "rgb_camera_path": f"{root_path}/RgbCamera",
        "depth_camera_path": f"{root_path}/DepthCamera",
        "reference_relative_to_link6_matrix4x4": matrix_to_rows(rel_link6_gemini),
        "relative_to_link6_matrix4x4": matrix_to_rows(corrected_rel_link6_gemini),
        "relative_to_link6_translation_m": [
            round(float(v), 9) for v in corrected_rel_link6_gemini.ExtractTranslation()
        ],
        "forward_flip_deg": float(forward_flip_deg),
        "forward_flip_axis": "camera_local_y",
        "visual_model": {
            "source_usd": ORBBEC_GEMINI2_MODEL_USD,
            "scale": CAMERA_MODEL_SCALE,
            "forward_flip_deg": CAMERA_MODEL_FORWARD_FLIP_DEG,
            "forward_flip_axis": "model_local_y",
            "path": f"{root_path}/CameraModel",
        },
        "rgb": rgb_info,
        "camera_image_roll": {
            "roll_deg": CAMERA_IMAGE_ROLL_DEG,
            "axis": "camera_local_z",
            "effect": "keeps forward unchanged; flips image up/right to remove upside-down rendering",
        },
        "camera_optical_offset": {
            "translation_xyz_m": [0.0, 0.0, CAMERA_OPTICAL_OFFSET_M],
            "axis": "camera_local_z",
            "effect": "moves RGB/Depth optical center to the front lens plane to avoid self-occlusion",
        },
        "depth": {
            **depth_info,
            "range_m": DEPTH_RANGE_M,
            "replicator_annotator": "distance_to_camera",
            "noise_model": {
                "enabled": True,
                "sigma_m": 0.003,
                "proportional_coeff": 0.0005,
                "invalid_ratio": 0.002,
            },
        },
        "camera_convention": {
            "usd_camera_forward_axis": "-Z",
            "opencv_depth_forward_axis": "+Z",
            "note": "Use the exported extrinsic matrix when converting Replicator camera data into OpenCV/ArUco frames.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-scene", default=DEFAULT_REFERENCE_SCENE)
    parser.add_argument("--target-scene", default=DEFAULT_TARGET_SCENE)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--forward-flip-deg", type=float, default=DEFAULT_FORWARD_FLIP_DEG)
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.target_scene)
    if stage is None:
        raise RuntimeError(f"Could not open target scene: {args.target_scene}")

    report = {
        "schema": "gemini2_wrist_camera_install/v1",
        "reference_scene_read_only": args.reference_scene,
        "target_scene": args.target_scene,
        "policy": "use only the manual Gemini2 pose relative to Link_6; do not import other reference-scene content",
        "forward_flip_deg": args.forward_flip_deg,
        "forward_flip_axis": "camera_local_y",
        "visual_model_source_usd": ORBBEC_GEMINI2_MODEL_USD,
        "visual_model_scale": CAMERA_MODEL_SCALE,
        "visual_model_forward_flip_deg": CAMERA_MODEL_FORWARD_FLIP_DEG,
        "visual_model_forward_flip_axis": "model_local_y",
        "camera_image_roll_deg": CAMERA_IMAGE_ROLL_DEG,
        "camera_image_roll_axis": "camera_local_z",
        "camera_optical_offset_xyz_m": [0.0, 0.0, CAMERA_OPTICAL_OFFSET_M],
        "sides": {},
    }
    for side in SIDES:
        rel = get_reference_gemini_rel_to_link6(args.reference_scene, side)
        report["sides"][side] = install_side(stage, side, rel, forward_flip_deg=args.forward_flip_deg)

    stage.GetRootLayer().Save()

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote scene {args.target_scene}")
    print(f"wrote report {report_path}")
    for side, info in report["sides"].items():
        print(side, info["rgb_camera_path"], info["depth_camera_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
