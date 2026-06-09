#!/usr/bin/env python3
"""GUI scene with optical-axis arrows at both Gemini2 wrist cameras.

Opens the curobo task scene in GUI mode, adds a colored arrow along the
-Z axis of each RgbCamera prim (left = red, right = blue), captures RGB-D
images via Replicator, saves a viewport screenshot to the validation directory,
then keeps the GUI open for visual inspection. Press Ctrl-C to quit.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import cv2
import numpy as np


SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
OUT_DIR = Path(__file__).resolve().parents[1] / "reports/gemini2_rgbd_validation"

CAMERA_RIGS = {
    "left": {
        "rgb_path":   "/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim/RgbCamera",
        "depth_path": "/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim/DepthCamera",
        "color": (1.0, 0.15, 0.0),   # red-orange
    },
    "right": {
        "rgb_path":   "/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim/RgbCamera",
        "depth_path": "/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim/DepthCamera",
        "color": (0.0, 0.35, 1.0),   # blue
    },
}

# Arrow dimensions (metres)
SHAFT_LEN = 0.20
SHAFT_R   = 0.005
HEAD_H    = 0.06
HEAD_R    = 0.016

RGB_W,   RGB_H   = 640, 360
DEPTH_W, DEPTH_H = 640, 400
RT_SUBFRAMES     = 16


# ─── Arrow geometry ────────────────────────────────────────────────────────────

def _add_optical_axis_arrow(stage, cam_path: str, color: tuple[float, float, float]):
    """Add cylinder+cone arrow child to *cam_path* pointing along its local -Z."""
    from pxr import Gf, UsdGeom, Vt

    root_path = f"{cam_path}/OpticalAxisArrow"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)

    col = Vt.Vec3fArray([Gf.Vec3f(*color)])

    # Shaft: cylinder axis=Z, runs from z=0 to z=-SHAFT_LEN
    shaft = UsdGeom.Cylinder.Define(stage, f"{root_path}/Shaft")
    shaft.GetAxisAttr().Set("Z")
    shaft.GetHeightAttr().Set(SHAFT_LEN)
    shaft.GetRadiusAttr().Set(SHAFT_R)
    shaft_xf = UsdGeom.Xformable(shaft.GetPrim())
    shaft_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -SHAFT_LEN / 2.0))
    shaft.GetDisplayColorAttr().Set(col)

    # Head: cone, tip at z=-(SHAFT_LEN+HEAD_H), base at z=-SHAFT_LEN.
    #
    # UsdGeom.Cone axis=Z has tip at local +HEAD_H/2 and base at -HEAD_H/2.
    # We need tip on the -Z side, so compose: T(0,0,-c) * RotX(180°)
    # where c = SHAFT_LEN + HEAD_H/2.
    # Applied to cone tip (0,0,+HEAD_H/2):
    #   RotX(180°) → (0, 0, -HEAD_H/2)
    #   + T → (0, 0, -(SHAFT_LEN+HEAD_H)) ✓  (furthest from camera)
    # Applied to cone base (0,0,-HEAD_H/2):
    #   RotX(180°) → (0, 0, +HEAD_H/2)
    #   + T → (0, 0, -SHAFT_LEN) ✓  (joins shaft)
    head = UsdGeom.Cone.Define(stage, f"{root_path}/Head")
    head.GetAxisAttr().Set("Z")
    head.GetHeightAttr().Set(HEAD_H)
    head.GetRadiusAttr().Set(HEAD_R)
    head_xf = UsdGeom.Xformable(head.GetPrim())
    # USD composes ops left-to-right: M = op0 * op1 → p_final = op0 * (op1 * p)
    head_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -(SHAFT_LEN + HEAD_H / 2.0)))
    head_xf.AddRotateXOp().Set(180.0)
    head.GetDisplayColorAttr().Set(col)


# ─── Image save helpers (same as capture_gemini2_rgbd_validation.py) ──────────

def _save_rgb(path: Path, rgba) -> dict:
    arr = np.asarray(rgba)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(np.uint8)
    else:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return {
        "path": str(path), "shape": list(rgb.shape),
        "min": int(rgb.min()), "max": int(rgb.max()),
        "mean": float(rgb.mean()), "std": float(rgb.std()),
        "nonzero_ratio": float(np.count_nonzero(rgb) / max(1, rgb.size)),
    }


def _save_depth(prefix: Path, depth) -> dict:
    arr = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(arr) & (arr > 0.0)
    np.save(prefix.with_suffix(".npy"), arr)
    vis = np.zeros((*arr.shape, 3), dtype=np.uint8)
    if np.any(valid):
        lo = float(np.percentile(arr[valid], 2.0))
        hi = float(np.percentile(arr[valid], 98.0))
        if hi <= lo:
            hi = lo + 1e-3
        gray = (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
        vis = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        vis[~valid] = 0
    cv2.imwrite(str(prefix.with_suffix(".png")), vis)
    return {
        "npy_path": str(prefix.with_suffix(".npy")),
        "visualization_path": str(prefix.with_suffix(".png")),
        "shape": list(arr.shape),
        "valid_ratio": float(np.count_nonzero(valid) / max(1, arr.size)),
        "min_m": float(arr[valid].min()) if np.any(valid) else None,
        "max_m": float(arr[valid].max()) if np.any(valid) else None,
        "mean_m": float(arr[valid].mean()) if np.any(valid) else None,
        "std_m": float(arr[valid].std()) if np.any(valid) else None,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp(
        launch_config={"renderer": "RaytracedLighting", "headless": False}
    )

    import omni.kit.app
    import omni.replicator.core as rep
    import omni.usd
    from pxr import UsdGeom

    keep_running = True

    def _stop(_s, _f):
        nonlocal keep_running
        keep_running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    app = omni.kit.app.get_app()
    ctx = omni.usd.get_context()

    print(f"[INFO] Opening scene: {SCENE}")
    ctx.open_stage(SCENE)

    # Wait for both camera prims to be valid
    stage = None
    for _ in range(240):
        app.update()
        stage = ctx.get_stage()
        if stage and stage.GetPrimAtPath(
            CAMERA_RIGS["left"]["rgb_path"]
        ).IsValid() and stage.GetPrimAtPath(
            CAMERA_RIGS["right"]["rgb_path"]
        ).IsValid():
            break
        time.sleep(0.05)
    if stage is None:
        raise RuntimeError("Stage did not load within timeout")

    # Inject arrows
    for side, info in CAMERA_RIGS.items():
        _add_optical_axis_arrow(stage, info["rgb_path"], info["color"])
        print(f"[INFO] {side} arrow → {info['rgb_path']}/OpticalAxisArrow  (color={info['color']})")

    # Let the viewport settle so the arrows are visible before screenshot
    for _ in range(60):
        app.update()
        time.sleep(0.016)

    # Viewport screenshot
    screenshot_path = out_dir / "viewport_camera_arrows.png"
    try:
        import omni.renderer_capture
        rc = omni.renderer_capture.acquire_renderer_capture_interface()
        rc.capture_next_frame_swapchain(str(screenshot_path))
        app.update()
        print(f"[INFO] Viewport screenshot → {screenshot_path}")
    except Exception as exc:
        print(f"[WARN] Viewport screenshot unavailable: {exc}")

    # ── RGB-D capture via Replicator ─────────────────────────────────────────
    products    = {}
    annotators  = {}
    for side, info in CAMERA_RIGS.items():
        for kind in ("rgb", "depth"):
            cam_path = info["rgb_path"] if kind == "rgb" else info["depth_path"]
            prim = stage.GetPrimAtPath(cam_path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
                print(f"[WARN] Missing {side} {kind} camera: {cam_path}")
                continue
            res = (RGB_W, RGB_H) if kind == "rgb" else (DEPTH_W, DEPTH_H)
            rp = rep.create.render_product(cam_path, res)
            annot_name = "rgb" if kind == "rgb" else "distance_to_camera"
            annot = rep.AnnotatorRegistry.get_annotator(annot_name, device="cpu")
            annot.attach([rp])
            products[(side, kind)]   = rp
            annotators[(side, kind)] = annot

    rep.orchestrator.preview()
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, delta_time=0.0, pause_timeline=False)
        app.update()
        time.sleep(0.05)

    summary: dict = {
        "schema": "gemini2_rgbd_validation/v2_arrow_check",
        "scene": SCENE,
        "arrows": {
            side: {
                "camera_path": info["rgb_path"],
                "arrow_prim": f"{info['rgb_path']}/OpticalAxisArrow",
                "axis_direction": "-Z (camera local frame)",
                "color_rgb": info["color"],
                "shaft_len_m": SHAFT_LEN,
                "head_h_m": HEAD_H,
            }
            for side, info in CAMERA_RIGS.items()
        },
        "outputs": {},
    }

    for side in ("left", "right"):
        summary["outputs"][side] = {}
        if (side, "rgb") in annotators:
            rgb_data = annotators[(side, "rgb")].get_data()
            summary["outputs"][side]["rgb"] = _save_rgb(
                out_dir / f"{side}_rgb.png", rgb_data
            )
        if (side, "depth") in annotators:
            depth_data = annotators[(side, "depth")].get_data()
            summary["outputs"][side]["depth"] = _save_depth(
                out_dir / f"{side}_depth", depth_data
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Written {summary_path}")

    print("[INFO] GUI open — arrows visible in viewport. Press Ctrl-C to quit.")
    while keep_running:
        app.update()
        time.sleep(0.016)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
