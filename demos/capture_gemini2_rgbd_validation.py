#!/usr/bin/env python3
"""Headless RGB-D validation capture for the two simulated Gemini2 wrist cameras."""


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
import time
from pathlib import Path

import cv2
import numpy as np
from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "reports/gemini2_rgbd_validation"))
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=360)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=400)
    parser.add_argument("--rt-subframes", type=int, default=16)
    return parser.parse_args()


def save_rgb(path: Path, rgba) -> dict:
    arr = np.asarray(rgba)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(np.uint8)
    else:
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return {
        "path": str(path),
        "shape": list(rgb.shape),
        "min": int(rgb.min()),
        "max": int(rgb.max()),
        "mean": float(rgb.mean()),
        "std": float(rgb.std()),
        "nonzero_ratio": float(np.count_nonzero(rgb) / max(1, rgb.size)),
    }


def save_depth(prefix: Path, depth) -> dict:
    arr = np.asarray(depth, dtype=np.float32).squeeze()
    finite = np.isfinite(arr)
    valid = finite & (arr > 0.0)
    npy = prefix.with_suffix(".npy")
    png = prefix.with_suffix(".png")
    np.save(npy, arr)
    vis = np.zeros((*arr.shape, 3), dtype=np.uint8)
    if np.any(valid):
        lo = float(np.percentile(arr[valid], 2.0))
        hi = float(np.percentile(arr[valid], 98.0))
        if hi <= lo:
            hi = lo + 1e-3
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        vis = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        vis[~valid] = (0, 0, 0)
    cv2.imwrite(str(png), vis)
    return {
        "npy_path": str(npy),
        "visualization_path": str(png),
        "shape": list(arr.shape),
        "valid_ratio": float(np.count_nonzero(valid) / max(1, arr.size)),
        "min_m": float(arr[valid].min()) if np.any(valid) else None,
        "max_m": float(arr[valid].max()) if np.any(valid) else None,
        "mean_m": float(arr[valid].mean()) if np.any(valid) else None,
        "std_m": float(arr[valid].std()) if np.any(valid) else None,
    }


def main() -> int:
    args = parse_args()
    simulation_app = SimulationApp(
        launch_config={
            "renderer": "RaytracedLighting",
            "headless": True,
        }
    )

    import omni.replicator.core as rep
    import omni.usd
    from pxr import UsdGeom

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omni.usd.get_context().open_stage(args.scene)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(f"Could not open stage: {args.scene}")

    camera_paths = {
        "left": {
            "rgb": "/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim/RgbCamera",
            "depth": "/World/robot/jaka_minicobo_left/Link_6/Gemini2Sim/DepthCamera",
        },
        "right": {
            "rgb": "/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim/RgbCamera",
            "depth": "/World/robot/jaka_minicobo_right/Link_6/Gemini2Sim/DepthCamera",
        },
    }

    products = {}
    annotators = {}
    for side, paths in camera_paths.items():
        for kind, cam_path in paths.items():
            prim = stage.GetPrimAtPath(cam_path)
            if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"Missing {side} {kind} camera: {cam_path}")
            res = (args.rgb_width, args.rgb_height) if kind == "rgb" else (args.depth_width, args.depth_height)
            rp = rep.create.render_product(cam_path, res)
            annot_name = "rgb" if kind == "rgb" else "distance_to_camera"
            annot = rep.AnnotatorRegistry.get_annotator(annot_name, device="cpu")
            annot.attach([rp])
            products[(side, kind)] = rp
            annotators[(side, kind)] = annot

    rep.orchestrator.preview()
    for _ in range(8):
        rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0, pause_timeline=False)
        simulation_app.update()
        time.sleep(0.05)

    summary = {
        "schema": "gemini2_rgbd_validation/v1",
        "scene": args.scene,
        "camera_paths": camera_paths,
        "outputs": {},
    }
    for side in ("left", "right"):
        rgb = annotators[(side, "rgb")].get_data()
        depth = annotators[(side, "depth")].get_data()
        summary["outputs"][side] = {
            "rgb": save_rgb(out_dir / f"{side}_rgb.png", rgb),
            "depth": save_depth(out_dir / f"{side}_depth", depth),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {summary_path}")
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
