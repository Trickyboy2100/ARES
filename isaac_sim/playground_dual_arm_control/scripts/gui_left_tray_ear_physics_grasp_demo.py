#!/usr/bin/env python3
"""PhysX validation demo for left-gripper top/bottom grasp on the tray ear.

This is intentionally a physics probe, not a full arm-dynamics controller: two
kinematic rigid-body pad proxies follow the intended left gripper path, contact
the real dynamic tray ear, then lift. It verifies the tray collision/grasp
geometry before the same Cartesian path is wired to IK/cuRobo/JAKA execution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = "/home/andyee/isaacsim/playground/2026060721_curobo_task_clean.usd"
DEFAULT_TRAJECTORY = PLAYGROUND_ROOT / "runtime/tray_handoff_curobo_trajectory.json"
DEFAULT_REPORT = PLAYGROUND_ROOT / "reports/left_tray_ear_physics_grasp_validation.json"
DEFAULT_LOG = PLAYGROUND_ROOT / "logs/left_tray_ear_physics_grasp/motion_log.jsonl"

TRAY_PATH = "/World/Tray"
EAR_FRAME_PATHS = {
    "YPlusEar": "/World/Tray/GraspFrames/YPlusEar",
    "YMinusEar": "/World/Tray/GraspFrames/YMinusEar",
}
PROBE_ROOT = "/World/LeftGripperPhysicsProbe"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--tray-path", default=TRAY_PATH)
    parser.add_argument("--ear", choices=sorted(EAR_FRAME_PATHS), default="YPlusEar")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--hold-open", action="store_true")
    parser.add_argument("--lift-m", type=float, default=0.07)
    parser.add_argument("--carry-to-chest", action="store_true")
    parser.add_argument(
        "--no-grasp-lock",
        action="store_true",
        help="Disable the fixed grasp joint after pad closure. Useful for friction-only experiments.",
    )
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument(
        "--chest-tray-world-xyz",
        type=float,
        nargs=3,
        default=None,
        help="Override the desired tray center at the chest handoff pose.",
    )
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    return parser.parse_args()


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def gf_matrix_from_axes(center, size):
    from pxr import Gf

    # Gf matrices are authored in row-vector convention, so transpose the
    # column-vector matrix we use for straightforward world-aligned boxes.
    sx, sy, sz = size
    x, y, z = center
    values = [
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, sz, 0.0,
        x, y, z, 1.0,
    ]
    return Gf.Matrix4d(*values)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def lerp(a, b, t):
    return [float(ai) * (1.0 - t) + float(bi) * t for ai, bi in zip(a, b)]


def vec3_add(a, b):
    return [float(a[i]) + float(b[i]) for i in range(3)]


def vec3_sub(a, b):
    return [float(a[i]) - float(b[i]) for i in range(3)]


def vec3_len(a):
    return math.sqrt(sum(float(v) * float(v) for v in a))


def load_chest_target(args):
    if args.chest_tray_world_xyz is not None:
        return [float(v) for v in args.chest_tray_world_xyz], "cli"
    trajectory_path = Path(args.trajectory)
    if trajectory_path.exists():
        payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
        matrix = payload.get("calibration", {}).get("tray_handoff_world_matrix4x4_expected")
        if matrix:
            return [float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3])], str(trajectory_path)
    return [0.135, 0.598, 1.032], "default_fallback"


def xform_world_xyz(stage, path: str):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim: {path}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mat = cache.GetLocalToWorldTransform(prim)
    return [float(v) for v in mat.ExtractTranslation()]


def bbox_payload(stage, path: str):
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing prim: {path}")
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"]
    ).ComputeWorldBound(prim).ComputeAlignedBox()
    return {
        "min": [float(v) for v in bbox.GetMin()],
        "max": [float(v) for v in bbox.GetMax()],
    }


def bbox_center(bbox):
    return [(bbox["min"][i] + bbox["max"][i]) * 0.5 for i in range(3)]


def apply_once(api_cls, prim):
    if not prim.HasAPI(api_cls):
        api_cls.Apply(prim)
    return api_cls(prim)


def make_display_material(stage, path: str, color):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color[:3])
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(color[3])
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material(prim, material):
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def make_pad(stage, name: str, color):
    from pxr import UsdGeom, UsdPhysics

    path = f"{PROBE_ROOT}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    prim = cube.GetPrim()
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([color[:3]])
    cube.CreateDisplayOpacityAttr([color[3]])

    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "left_grasp_probe")

    collision = apply_once(UsdPhysics.CollisionAPI, prim)
    collision.CreateCollisionEnabledAttr().Set(True)
    rigid = apply_once(UsdPhysics.RigidBodyAPI, prim)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    apply_once(UsdPhysics.MassAPI, prim).CreateMassAttr().Set(0.08)
    return prim, op


def make_carrier(stage):
    from pxr import UsdGeom, UsdPhysics

    path = f"{PROBE_ROOT}/Carrier"
    cube = UsdGeom.Cube.Define(stage, path)
    prim = cube.GetPrim()
    cube.CreateSizeAttr(0.01)
    cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rigid = apply_once(UsdPhysics.RigidBodyAPI, prim)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    apply_once(UsdPhysics.MassAPI, prim).CreateMassAttr().Set(0.05)
    return prim, op


def ensure_probe(stage):
    from pxr import UsdGeom

    if stage.GetPrimAtPath(PROBE_ROOT):
        stage.RemovePrim(PROBE_ROOT)
    UsdGeom.Xform.Define(stage, PROBE_ROOT)
    upper_mat = make_display_material(stage, f"{PROBE_ROOT}/UpperPadMaterial", (0.95, 0.18, 0.12, 0.82))
    lower_mat = make_display_material(stage, f"{PROBE_ROOT}/LowerPadMaterial", (0.10, 0.45, 1.00, 0.82))
    upper_prim, upper_op = make_pad(stage, "UpperPad", (0.95, 0.18, 0.12, 0.82))
    lower_prim, lower_op = make_pad(stage, "LowerPad", (0.10, 0.45, 1.00, 0.82))
    carrier_prim, carrier_op = make_carrier(stage)
    bind_material(upper_prim, upper_mat)
    bind_material(lower_prim, lower_mat)
    return {
        "upper": {"prim": upper_prim, "op": upper_op},
        "lower": {"prim": lower_prim, "op": lower_op},
        "carrier": {"prim": carrier_prim, "op": carrier_op},
    }


def set_pad_pose(pads, upper_center, lower_center, pad_size):
    from pxr import Gf

    pads["upper"]["op"].Set(gf_matrix_from_axes(upper_center, pad_size))
    pads["lower"]["op"].Set(gf_matrix_from_axes(lower_center, pad_size))
    carrier = [
        (float(upper_center[i]) + float(lower_center[i])) * 0.5 for i in range(3)
    ]
    pads["carrier"]["op"].Set(Gf.Vec3d(*carrier))
    return carrier


def create_grasp_lock(stage, tray_path: str, carrier_path: str):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    joint_path = f"{PROBE_ROOT}/GraspFixedJoint"
    if stage.GetPrimAtPath(joint_path):
        return joint_path
    tray = stage.GetPrimAtPath(tray_path)
    carrier = stage.GetPrimAtPath(carrier_path)
    if not tray or not carrier:
        raise RuntimeError("Cannot create grasp lock: missing tray or carrier")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    tray_world = cache.GetLocalToWorldTransform(tray)
    carrier_world = cache.GetLocalToWorldTransform(carrier)
    anchor = carrier_world.ExtractTranslation()

    local0 = carrier_world.GetInverse().Transform(anchor)
    local1 = tray_world.GetInverse().Transform(anchor)
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(carrier_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tray_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(float(local0[0]), float(local0[1]), float(local0[2])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(local1[0]), float(local1[1]), float(local1[2])))
    tray_rot_inv = tray_world.ExtractRotationQuat().GetInverse().GetNormalized()
    tray_rot_inv_i = tray_rot_inv.GetImaginary()
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(
            float(tray_rot_inv.GetReal()),
            float(tray_rot_inv_i[0]),
            float(tray_rot_inv_i[1]),
            float(tray_rot_inv_i[2]),
        )
    )
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateCollisionEnabledAttr().Set(False)
    return joint_path


def set_collision_enabled_recursive(stage, root_path: str, enabled: bool):
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root:
        return []
    touched = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(enabled)
            touched.append(str(prim.GetPath()))
    return touched


def maybe_hide_probe_robot_visual_collision(stage):
    # The imported gripper collision meshes are not yet tuned for this tray ear
    # test; keep the visual robot intact but leave physical contact to the two
    # well-controlled pad proxies.
    return []


def phase_pose(t: float, ear_center, lift_m: float, chest_pad_base=None):
    pad_size = [0.12, 0.028, 0.012]
    open_gap = 0.092
    closed_gap = 0.039
    pre_y = 0.125
    contact_y = 0.012

    if t < 0.8:
        phase = "pregrasp_open"
        a = 0.0
    elif t < 2.1:
        phase = "approach_minus_world_y"
        a = ease((t - 0.8) / 1.3)
    elif t < 3.0:
        phase = "close_top_bottom_on_ear"
        a = 1.0
    elif t < 4.4:
        phase = "lift_with_closed_pads"
        a = 1.0
    elif chest_pad_base is not None and t < 7.4:
        phase = "carry_to_chest_handoff"
        a = 1.0
    elif chest_pad_base is not None:
        phase = "hold_at_chest_handoff"
        a = 1.0
    elif t < 5.4:
        phase = "hold_lifted"
        a = 1.0
    else:
        phase = "settled_hold"
        a = 1.0

    y_offset = pre_y * (1.0 - a) + contact_y * a
    if phase == "close_top_bottom_on_ear":
        gap = open_gap * (1.0 - ease((t - 2.1) / 0.9)) + closed_gap * ease((t - 2.1) / 0.9)
        lift = 0.0
    elif phase in {"lift_with_closed_pads", "hold_lifted", "settled_hold"}:
        gap = closed_gap
        lift = lift_m * ease((min(t, 4.4) - 3.0) / 1.4)
    elif phase == "carry_to_chest_handoff":
        gap = closed_gap
        lift = lift_m
    else:
        gap = open_gap
        lift = 0.0

    base = [ear_center[0], ear_center[1] + y_offset, ear_center[2] + lift]
    if phase == "carry_to_chest_handoff":
        base = lerp(base, chest_pad_base, ease((t - 4.4) / 3.0))
    elif phase == "hold_at_chest_handoff":
        base = list(chest_pad_base)
    upper = [base[0], base[1], base[2] + gap * 0.5]
    lower = [base[0], base[1], base[2] - gap * 0.5]
    return phase, upper, lower, pad_size, gap, y_offset, lift


def run_demo(args, app):
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdPhysics

    ctx = omni.usd.get_context()
    print(f"[LEFT-GRASP] Opening scene: {args.scene}", flush=True)
    ctx.open_stage(args.scene)
    for _ in range(120):
        app.update()
        time.sleep(0.002)

    stage = ctx.get_stage()
    print("[LEFT-GRASP] Stage opened.", flush=True)
    tray = stage.GetPrimAtPath(args.tray_path)
    if not tray:
        raise RuntimeError(f"Tray not found: {args.tray_path}")
    if not tray.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Tray is not a rigid body: {args.tray_path}")
    ear_path = EAR_FRAME_PATHS[args.ear]
    if not stage.GetPrimAtPath(ear_path):
        raise RuntimeError(f"Missing grasp frame: {ear_path}")

    print("[LEFT-GRASP] Creating probe pads/carrier.", flush=True)
    pads = ensure_probe(stage)
    print("[LEFT-GRASP] Probe ready.", flush=True)
    maybe_hide_probe_robot_visual_collision(stage)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    # Keep the probe away while the tray settles onto the loading equipment.
    set_pad_pose(pads, [0.0, 2.0, 2.0], [0.0, 2.0, 1.9], [0.02, 0.02, 0.02])
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_current_time(0.0)
    timeline.play()
    print("[LEFT-GRASP] Settling tray.", flush=True)
    for _ in range(int(args.settle_sec / args.dt)):
        app.update()

    settled_bbox = bbox_payload(stage, args.tray_path)
    print("[LEFT-GRASP] Settled; entering grasp loop.", flush=True)
    settled_center = bbox_center(settled_bbox)
    ear_center = xform_world_xyz(stage, ear_path)
    # The grasp frame is authored relative to the tray, so after PhysX settling
    # it follows the tray body. If USD evaluation lags, snap to bbox-derived
    # height while preserving the intended Y-end and X center.
    if abs(ear_center[2] - settled_center[2]) > 0.20:
        ear_center[2] = settled_center[2]

    chest_target, chest_target_source = load_chest_target(args)
    ear_to_tray_center = vec3_sub(ear_center, settled_center)
    chest_ear_center = vec3_add(chest_target, ear_to_tray_center)
    chest_pad_base = None

    samples = []
    total_t = 9.2 if args.carry_to_chest else 5.8
    steps = int(total_t / args.dt)
    grasp_lock_path = None
    for i in range(steps):
        t = i * args.dt
        if args.carry_to_chest and chest_pad_base is None and t >= 4.4:
            phase_now, upper_now, lower_now, pad_size_now, _, _, _ = phase_pose(
                4.4, ear_center, args.lift_m, None
            )
            carrier_now = set_pad_pose(pads, upper_now, lower_now, pad_size_now)
            app.update()
            tray_center_now = bbox_center(bbox_payload(stage, args.tray_path))
            tray_from_carrier = vec3_sub(tray_center_now, carrier_now)
            chest_pad_base = vec3_sub(chest_target, tray_from_carrier)
            print(
                "[LEFT-GRASP] Runtime chest carrier target: "
                f"{[round(v, 6) for v in chest_pad_base]} "
                f"from tray_from_carrier={[round(v, 6) for v in tray_from_carrier]}",
                flush=True,
            )
        phase, upper, lower, pad_size, gap, y_offset, lift = phase_pose(
            t, ear_center, args.lift_m, chest_pad_base
        )
        carrier = set_pad_pose(pads, upper, lower, pad_size)
        if (
            args.carry_to_chest
            and not args.no_grasp_lock
            and grasp_lock_path is None
            and t >= 3.05
        ):
            print("[LEFT-GRASP] Creating grasp fixed joint.", flush=True)
            grasp_lock_path = create_grasp_lock(
                stage, args.tray_path, str(pads["carrier"]["prim"].GetPath())
            )
            print(f"[LEFT-GRASP] Grasp lock active: {grasp_lock_path}", flush=True)
        app.update()
        if i % 6 == 0 or i == steps - 1:
            tray_bbox = bbox_payload(stage, args.tray_path)
            tray_xyz = xform_world_xyz(stage, args.tray_path)
            row = {
                "time_sec": round(t, 4),
                "phase": phase,
                "ear_target_world_xyz": [round(float(v), 6) for v in ear_center],
                "upper_pad_world_xyz": [round(float(v), 6) for v in upper],
                "lower_pad_world_xyz": [round(float(v), 6) for v in lower],
                "carrier_world_xyz": [round(float(v), 6) for v in carrier],
                "pad_gap_m": round(float(gap), 6),
                "approach_y_offset_m": round(float(y_offset), 6),
                "commanded_lift_m": round(float(lift), 6),
                "chest_tray_target_world_xyz": [round(float(v), 6) for v in chest_target],
                "tray_world_translation_xyz": [round(float(v), 6) for v in tray_xyz],
                "tray_bbox": tray_bbox,
                "tray_bbox_center_xyz": [round(float(v), 6) for v in bbox_center(tray_bbox)],
            }
            samples.append(row)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    timeline.stop()
    final_bbox = bbox_payload(stage, args.tray_path)
    final_center = bbox_center(final_bbox)
    lift_by_bbox_min = final_bbox["min"][2] - settled_bbox["min"][2]
    lift_by_center = final_center[2] - settled_center[2]
    chest_error_m = vec3_len(vec3_sub(final_center, chest_target)) if args.carry_to_chest else None
    pad_lower_final = samples[-1]["lower_pad_world_xyz"]
    report = {
        "schema": "left_tray_ear_physics_grasp_validation/v1",
        "scene": args.scene,
        "tray_path": args.tray_path,
        "ear": args.ear,
        "ear_grasp_frame": ear_path,
        "method": {
            "approach_axis": "world Y; pregrasp at +Y, motion command toward -Y",
            "clamp_axis": "world Z; one kinematic pad above and one below the ear",
            "physics_contact": "tray is a dynamic rigid body; pad proxies are kinematic rigid bodies with collision",
            "scope_note": "This validates grasp geometry/PhysX contact. The full robot arm is visual/static in this probe and will be driven by IK/cuRobo/JAKA commands next.",
        },
        "carry_to_chest": {
            "enabled": bool(args.carry_to_chest),
            "target_source": chest_target_source,
            "desired_tray_center_world_xyz": [round(float(v), 6) for v in chest_target],
            "derived_ear_center_target_world_xyz": [round(float(v), 6) for v in chest_ear_center],
            "derived_pad_base_target_world_xyz": [round(float(v), 6) for v in chest_pad_base] if chest_pad_base else None,
            "grasp_lock_enabled": bool(args.carry_to_chest and not args.no_grasp_lock),
            "grasp_lock_path": grasp_lock_path,
        },
        "settled_before_grasp": {
            "bbox": settled_bbox,
            "bbox_center_xyz": [round(float(v), 6) for v in settled_center],
            "ear_center_world_xyz": [round(float(v), 6) for v in ear_center],
        },
        "final": {
            "bbox": final_bbox,
            "bbox_center_xyz": [round(float(v), 6) for v in final_center],
            "lift_by_bbox_min_m": float(lift_by_bbox_min),
            "lift_by_bbox_center_m": float(lift_by_center),
            "chest_center_error_m": float(chest_error_m) if chest_error_m is not None else None,
            "lower_pad_final_distance_to_tray_center_m": float(vec3_len(vec3_sub(pad_lower_final, final_center))),
        },
        "thresholds": {
            "min_required_lift_by_bbox_min_m": 0.025,
            "commanded_lift_m": args.lift_m,
            "max_chest_center_error_m": 0.12 if args.carry_to_chest else None,
        },
        "status": "pass"
        if lift_by_bbox_min > 0.025 and (chest_error_m is None or chest_error_m < 0.12)
        else "fail",
        "samples": samples,
        "log_path": str(log_path),
    }
    atomic_write_json(Path(args.report), report)
    print(json.dumps(report["final"], ensure_ascii=False, indent=2), flush=True)
    print(f"[LEFT-GRASP] status={report['status']} report={args.report}", flush=True)

    if args.hold_open:
        timeline.play()
        while app.is_running():
            app.update()
            time.sleep(0.01)
    return report


def main():
    args = parse_args()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1440,
            "height": 900,
            "renderer": "RaytracedLighting",
        }
    )
    try:
        run_demo(args, app)
    finally:
        if not args.hold_open:
            app.close()


if __name__ == "__main__":
    main()
