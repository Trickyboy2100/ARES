"""Rewrite external USD asset paths in main.usd to use bundled ../assets/.

Run with Isaac Sim Python:
    ~/isaacsim/kit/python/bin/python3 scenes/rewrite_paths.py
"""

import sys
import os
import shutil

USD_LIB = None
for d in os.listdir(os.path.expanduser("~/isaacsim/extscache")):
    if d.startswith("omni.usd.libs") and "lx64" in d:
        USD_LIB = os.path.expanduser(f"~/isaacsim/extscache/{d}")
        break

if USD_LIB:
    sys.path.insert(0, USD_LIB)
    os.environ["LD_LIBRARY_PATH"] = (
        f"{USD_LIB}/bin:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )

from pxr import Sdf, UsdUtils  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_PLAYGROUND = os.path.expanduser(
    "~/isaacsim/playground/2026061100_main.usd"
)
DST_SCENE = os.path.join(HERE, "main.usd")

# Absolute resolved path → new relative path (from scenes/ to assets/)
_HOME = os.path.expanduser("~")
PATH_MAP = {
    # ── CAD models ──────────────────────────────────────────────────────────
    f"{_HOME}/Developer/PG-JY/isaac_sim/cad_assets/robot-striped.usd":
        "../assets/cad/robot-striped.usd",
    f"{_HOME}/Developer/PG-JY/isaac_sim/cad_assets/CAM_Mount.usd":
        "../assets/cad/CAM_Mount.usd",
    f"{_HOME}/Developer/PG-JY/isaac_sim/cad_assets/force_sensor.usd":
        "../assets/cad/force_sensor.usd",
    f"{_HOME}/Developer/PG-JY/isaac_sim/cad_assets/gripper_flange.usd":
        "../assets/cad/gripper_flange.usd",
    # ── JAKA MiniCobo ────────────────────────────────────────────────────────
    f"{_HOME}/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo/jaka_minicobo.usd":
        "../assets/jaka_minicobo/jaka_minicobo.usd",
    f"{_HOME}/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo/configuration/jaka_minicobo_base.usd":
        "../assets/jaka_minicobo/configuration/jaka_minicobo_base.usd",
    f"{_HOME}/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo/configuration/jaka_minicobo_physics.usd":
        "../assets/jaka_minicobo/configuration/jaka_minicobo_physics.usd",
    f"{_HOME}/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo/configuration/jaka_minicobo_robot.usd":
        "../assets/jaka_minicobo/configuration/jaka_minicobo_robot.usd",
    f"{_HOME}/Developer/PG-JY/jaka_ros2/src/jaka_description/urdf/jaka_minicobo/configuration/jaka_minicobo_sensor.usd":
        "../assets/jaka_minicobo/configuration/jaka_minicobo_sensor.usd",
    # ── Inspire EG2-4C2 gripper ──────────────────────────────────────────────
    f"{_HOME}/Developer/robot_usds/grippers/Inspire_EG2_4C2/Inspire_EG2_4C2.usd":
        "../assets/Inspire_EG2_4C2/Inspire_EG2_4C2.usd",
    f"{_HOME}/Developer/robot_usds/grippers/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_base.usd":
        "../assets/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_base.usd",
    f"{_HOME}/Developer/robot_usds/grippers/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_physics.usd":
        "../assets/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_physics.usd",
    f"{_HOME}/Developer/robot_usds/grippers/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_robot.usd":
        "../assets/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_robot.usd",
    f"{_HOME}/Developer/robot_usds/grippers/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_sensor.usd":
        "../assets/Inspire_EG2_4C2/configuration/Inspire_EG2_4C2_sensor.usd",
    # ── Playground configuration (scene environment + physics) ───────────────
    f"{_HOME}/isaacsim/playground/configuration/2026060721_base.usd":
        "../assets/playground_config/2026060721_base.usd",
    f"{_HOME}/isaacsim/playground/configuration/2026060721_physics.usd":
        "../assets/playground_config/2026060721_physics.usd",
    f"{_HOME}/isaacsim/playground/configuration/2026060721_robot.usd":
        "../assets/playground_config/2026060721_robot.usd",
    f"{_HOME}/isaacsim/playground/configuration/2026060721_sensor.usd":
        "../assets/playground_config/2026060721_sensor.usd",
    # ── Textures ─────────────────────────────────────────────────────────────
    f"{_HOME}/Developer/PG-JY/vision_pose_benchmark/assets/concrete_floor.png":
        "../assets/textures/concrete_floor.png",
    f"{_HOME}/isaacsim/playground/aruco_eval/two_cam/tag.png":
        "../assets/textures/tag.png",
}

def main():
    if not os.path.exists(SRC_PLAYGROUND):
        print(f"ERROR: source playground scene not found:\n  {SRC_PLAYGROUND}")
        sys.exit(1)

    print(f"Source : {SRC_PLAYGROUND}")
    print(f"Dest   : {DST_SCENE}")

    # Open source layer (its ComputeAbsolutePath resolves relative paths correctly)
    src_layer = Sdf.Layer.FindOrOpen(SRC_PLAYGROUND)
    if src_layer is None:
        print("ERROR: cannot open source layer")
        sys.exit(1)

    # Copy the source USD to destination
    shutil.copy2(SRC_PLAYGROUND, DST_SCENE)
    print(f"Copied {os.path.getsize(DST_SCENE):,} bytes → {DST_SCENE}")

    # Open the copy
    dst_layer = Sdf.Layer.FindOrOpen(DST_SCENE)
    if dst_layer is None:
        print("ERROR: cannot open destination layer")
        sys.exit(1)

    remapped = []
    skipped  = []

    def remap(raw_path: str) -> str:
        abs_path = src_layer.ComputeAbsolutePath(raw_path)
        if abs_path in PATH_MAP:
            new_rel = PATH_MAP[abs_path]
            remapped.append((raw_path, new_rel, abs_path))
            return new_rel
        skipped.append((raw_path, abs_path))
        return raw_path

    UsdUtils.ModifyAssetPaths(dst_layer, remap)
    dst_layer.Save()

    print(f"\n✓ Remapped {len(remapped)} paths:")
    for raw, new, abs_ in remapped:
        print(f"  {raw!r:60s} → {new}")

    if skipped:
        print(f"\n~ Kept {len(skipped)} paths unchanged:")
        for raw, abs_ in skipped:
            print(f"  {raw!r:60s}  [{abs_}]")

    print("\nDone — main.usd saved.")


if __name__ == "__main__":
    main()
