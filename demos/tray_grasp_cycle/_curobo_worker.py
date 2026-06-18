#!/usr/bin/env python3
"""Standalone cuRobo path planner — called as a subprocess from demo.py.

Reads a JSON batch from stdin:
  {"jobs": [{"q_start": [...], "q_goal": [...], "label": "..."}, ...]}

Writes results to stdout as JSON:
  {"results": [{"label": "...", "path": [[...], ...]}, ...]}

Runs with miniconda python3 (which has warp 1.13 + curobo + torch),
completely outside Isaac Sim's Python environment.
"""
from __future__ import annotations
import json, sys, os, tempfile
from pathlib import Path

import numpy as np
import torch

# ── repo paths ────────────────────────────────────────────────────────────────
_SIMFORGE  = Path(__file__).resolve().parents[2]   # simforge/ (repo root)
_ROBOT_DIR = _SIMFORGE / "robot"
sys.path.insert(0, str(_SIMFORGE))
sys.path.insert(0, str(_SIMFORGE / "core"))

_CUROBO_ROOT = os.environ.get("CUROBO_ROOT")
if _CUROBO_ROOT and Path(_CUROBO_ROOT, "curobo").is_dir():
    # Keep this local to the subprocess so Isaac Sim's Python path is not
    # polluted by the external cuRobo checkout.
    sys.path.append(_CUROBO_ROOT)

# ── patch jaka_minicobo_curobo.yml to use bundled robot/ assets ───────────────
# The YAML ships with absolute paths from the developer's machine.  We rewrite
# asset_root_path and urdf_path to the bundled robot/ directory before cuRobo
# reads the file, so the planner works on any clone without manual edits.
def _make_patched_cfg() -> str:
    src = _ROBOT_DIR / "jaka_minicobo_curobo.yml"
    text = src.read_text()
    robot_str = str(_ROBOT_DIR)
    urdf_str  = str(_ROBOT_DIR / "jaka_minicobo_gripper.urdf")
    import re
    text = re.sub(r"asset_root_path:.*", f"asset_root_path: {robot_str}", text)
    text = re.sub(r"urdf_path:.*",       f"urdf_path: {urdf_str}",        text)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name

_PATCHED_CFG = _make_patched_cfg()

import planning as _planning_mod                   # noqa: E402
_planning_mod.CUROBO_CFG = _PATCHED_CFG            # override before first call

from kinematics_probe import (                     # noqa: E402
    CUROBO_JOINTS, DEFAULT_CUROBO_URDF,
    chain_to_link, load_joints, matrix_to_quat_wxyz, fk,
)
from planning import init_curobo_planner, make_goal  # noqa: E402


def _obstacle_key(obstacles):
    return json.dumps(obstacles or [], sort_keys=True, separators=(",", ":"))


def _planner_for(obstacles, cache):
    key = _obstacle_key(obstacles)
    planner = cache.get(key)
    if planner is None:
        planner = init_curobo_planner(obstacles or [])
        cache[key] = planner
    return planner


def curobo_tool_pose(chain, q_arm: np.ndarray):
    qmap = dict(zip(CUROBO_JOINTS, q_arm.tolist()))
    qmap["4C2_Joint1"] = 0.0
    T = fk(chain, qmap)
    return T[:3, 3], matrix_to_quat_wxyz(T[:3, :3])


def main():
    batch = json.load(sys.stdin)
    jobs = batch["jobs"]

    try:
        import yaml
        cfg = yaml.safe_load(Path(_PATCHED_CFG).read_text())
        spheres = cfg.get("kinematics", {}).get("collision_spheres", {})
        gripper_spheres = sum(len(spheres.get(name, []) or []) for name in (
            "4C2_baselink", "4C2_Link1", "4C2_Link2", "4C2_Link3",
            "4C2_Link4", "4C2_Link5", "4C2_Link6",
        ))
        print(f"[worker] gripper collision spheres: {gripper_spheres}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[worker] gripper collision sphere check failed: {e}", file=sys.stderr, flush=True)

    planner_cache = {}
    chain = chain_to_link(load_joints(Path(DEFAULT_CUROBO_URDF)), "base_link", "4C2_Link5")

    results = []
    for job in jobs:
        label = job.get("label", "?")
        q_start = np.array(job["q_start"], dtype=float)
        q_goal  = np.array(job["q_goal"],  dtype=float)
        max_attempts = job.get("max_attempts", 8)
        obstacles = job.get("obstacles") or []
        cuboid_count = sum(1 for obs in obstacles if obs.get("kind", "cuboid") != "mesh")
        mesh_count = sum(1 for obs in obstacles if obs.get("kind", "cuboid") == "mesh")
        print(
            f"[worker] {label}: obstacles cuboid={cuboid_count} mesh={mesh_count}",
            file=sys.stderr,
            flush=True,
        )
        planner = _planner_for(obstacles, planner_cache)

        path = None
        try:
            pos, quat = curobo_tool_pose(chain, q_goal)
            goal_pose = make_goal(pos, quat)
            q_full = np.r_[q_start, 0.0]
            from curobo.types import JointState
            cur = JointState.from_position(
                torch.tensor(q_full.reshape(1, -1), device="cuda:0", dtype=torch.float32),
                joint_names=planner.joint_names,
            )
            result = planner.plan_pose(goal_pose, cur, max_attempts=max_attempts)
            if result is not None:
                raw = result.get_interpolated_plan().position.cpu().numpy()[0, 0, :, :]
                path = raw[:, :6].tolist()
                first_err = float(np.linalg.norm(raw[0, :6] - q_start))
                last_err = float(np.linalg.norm(raw[-1, :6] - q_goal))
                max_step = float(np.max(np.linalg.norm(np.diff(raw[:, :6], axis=0), axis=1))) if raw.shape[0] > 1 else 0.0
                print(
                    f"[worker] {label}: {len(path)} steps "
                    f"first_err={first_err:.3f} last_err={last_err:.3f} max_step={max_step:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(f"[worker] {label}: planner returned None", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[worker] {label} error: {e}", file=sys.stderr, flush=True)

        results.append({"label": label, "path": path})

    json.dump({"results": results}, sys.stdout)


if __name__ == "__main__":
    main()
