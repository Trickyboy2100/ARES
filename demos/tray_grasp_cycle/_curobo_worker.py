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


def curobo_tool_pose(chain, q_arm: np.ndarray):
    qmap = dict(zip(CUROBO_JOINTS, q_arm.tolist()))
    qmap["4C2_Joint1"] = 0.0
    T = fk(chain, qmap)
    return T[:3, 3], matrix_to_quat_wxyz(T[:3, :3])


def main():
    batch = json.load(sys.stdin)
    jobs = batch["jobs"]

    planner = init_curobo_planner()
    chain = chain_to_link(load_joints(Path(DEFAULT_CUROBO_URDF)), "base_link", "4C2_Link5")

    results = []
    for job in jobs:
        label = job.get("label", "?")
        q_start = np.array(job["q_start"], dtype=float)
        q_goal  = np.array(job["q_goal"],  dtype=float)
        max_attempts = job.get("max_attempts", 8)

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
                print(f"[worker] {label}: {len(path)} steps", file=sys.stderr, flush=True)
            else:
                print(f"[worker] {label}: planner returned None", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[worker] {label} error: {e}", file=sys.stderr, flush=True)

        results.append({"label": label, "path": path})

    json.dump({"results": results}, sys.stdout)


if __name__ == "__main__":
    main()
