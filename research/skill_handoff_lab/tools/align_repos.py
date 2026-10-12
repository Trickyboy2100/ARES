#!/usr/bin/env python3
"""Offline ARES vs ARES-R geometry/contract parity audit. No hardware access.

Pass here means source assets and static contracts match. It does NOT mean
Isaac USD physical frames, live controller TCP, contact physics, or the latest
site worktree have been commissioned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head(path):
    try:
        p = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return p.stdout.strip() if p.returncode == 0 else "NOT_A_GIT_CHECKOUT"
    except (OSError, subprocess.TimeoutExpired):
        return "GIT_UNAVAILABLE"


def joint_limits(urdf):
    root = ET.parse(str(urdf)).getroot()
    results = {}
    for i in range(1, 7):
        j = root.find("joint[@name='joint%d']" % i)
        if j is None:
            raise ValueError("joint%d missing in ARES URDF" % i)
        axis = j.find("axis")
        limit = j.find("limit")
        if limit is None or axis is None:
            raise ValueError("joint%d axis or limits missing" % i)
        results["joint%d" % i] = {
            "axis": tuple(float(x) for x in axis.get("xyz").split()),
            "lower": float(limit.get("lower")),
            "upper": float(limit.get("upper")),
        }
    return results


def soft_limit_overlap(urdf_limits, site_limits):
    margin = float(site_limits.get("soft_limit_margin_rad", 0.0))
    result = {}
    ok = True
    for i, name in enumerate(site_limits["joint_names"]):
        u = urdf_limits[name]
        lo = max(u["lower"], float(site_limits["lower_rad"][i]) + margin)
        hi = min(u["upper"], float(site_limits["upper_rad"][i]) - margin)
        result[name] = {"sample_lower_rad": lo, "sample_upper_rad": hi,
                        "overlap": lo < hi}
        ok &= lo < hi
    return ok, result


def compare_assets(ares_robot_dir, collision_model):
    checks = []
    for rel_path, expected in sorted(collision_model["asset_sha256"].items()):
        f = ares_robot_dir / rel_path
        observed = sha256(f) if f.is_file() else None
        checks.append({"path": rel_path, "exists": f.is_file(),
                       "expected_sha256": expected,
                       "actual_sha256": observed,
                       "match": observed == expected})
    return checks


def audit(ares, ares_r):
    simforge = ares / "isaac_sim/simforge"
    robot = simforge / "robot"
    conf = ares_r / "config"
    world = read_json(conf / "robot_world.json")
    collision = read_json(conf / "robot_collision_model.json")
    site = read_json(conf / "jaka_mini2_motion.site.json")
    system = read_json(conf / "system.json")
    poses = read_json(conf / "named_poses.json")
    urdf = robot / collision["urdf"]
    if not urdf.exists():
        raise FileNotFoundError(str(urdf))
    limits = joint_limits(urdf)
    limit_ok, limit_report = soft_limit_overlap(limits, site)
    checks = compare_assets(robot, collision)
    base_l = world["arms"]["left"]
    base_r = world["arms"]["right"]
    left_y = float(base_l["base_xyz_m"][1])
    right_y = float(base_r["base_xyz_m"][1])
    mounted = left_y > 0 and right_y < 0 and abs(left_y+right_y) < 1e-5
    relevant_profiles = {key: {
        "tool_revision": p.get("tool_revision"),
        "calibration_revision": p.get("calibration_revision"),
        "orientation_convention": p.get("orientation_convention"),
        "state": p.get("state"),
    } for key, p in system["epic"]["task_profiles"].items()
      if key in ("right_pick", "right_place_rightmost")}
    simcfg = robot / "jaka_minicobo_curobo.yml"
    yaml_text = simcfg.read_text(encoding="utf-8") if simcfg.exists() else ""
    absolute_path_hazard = "/home/" in yaml_text and (
        "asset_root_path: /home/" in yaml_text or "urdf_path: /home/" in yaml_text)
    if len(system.get("epic_pointcloud", {}).get("T_body_camera") or []) == 4:
        camera_known = True
    else:
        camera_known = False
    source_sha = collision.get("asset_revision", "")
    outcome = {
        "schema_version": 1,
        "audit_kind": "STATIC_MODEL_ALIGNMENT_ONLY",
        "execution_allowed": False,
        "live_hardware_verified": False,
        "input_commits": {"ARES": git_head(ares), "ARES_R": git_head(ares_r)},
        "ares_simforge_robot_root": str(robot.resolve()),
        "ares_r_pinned_asset_revision": source_sha,
        "ares_r_site_asset_root_note": collision.get("asset_root"),
        "model_asset_checks": checks,
        "asset_hash_parity_pass": len(checks) > 0 and all(c["match"] for c in checks),
        "joint_limit_contract": {"site_limits_and_urdf_intersect": limit_ok,
                                 "joint_sampling_intersection": limit_report},
        "frame_contract": {
            "body_axes": world["frame"]["axes"],
            "left_base_xyz_m": base_l["base_xyz_m"],
            "right_base_xyz_m": base_r["base_xyz_m"],
            "left_base_rpy_rad": base_l["base_rpy_rad"],
            "right_base_rpy_rad": base_r["base_rpy_rad"],
            "left_right_signs_valid": mounted,
            "ares_usd_world_to_body": "UNVERIFIED_REQUIRES_STAGE_AUDIT",
        },
        "tool_contract": {"profiles": relevant_profiles,
                          "current_controller_T_link6_tcp": "NOT_IN_STATIC_FILE_NEEDS_READONLY_EXPORT",
                          "sim_tcp_matches_live_controller": False},
        "camera_contract": {"real_T_body_camera_is_recorded": camera_known,
                            "sim_camera_mount_alignment": "NOT_TESTED",
                            "use_live_camera_to_calibrate_sim_directly": False},
        "planning_contract": {
            "old_curobo_yaml_has_nonportable_absolute_paths": absolute_path_hazard,
            "site_motion_soft_limit_margin_rad": site["soft_limit_margin_rad"],
            "legacy_sim_fixed_midline_rule": "DO_NOT_INHERIT_AS_REAL_SITE_COLLISION",
            "physical_collision_protections_must_remain": True,
        },
        "replay_policy": {
            "robot_commands": "DISABLED",
            "old_site_snapshots": "NOT_ACTIVE_AT_CURRENT_BASE_POSE",
            "actual_october_site_config": "REIMPORT_AND_DIFF_REQUIRED",
            "pose_center": "CARTESIAN_TCP_GOAL_NOT_A_FIXED_JOINT_STATE",
            "attachment": "CHECK_PLANNER_AND_SCENE_REVISIONS",
        },
        "status": "STATIC_ASSETS_PASS" if all(c["match"] for c in checks)
                  and limit_ok and mounted else "STATIC_ALIGNMENT_BLOCKED",
        "remaining": [
            "Audit ARES USD world/body/arm base transformations independently",
            "Record current site controller tool/TCP SE(3) (read-only, with revision)",
            "Verify object/attached collision, soft limits and horizontal TCP convention",
            "Re-import recent 2026-10 site changes; September GitHub branch is not the live state",
            "Build a generated, local-only cuRobo YAML with portable asset paths",
        ]
    }
    return outcome


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ares", type=Path, required=True)
    p.add_argument("--ares-r", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    report = audit(args.ares.resolve(), args.ares_r.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(report["status"])
    print("MODEL_FILES_MATCHING", sum(c["match"] for c in report["model_asset_checks"]),
          "/", len(report["model_asset_checks"]))
    print("SITE_SOFT_LIMIT_INTERSECTION", report["joint_limit_contract"]["site_limits_and_urdf_intersect"])
    print("STILL_UNVERIFIED: simulated USD mount, live controller TCP, recent site changes")
    return 0 if report["status"] == "STATIC_ASSETS_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
