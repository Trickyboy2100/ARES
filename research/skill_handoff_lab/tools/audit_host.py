#!/usr/bin/env python3
"""Read-only Ubuntu/RTX environment inventory. No Docker start, install, or robot I/O."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def command(argv, timeout=8):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout.strip()[:10000],
                "stderr": p.stderr.strip()[:1200]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"returncode": None, "error": type(e).__name__ + ": " + str(e)[:400]}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def os_info():
    info = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, val = line.split("=", 1)
                if key in ("ID", "VERSION_ID", "PRETTY_NAME"):
                    info[key] = val.strip('"')
    return info


def collect():
    home = Path.home()
    old = home / "Developer/PG-JY/jinyu_ros_pkg/nodes/simulation"
    sim_candidates = [home / "isaacsim", home / ".local/share/ov/pkg",
                      home / "Developer/PG-JY", home / "simforge"]
    disk = shutil.disk_usage(home)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "safety": {"hardware_io": False, "docker_started": False,
                   "driver_install": False, "robot_motion": False},
        "hostname": platform.node(),
        "os": os_info(),
        "kernel": platform.release(),
        "architecture": platform.machine(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages_in_current_python": {name: package_version(name) for name in (
            "numpy", "scipy", "torch", "mujoco", "gymnasium",
            "stable-baselines3", "sb3-contrib", "isaacsim", "curobo")},
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                               "--format=csv,noheader"]),
        "docker_version": command(["docker", "--version"]),
        "docker_isaac_images": command(["docker", "images", "--format",
                                       "{{.Repository}}:{{.Tag}}", "nvcr.io/nvidia/isaac-sim"]),
        "legacy_simulation": {"path": str(old), "exists": old.is_dir(),
                              "mujoco_script_exists": (old / "mujoco_dual_arm.py").is_file(),
                              "isaac_script_exists": (old / "isaac_dual_arm_scene.py").is_file()},
        "isaac_root_candidates": {str(x): x.exists() for x in sim_candidates},
        "disk_free_gb": round(disk.free / 10**9, 2),
        "notes": [
            "Audit only: no package installs and no model launch.",
            "Use the existing Isaac 4.5 Docker/MuJoCo path only if present and smoke-tested.",
            "A 10 GB RTX 3080 is below Isaac Sim 5.1 documented minimum 16 GB VRAM.",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    report = collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("AUDIT_COMPLETE; saved " + str(args.output))
    print(json.dumps({
        "gpu": report["nvidia_smi"].get("stdout"),
        "docker_isaac_images": report["docker_isaac_images"].get("stdout"),
        "legacy_simulation": report["legacy_simulation"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
