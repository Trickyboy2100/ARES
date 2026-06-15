"""simforge/config.py — centralised path configuration.

All robot URDF files, meshes, and cuRobo configs ship with this repo under
robot/.  Isaac Sim installation is the only external dependency.

Quick setup (only ISAACSIM_ROOT is usually needed):

    export ISAACSIM_ROOT=~/isaacsim     # Isaac Sim installation root
                                        # default: ~/isaacsim

Override any individual path if your layout differs:

    export SIMFORGE_SCENE=~/my_scene.usd
    export SIMFORGE_URDF_DIR=~/robot_urdf/
    export SIMFORGE_CUROBO_CFG=~/robot/jaka_minicobo_curobo.yml

Run  python config.py  to verify all paths resolve correctly.
"""

import os
from pathlib import Path

# ── Repo root: this file lives at <repo_root>/config.py ──────────────────────
_SIMFORGE = Path(__file__).resolve().parent   # repo root (simforge/)
_ROBOT    = _SIMFORGE / "robot"               # bundled URDF / mesh assets

# ── Isaac Sim installation ────────────────────────────────────────────────────
ISAACSIM_ROOT = Path(os.environ.get("ISAACSIM_ROOT", str(Path.home() / "isaacsim")))
ISAACSIM_SH   = ISAACSIM_ROOT / "isaac-sim.sh"

# ── Main scene USD ────────────────────────────────────────────────────────────
_SCENE_CANDIDATES = [
    os.environ.get("SIMFORGE_SCENE", ""),
    str(_SIMFORGE / "scenes" / "main.usd"),
    str(ISAACSIM_ROOT / "playground" / "2026061100_main.usd"),
]
SCENE_USD = next(
    (p for p in _SCENE_CANDIDATES if p and Path(p).exists()),
    str(_SIMFORGE / "scenes" / "main.usd"),
)

# ── Robot URDF files ──────────────────────────────────────────────────────────
# Bundled under robot/ — override with SIMFORGE_URDF_DIR if needed.
URDF_DIR    = Path(os.environ.get("SIMFORGE_URDF_DIR", str(_ROBOT)))
ARM_URDF    = URDF_DIR / "jaka_minicobo.urdf"
CUROBO_URDF = URDF_DIR / "jaka_minicobo_gripper.urdf"
SIM_URDF    = CUROBO_URDF   # alias kept for backward compat

# ── cuRobo config ─────────────────────────────────────────────────────────────
CUROBO_CFG = Path(os.environ.get(
    "SIMFORGE_CUROBO_CFG",
    str(_ROBOT / "jaka_minicobo_curobo.yml"),
))


def check() -> list[str]:
    """Return a list of missing/broken path descriptions (empty = all OK)."""
    issues = []
    if not ISAACSIM_SH.exists():
        issues.append(
            f"Isaac Sim launcher not found: {ISAACSIM_SH}\n"
            f"  → set ISAACSIM_ROOT to your Isaac Sim installation directory"
        )
    if not Path(SCENE_USD).exists():
        issues.append(
            f"Scene USD not found: {SCENE_USD}\n"
            f"  → override with  export SIMFORGE_SCENE=<path>"
        )
    if not ARM_URDF.exists():
        issues.append(
            f"Arm URDF not found: {ARM_URDF}\n"
            f"  → override with  export SIMFORGE_URDF_DIR=<dir>"
        )
    if not CUROBO_CFG.exists():
        issues.append(
            f"cuRobo config not found: {CUROBO_CFG}\n"
            f"  → override with  export SIMFORGE_CUROBO_CFG=<path>"
        )
    return issues


def assert_ready():
    """Raise RuntimeError listing any missing paths."""
    issues = check()
    if issues:
        raise RuntimeError(
            "SimForge path configuration incomplete:\n"
            + "\n".join(f"  • {i}" for i in issues)
        )


if __name__ == "__main__":
    issues = check()
    if issues:
        print("SimForge config — MISSING:")
        for i in issues:
            print(f"  • {i}")
    else:
        print("SimForge config — all paths OK:")
        print(f"  ISAACSIM_ROOT : {ISAACSIM_ROOT}")
        print(f"  SCENE_USD     : {SCENE_USD}")
        print(f"  ARM_URDF      : {ARM_URDF}")
        print(f"  CUROBO_URDF   : {CUROBO_URDF}")
        print(f"  CUROBO_CFG    : {CUROBO_CFG}")
