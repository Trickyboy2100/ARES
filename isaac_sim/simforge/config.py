"""simforge/config.py — centralised path configuration.

Every external dependency (Isaac Sim install, URDF files, scene USD, cuRobo
config) is resolved here, with an environment-variable override for each one.
Colleagues only need to set the relevant env vars for their machine layout.

Quick setup (add to ~/.bashrc or a project .env):

    export ISAACSIM_ROOT=~/isaacsim          # Isaac Sim installation root
    export SIMFORGE_SCENE=~/scenes/main.usd  # main USD scene
    export SIMFORGE_URDF_DIR=~/robot_urdf/   # dir containing jaka_minicobo.urdf
    export SIMFORGE_CUROBO_CFG=~/robot/jaka_minicobo_curobo.yml

All paths fall back to the original developer's layout when the env vars are
absent, so existing scripts keep working without any changes.
"""

import os
from pathlib import Path

# ── Repo root (two levels up from this file: simforge/config.py) ──────────────
_REPO_ROOT   = Path(__file__).resolve().parents[2]   # PG-JY/
_SIMFORGE    = Path(__file__).resolve().parent        # PG-JY/isaac_sim/simforge/

# ── Isaac Sim installation ────────────────────────────────────────────────────
ISAACSIM_ROOT = Path(os.environ.get(
    "ISAACSIM_ROOT",
    str(Path.home() / "isaacsim"),
))
ISAACSIM_SH = ISAACSIM_ROOT / "isaac-sim.sh"

# ── Main scene USD ────────────────────────────────────────────────────────────
# Use the scene that ships with this repo (simforge/scenes/main.usd) when
# SIMFORGE_SCENE is set.  Otherwise fall back to the original playground scene.
_DEFAULT_SCENE_CANDIDATES = [
    os.environ.get("SIMFORGE_SCENE", ""),
    str(_SIMFORGE / "scenes" / "main.usd"),
    str(ISAACSIM_ROOT / "playground" / "2026061100_main.usd"),
]
SCENE_USD = next(
    (p for p in _DEFAULT_SCENE_CANDIDATES if p and Path(p).exists()),
    str(ISAACSIM_ROOT / "playground" / "2026061100_main.usd"),
)

# ── Robot URDF files ──────────────────────────────────────────────────────────
# Expected layout:  <SIMFORGE_URDF_DIR>/jaka_minicobo.urdf
#                                      /jaka_minicobo_gripper.urdf
_URDF_DIR_DEFAULT = str(
    _REPO_ROOT / "jaka_ros2" / "src" / "jaka_description" / "urdf"
)
URDF_DIR = Path(os.environ.get("SIMFORGE_URDF_DIR", _URDF_DIR_DEFAULT))

ARM_URDF     = URDF_DIR / "jaka_minicobo.urdf"
_CUROBO_URDF_CANDIDATE = URDF_DIR / "jaka_minicobo_gripper.urdf"
# Fall back to the sim URDF in jinyu_ros_pkg if jaka_ros2 tree is absent
_CUROBO_URDF_SIM = _REPO_ROOT / "jinyu_ros_pkg" / "nodes" / "simulation" / "jaka_minicobo_gripper.urdf"
CUROBO_URDF  = _CUROBO_URDF_CANDIDATE if _CUROBO_URDF_CANDIDATE.exists() else _CUROBO_URDF_SIM

# ── cuRobo config ─────────────────────────────────────────────────────────────
_CUROBO_CFG_DEFAULT = str(
    _REPO_ROOT / "jinyu_ros_pkg" / "nodes" / "simulation"
    / "jaka_minicobo_curobo.yml"
)
CUROBO_CFG = Path(os.environ.get("SIMFORGE_CUROBO_CFG", _CUROBO_CFG_DEFAULT))

# ── cuRobo simulation URDF (contains gripper) ─────────────────────────────────
_SIM_URDF_DEFAULT = str(
    _REPO_ROOT / "jinyu_ros_pkg" / "nodes" / "simulation"
    / "jaka_minicobo_gripper.urdf"
)
SIM_URDF = Path(os.environ.get("SIMFORGE_SIM_URDF", _SIM_URDF_DEFAULT))


def check() -> list[str]:
    """Return a list of missing/broken path descriptions (empty = all OK)."""
    issues = []
    if not ISAACSIM_SH.exists():
        issues.append(f"Isaac Sim launcher not found: {ISAACSIM_SH}\n"
                      f"  → set ISAACSIM_ROOT to your Isaac Sim install dir")
    if not Path(SCENE_USD).exists():
        issues.append(f"Scene USD not found: {SCENE_USD}\n"
                      f"  → set SIMFORGE_SCENE to the scene .usd path")
    if not ARM_URDF.exists():
        issues.append(f"URDF not found: {ARM_URDF}\n"
                      f"  → set SIMFORGE_URDF_DIR to the directory with jaka_minicobo.urdf")
    if not CUROBO_CFG.exists():
        issues.append(f"cuRobo config not found: {CUROBO_CFG}\n"
                      f"  → set SIMFORGE_CUROBO_CFG to the .yml path")
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
        print("Missing dependencies:")
        for i in issues:
            print(f"  • {i}")
    else:
        print("All paths OK.")
        print(f"  ISAACSIM_ROOT : {ISAACSIM_ROOT}")
        print(f"  SCENE_USD     : {SCENE_USD}")
        print(f"  ARM_URDF      : {ARM_URDF}")
        print(f"  CUROBO_URDF   : {CUROBO_URDF}")
        print(f"  CUROBO_CFG    : {CUROBO_CFG}")
