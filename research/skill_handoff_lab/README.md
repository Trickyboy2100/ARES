# ARES Skill-Handoff Lab — RTX 3080 Ubuntu sim-to-real starting point

**Purpose:** prepare a **research-only**, hardware-free experimental base using the real ARES dual-arm URDF and the ARES-R Skill / Motion / Scene contracts. Prove (or disprove) the *skill terminal-state / successor feasibility* hypothesis before training RL.

**Branch:** ARES \`research/skill-handoff-simbase-20261013\`. Parent ARES commit \`b978cbd669b5a3f6bc0bd19defcbe5256692f145\`.
**Reference:** ARES-R \`research/skill-handoff-smoke-20261012\` (includes code based on integration \`7120c3f3aad17330d0af6ada1e39c60bc58fe7fb\`). Newer uncommitted October site configs MUST be imported later and carefully diffed; do not assume this snapshot matches the live site.

## Why this repository arrangement

- **ARES = active simulation and kinematics research tree**, with existing \`isaac_sim/simforge\`, full JAKA MiniCobo/EG2-4C2 URDF+meshes, real IK/FK and Isaac physics demos.
- **ARES-R = read-only contract reference** in the local research sandbox: BODY axes, arm mounts, tool and collision revisions, site soft limits, SceneSnapshot, SkillRuntime, transfer-chain heuristic and frozen evidence. Do not modify \`feat/e2e-v0-integration-20260917\` from here.
- ARES-R's separate \`research/skill-handoff-smoke-20261012\` already includes an evidence audit and controlled-candidate headroom gate.
- The previous non-research \`~/Developer/PG-JY/jinyu_ros_pkg/nodes/simulation\` tree, installed MuJoCo and Isaac 4.5 Docker are **inspect-only**. Do not change or delete them.
- Preserve any locally installed cuRobo/Isaac/PyTorch. In particular, do not blindly follow old SimForge SETUP advice to install Python 3.13 or Omniverse Launcher.

Read **`SITE_20261012_REFERENCE.md`** before generating any synthetic industrial scene: October field logs describe a **cabinet object=4 / 70 mm preland target**, which is distinct from the earlier groove object=3 task. Neither September git state nor archived scene is a live controller snapshot.

## Step 1 — create an isolated research workspace

Run in an ordinary **Ubuntu terminal on the RTX 3080 host** (not on the .32 robot controller):

\`\`\`bash
export LAB="$HOME/robot-research/ARES-SkillLab"
mkdir -p "$LAB"/{repos,envs,data/site_exports,data/scenes,runs,logs,reports}
cd "$LAB/repos"

git clone https://github.com/Trickyboy2100/ARES.git ARES
git -C ARES fetch origin research/skill-handoff-simbase-20261013
git -C ARES switch --track origin/research/skill-handoff-simbase-20261013

git clone https://github.com/Trickyboy2100/ARES-R.git ARES-R
git -C ARES-R fetch origin research/skill-handoff-smoke-20261012
git -C ARES-R switch --track origin/research/skill-handoff-smoke-20261012
\`\`\`

If repositories are private, use an already configured Git credential method. Do not overwrite a pre-existing directory; use the existing local clone or choose a fresh workspace.

## Step 2 — perform an audit before installs/upgrades

\`\`\`bash
cd "$LAB/repos/ARES"
python3 research/skill_handoff_lab/tools/audit_host.py \\
  --output "$LAB/runs/host/host_audit.json"
\`\`\`

Review NVIDIA GPU VRAM, driver, Docker images, old 4.5 container, installed Python/NumPy/MuJoCo and the legacy simulation path.

**Stop before installing or upgrading** Isaac, CUDA, drivers, torch or cuRobo. The legacy 3080 system reportedly had Ubuntu 22.04, 10 GB VRAM, 32 GB RAM, a working Isaac Sim 4.5 Docker and MuJoCo; actual current state wins.

## Step 3 — clean L0 Python for CPU geometry contract

\`\`\`bash
cd "$LAB"
python3 -m venv envs/kinematics
source envs/kinematics/bin/activate
python -m pip install numpy scipy
cd "$LAB/repos/ARES"

PYTHONPATH=. python3 -m unittest discover \\
  -s research/skill_handoff_lab/tests -p 'test_*.py' -v

PYTHONPATH=. python3 research/skill_handoff_lab/tools/align_repos.py \\
  --ares "$LAB/repos/ARES" --ares-r "$LAB/repos/ARES-R" \\
  --output "$LAB/runs/alignment/static_alignment.json"

PYTHONPATH=.:$LAB/repos/ARES-R/src python3 \\
  research/skill_handoff_lab/tools/smoke_fk.py \\
  --ares "$LAB/repos/ARES" --ares-r "$LAB/repos/ARES-R" \\
  --samples 16 --output "$LAB/runs/alignment/fk_parity.json"
\`\`\`

Required first checkpoints: tests PASS, pinned mesh/URDF hashes match, site soft joint limits intersect robot URDF bounds, L/R BODY base signs correct, two independent FK implementations agree over randomized and selected recorded poses.

If asset parity fails, investigate **which file/version** differs. Do not edit or replace the real pinned model to force a pass. Existing \`jaka_minicobo_curobo.yml\` has an **old absolute asset path**; derive a local research-only config later instead of changing the original.

**Not checked yet:** USD world→BODY alignment, real live tool/TCP full SE(3), site camera position in USD, actual contact dynamics, current October controller soft limits and current robot worktree.

## Step 4 — P1/P2 and P2.1 status

The initial CPU smoke and the bounded P2.1 mechanism audit are implemented.
Read, in order:

- `reports/P1_P2_CPU_SMOKE.md`
- `reports/P2_1_ROOT_CAUSE.md`
- `reports/P2_1_SCIENCE_DECISION.md`
- `reports/CASH_METHOD_V0.md`
- `reports/TRAINING_ENV_REPORT.md`

P2.1 does not yet establish within-scene sensitivity among legal terminal
states; cuRobo/Isaac execution remains blocked by the documented NVIDIA driver
mismatch. The following list remains the experimental target rather than a
claim of completion.

Read \`WORK_ORDER_CODEX_3080.md\` in this folder. Codex should first deliver:

1. Existing MuJoCo/Isaac 4.5 install audit without modification;
2. a canonical BODY-frame scene contract for both arms, base, tray, gripper and goal regions;
3. a model-backed, reproducible *planning-only* scenario generator with paired terminal candidates;
4. a bound candidate → successor-feasibility scorer using genuine ARES-R cuRobo/soft limits and dense/level validation;
5. a smoke report comparing greedy, existing hand-designed transfer-chain ranking and an equal-budget oracle;
6. only then discuss whether a contextual ranker, bandit or RL can improve held-out cases.

### Do not claim

- A matching URDF proves the physical controller TCP is matched;
- simulated FixedJoint attachment equals verified physical grasp;
- Isaac's single-frame fallback joint jump is a valid trajectory;
- a planner \`SUCCESS\` is a physical skill success;
- replay tests prove skill RL is useful;
- random labels or fabricated results are a meaningful smoke;
- site pose \`center\` is one mandatory joint vector (it is a Cartesian task goal);
- an unchanged source snapshot remains valid after AMR movement.

**No robot motion, no remote hardware controls, no production branch updates.**
# 190 实际规划冲刺更新（2026-10-13）

最新实际结果见 `results/mentor_preview/latest/MENTOR_BRIEF.md`，原始证据见
`results/server190_20261013/`。相比上次提交470686b：190已完成真实CUDA、
cuRobo规划调用与一条41点诊断轨迹，独立有限范围验证通过；Cabinet两候选
共同模型碰撞失败，附着物和底盘代理问题已定位。正式Isaac5.1与真实学习
仍未完成。诊断成功不能作为handoff标签，旧负结果未覆盖。
