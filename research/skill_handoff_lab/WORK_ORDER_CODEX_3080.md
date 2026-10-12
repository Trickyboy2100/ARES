# Codex Work Order — Bootstrap aligned ARES / ARES-R SkillHandoffBench on Ubuntu RTX 3080

**Date:** 2026-10-13
**Repo:** ARES research branch \`research/skill-handoff-simbase-20261013\`.
**Secondary reference:** ARES-R \`research/skill-handoff-smoke-20261012\`; production integration base \`7120c3f\`.
**Execution:** local Ubuntu workstation only. No live JAKA / AMR / gripper / Epic commands, and no workstation OS upgrades as a side effect.

## Goal and non-goal

Create a real model-aligned simulation/planning substrate on the existing Ubuntu RTX 3080 (reported Ubuntu 22.04, i5-13400F, RTX 3080 10 GB, RAM ~32 GB; must audit current state). **Do not train RL until a physics/geometric headroom smoke has shown that choosing the preceding skill's terminal state actually changes the succeeding skill's success, and that the existing heuristic has exploitable regret.**

Qwen, VLA, KG, WebUI/Task Studio and real robot execution are explicitly out of scope.

Read `SITE_20261012_REFERENCE.md`: October on-site HEAD was `a9ec30ee...`, with an Epic cabinet **object=4** and 70 mm preland goal; older September `object=3` is the groove task. Include these as distinct task families. Never confuse the two or use archived snapshots as current scenes.

## 0. Repositories and preservation

Read:
- \`research/skill_handoff_lab/README.md\` and scripts in this branch;
- ARES \`isaac_sim/simforge/SETUP.md\`, \`config.py\`, \`core/kinematics.py\`, \`demos/tray_grasp_cycle/demo.py\`, \`demos/dual_arm_planning_api.py\`;
- ARES-R \`config/robot_world.json\`, \`config/robot_collision_model.json\`, \`config/jaka_mini2_motion.site.json\`, \`config/named_poses.json\`, \`src/ares_r/manipulation/transfer_chain.py\`, \`src/ares_r/motion/runtime_goal_ik.py\`, \`src/ares_r/motion/tcp_orientation.py\`, \`research/skill_handoff_smoke/README.md\`.

Use \`~/robot-research/ARES-SkillLab/repos/ARES\` as the **only writable research code checkout**. Keep \`ARES-R\` read-only. Do not touch previously used \`~/Developer/PG-JY/jinyu_ros_pkg/nodes/simulation\`, the existing Docker image or the live .32 robot repo. Inspect \`git status\` before commits; avoid \`git reset --hard\`, bulk wiping, force-push and broad \`pkill\` commands.

## P0 — host audit / static model parity / FK parity

Run the included scripts; save output outside both repos in \`$LAB/runs/{host,alignment}\`.

\`\`\`bash
python3 research/skill_handoff_lab/tools/audit_host.py --output "$LAB/runs/host/host_audit.json"
# in a clean lightweight NumPy env
PYTHONPATH=. python3 -m unittest discover -s research/skill_handoff_lab/tests -v
PYTHONPATH=. python3 research/skill_handoff_lab/tools/align_repos.py \
  --ares "$LAB/repos/ARES" --ares-r "$LAB/repos/ARES-R" \
  --output "$LAB/runs/alignment/static_alignment.json"
PYTHONPATH=.:$LAB/repos/ARES-R/src python3 research/skill_handoff_lab/tools/smoke_fk.py \
  --ares "$LAB/repos/ARES" --ares-r "$LAB/repos/ARES-R" \
  --output "$LAB/runs/alignment/fk_parity.json"
\`\`\`

Important:
- ARES simforge's JAKA URDF and all mesh hashes must be compared against the **pinned ARES-R collision asset manifest**. Compare full lengths and geometries, not only robot model names.
- Check BODY axes \`+X forward +Y left +Z up\`; right arm at BODY Y=-0.20 m, left arm Y=+0.20 m; original ARES USD uses its own world convention.
- Use ARES-R's site soft joint limits as the target experimental limits; the real controller's latest limits may have changed, so import newer read-only 2026-10 material before claiming live alignment.
- Check all **six** joint axes and ordering, FK at q0 and randomized q and saved named poses.
- Do NOT infer physical \`T_link6_tcp\` from the ARES gripper's visual center, or from a tool ID alone. Produce \`UNVERIFIED_LIVE_TCP\` until a read-only current site tool transform/revision exists.
- The historical \`center\` is a **BODY Cartesian target**, not a fixed six-joint pose; allow multiple IK solutions. \`ready\` does not substitute it.
- ARES \`jaka_minicobo_curobo.yml\` contains an old machine-specific absolute path; if needed, generate a derived config in \`$LAB/runs/generated_configs\`, never silently change the canonical pinned YAML.
- Record every unresolved mismatch in \`alignment_delta.md\`; never modify model constants until provenance explains the difference.

Expected: \`STATIC_ASSETS_PASS\`, \`FK_IMPLEMENTATION_PARITY_PASS\`, explicit \`LIVE_TCP_NOT_TESTED\` and \`USD_MOUNT_NOT_TESTED\`. A FAIL must be triaged, not overridden.

## P1 — choose existing simulator, not a fresh install

1. Examine the prior \`mujoco_dual_arm.py\` and \`isaac_dual_arm_scene.py\` **read-only** (if still present) and existing Docker image \`nvcr.io/nvidia/isaac-sim:4.5.0\`.
2. Prioritize low-memory no-GUI FK/IK/collision model for the RL oracle. A small MuJoCo scene is the recommended interactive physics fixture **if the prior local model is working**.
3. Run a narrowly scoped *headless inspect-only* Isaac smoke only if the existing 4.5 container passes basic GPU/compatibility and asset/version tests. ARES SimForge docs target 5.1.0, so loading the current USD unmodified under 4.5 is **not guaranteed**. Do not install 5.1 just for a smoke.
4. In headless scene inspection, report: real BODY→sim WORLD transform, left/right base positions + yaw, both JAKA arm geometries, EG2-4C2 frame, tray dimensions, dock and cabinet geometry, collision/visual meshes, gripper opening convention, camera mounting. Save a diagram and \`sim_scene_alignment.json\`.
5. Do **not** run \`tray_grasp_cycle/launch.sh\` as an audit: it drives the simulation cyclically and may kill existing Kit processes. Its fallback on cuRobo failure may jump one frame to the goal. This fallback is unsuitable for scientific collision/reachability labels.
6. If Isaac 4.5 headless is infeasible on 10 GB VRAM, keep it as visualization only and continue P2 with an aligned MuJoCo/URDF static scene. Lack of Isaac GUI is NOT a blocker for this research.

Deliver \`sim_backend_decision.md\` with evidence-based choice and low-memory configuration.

## P2 — implement shared SkillHandoffBench interfaces

Use Python package \`research/skill_handoff_lab/bench/\` (new files); keep production ARES and ARES-R APIs untouched.

Typed scenario:
- \`scenario_id\`, \`task_family\`, \`seed\`, \`source_object\`, \`target_region\`, \`T_body_base\`, \`robot_model_revision\`, \`tool_tcp_revision\`, \`site_soft_limits\`, \`scene_snapshot_revision\`, \`static_and_unknown_obstacles\`, \`attachment_geometry\`, \`task_remaining\`;
- \`PICK_SCENE\` and \`PLACE_SCENE\` are distinct scene contexts with a base transform; place-side may NEVER inherit a stale pick snapshot.

Candidate:
- \`grasp_symmetry = NORMAL | FLIPPED_180\`;
- valid pregrasp/grasp/lift FK/IK states; variable center Cartesian goal (not fixed joints), center IK/yaw; destination \`rear-preplace = preplace + BODY[-.15,0,0]\`;
- typed \`SkillHandoff\` with required predecessor effects, successor preconditions, chosen terminal state, admissible preparation actions and resource revisions;
- hard collision/soft limit/horizontal-orientation filters separate from learned costs.

Trajectory oracle:
- reuse existing ARES-R cuRobo planner/validator via an offline adapter (external process if environments differ), never invent an always-successful fake oracle;
- cache by scenario/robot/tool/scene/candidate/planner profile/source revisions;
- planner \`FAILURE\`, \`TIMEOUT\`, \`OOM\`, \`STALE_SCENE\`, \`INVALID_ASSET\` are distinct and cannot be collapsed to physical infeasibility;
- avoid long one-shot workers; at most 1 GPU planner owner, bounded FAST→FALLBACK, and preserve the same collision policy across baselines.

## P3 — actual scientific Smoke-1 (planning-only; no RL)

Use two industrial task families:
1. tray→groove with level carrying, Cartesian center and rear-preplace;
2. tray→cabinet/preland with entrance-direction constraints.

Create **6** initially valid scenario contexts (minimum three per task family), and ≥4 terminal candidates per context. Randomize target-relative pose, obstacles and permitted center yaw/IK branch while preserving basic reachability. Do not reuse old site scenes as if current. All candidates within one scenario share the same obstacle sets and task.

Check:
- predecessor terminal \`feasible = True\` (real IK and currently valid geometry);
- successor launch/complete planning \`feasible = True/False\` via cuRobo + native site soft limits + horizontal path validator + attached geometry;
- compare greedy local cost, existing ARES-R \`_edge_cost\` / \`TransferChainPlanner\`, equal-budget beam/DP and an **in-sample oracle diagnostic**;
- report whether any task context has different successor feasibility depending on terminal candidate, and whether the heuristic misses the better one;
- use fixed datasets and seeds, independent hold-out task/scene groups, and report time/solver calls not just success.

Positive smoke gate: **verified candidate-sensitive successor feasibility exists** AND **headroom relative to a relevant baseline exists**, with actual request/result artifacts. There is no requirement to force a positive outcome. Negative gate: if all candidates equal, or existing DP matches the oracle at acceptable cost, **do not begin RL training**.

Deliver results under \`$LAB/runs/smoke1\`:
- \`scenarios.jsonl\`
- \`candidates.jsonl\`
- \`planner_outcomes.jsonl\`
- \`baseline_comparison.csv\`
- \`smoke1_summary.json\`
- \`smoke1_report.md\` with at least two diagnostic figures
- \`model_manifest.json\` and \`software_versions.json\`

## P4 — RL only after headroom

Three method candidates to reserve:
A) downstream-feasibility conditioned terminal selector, supervised ranker/contextual bandit then masked Q;
B) goal-conditioned Bridge Option when terminal redistribution truly requires a learned motion policy;
C) learned skill-effect model with rolling MPC when observation/preparation timing matters.

First question: can a supervised candidate ranker beat greedy under a fixed planning budget on held-out contexts? If yes, add 2–3 high-level steps and compare bandit vs RL on delayed effects. Do not claim RL improves anything based solely on synthetic label construction.

Strong references: STAP, Adversarial Skill Chaining, SkiMo, RAPS, skill chaining transition policies, cuRobo, and SB3-Contrib MaskablePPO.

## Reporting and commits

Create \`research/skill_handoff_lab/reports/BOOTSTRAP_3080.md\` and \`SMOKE_1.md\`. Record genuine executed commands, SHA, durations, CPU/GPU use, matching asset hashes, fail reasons and uncertain results.

Small isolated commits to the research branch only; never push commits to ARES main or ARES-R integration. Do not reconfigure the old local simulation environment or .32. At the end provide:
\`\`\`text
HOST_ENV_AUDIT
ASSET_PARITY
FK_IMPLEMENTATION_PARITY
SIM_USD_BODY_ALIGNMENT
CONTROLLER_TCP_CONTRACT
SCENARIO_GENERATION
CUROBO_ORACLE
SMOKE1_CANDIDATE_SENSITIVITY
SMOKE1_HEURISTIC_HEADROOM
READY_TO_TRY_RL
\`\`\`
Each item must be PASS / FAIL / NOT_TESTED with evidence; no optimistic flags.
