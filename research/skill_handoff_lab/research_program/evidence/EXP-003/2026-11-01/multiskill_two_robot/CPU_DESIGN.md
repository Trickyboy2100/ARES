# Native TwoRobotPickCube opportunity scout — CPU preparation only

Historical pre-run design, preserved after execution. The current twelve actual physical episodes and outcomes are in README.md / ANALYSIS.json. At design time no physical result existed; this text is not the current outcome summary.

Pinned ManiSkill `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3` source facts:

- `mani_skill/envs/tasks/tabletop/two_robot_pick_cube.py`: native horizon100, 40mm cube; initial cube y in[-.20,-.10]m, goal y[.10,.20]m and z[.02,.32]m; success is cube/goal Euclidean distance≤.025m AND right Panda static(.2). Airborne handoff is not required.
- `utils/scene_builder/table/scene_builder.py`, two `panda_wristcam` branch: actual bases y=±.75m, yaw=±π/2. These supersede the initial loader poses y=±1m.
- `agents/multi_agent.py`: physical actions are dictionary entries `panda_wristcam-0` and `panda_wristcam-1`.
- `envs/sapien_env.py::_load_agent` passes the same control-mode string to both agents; constructor mode is `pd_joint_pos`, not a list.

`tools/program_multiskill_two_robot.py` reuses native Panda planners through read-only single-agent views. Both arms still receive commands in one native environment. No reset, state setter, geometry change, success relaxation or horizon extension occurs between decisions. Initial default follows every native TOPP position (stride1). Optional subsampling is explicitly a controller configuration with measured tracking, not native retiming.

First decision: requested tabletop push endpoint y∈[0,.12]m. The left closed fingers approach the cube from behind and reobserve lateral drift across short push segments. Actual cube displacement, velocity, both-arm state and link/cube forces are recorded. Boundary requires cube y≥0 and remaining on table; it does not falsely declare zero velocity.

Second decision: after actual push/release, choose a right-grasp closing axis from current cube x/y using the same kinematic planner cost. The right arm closes, confirms grasp, lifts, transports using actual TCP-to-cube attachment, and corrects final observed displacement. Fixed axis interventions share all subsequent feedback. The other arm holds its last absolute target. Final native static/goal result is recorded at100, including failures and terminal hold.

Important limitations before physical testing:

1. It is not yet a strong baseline. Push contact geometry, path tracking, reset overhead and100-step budget require physical confirmation. If initial TOPP timing exceeds100, record that failure first; legitimate concurrent arm preparation or controller improvements must be shared across comparisons.
2. The planner uses robot self-collision but does not include the other robot or cube as world obstacles. Native contacts are fully measured. Any resulting controller collision defect is to be fixed before proposing learning around it.
3. A symmetric cube may make both grasp axes equivalent physically; mere IK failures or one dominant endpoint do not establish task-conditioned physical adaptation. Replanning may solve the entire panel.
4. Full object pose and velocity are observed. No latent-state/POMDP claim follows from using a previous SkillResult.
5. Whole-task native success remains the objective. A push reaching center alone is not task success.

Planned bounded first physics: seeds742000/742001 at endpoint.05 and full TOPP. Only after controller diagnosis, paired endpoints.02/.08 and fixed axis can test whether a meaningful second-action ranking changes. No RL training is justified by this CPU design alone.

CPU checks: compilation and pinned-PYTHONPATH `--help` passed. This was the pre-run status; subsequent execution is recorded separately with SHARED_GPU_CONTENTION.
