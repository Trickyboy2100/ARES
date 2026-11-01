# Minimal physical attribution experiment — preserved pre-run design

Status update: all four episodes were subsequently executed; see ../multiskill_slowdown/README.md for actual results. The pre-run reasoning below is preserved.

The saved evidence distinguishes two mechanisms. In740001/g035, measured attachment rotates10.41° by controlframe57, before the first nonzero tool/cube contact at58/substep2 (4.01N). For740000/g070 andg105, maximum TRANSFER attachment drift is only.295°/.215° despite33/54 control steps. The latter are not examples of the same large pre-contact grasp drift; long geometric route and budget remain the leading measured distinction.

Timing has not been dynamically certified. Pinned mplib defaults velocity and acceleration toones and the native solver scales both by.9. Existing manual TracePlanner then sends every second path sample at20Hz. This does not preserve the original TOPP acceleration envelope. Five saved reference cases have actual20Hz joint velocity below.667×URDF velocity limits, while command finite-difference acceleration reaches5.48–14.13rad/s². URDF carries no acceleration limit. These facts do not justify faster stride or a blanket dynamic SkillContract PASS.

Proposed four episodes only:

- seed740001, g035 andg070, each one stride2 unchanged control and one stride1 TRANSFER-only slowdown.
- Same native100, scene, first-grasp algorithm, candidate search and selected transfer endpoint. No larger/faster stride or dynamics-limit relaxation.
- Retain the exact raw planned TRANSFER joint sequence and verify same prefix/state and path SHA. Stride1 changes temporal sample selection only for TRANSFER. Other controller parameters/algorithms remain shared; downstream feedback plans may legitimately differ because the actual attachment/state changed. They must not be mislabeled as identical downstream trajectories.
- Fresh controls use the same expanded observer as slowdowns, permitting verification against historical action/state traces. The observer reads actual100Hz q/qvel/qacc with explicit PhysX articulation fetch; native BaseEnv otherwise fetches state only after each20Hz control step. No extra env.step or state setter.
- Measure pre-contact attachment drift at comparable path progress, first actual contact event, robot tracking,100Hz velocity/acceleration, eventual native outcome and censored suffix budget. Slow execution can use the whole100-step budget before task completion; that is retained, not hidden by extending time.

This is a mechanistic simulation intervention, not a certified hardware-safe retiming or an RL evaluation. A g035 improvement alone would not establish useful task-conditioned RL: the stronger g070/g105 grasps may dominate it. If slowdown reduces pre-contact drift but sacrifices the native budget, the result describes a compliance/timing tradeoff, not a complete-task method improvement. If strong grasps remain stable, do not generalize a deliberately unfavorable grasp's recovery to the whole parameter space.
