# Execution representation: nearest methods and implementation comparison

Read2026-11-05. Original papers downloaded for local reading; no third-party source redistributed. Standard constructs are not claimed original.

- **rLfD**, Davchev et al., RA-L2022: https://arxiv.org/html/2008.07682v5 . SectionsIII-B/III-C compare DMP parameter, coupling and task-space corrections, including orientation. DMP100Hz, learned correction10Hz, controller500Hz; corrections interpolated. This directly precedes any claim that the *location* of adaptation matters. Our scope is measured-reference versus existing-plan endpoint semantics in an actual multi-skill handoff, not invention of task-space residuals. No verified public implementation was obtained from the paper links in this pass.
- **Residual RL for Robot Control**, Johannink et al.: https://arxiv.org/html/1812.03201 . Eq5 adds learned and hand-designed control. Standard additive residual is a baseline, not our contribution.
- **Fabrica**, https://fabrica.csail.mit.edu/ and https://github.com/yunshengtian/Fabrica . Code pinned215a30fe51e59299588a5b5e417a9cb934fb393e, `learning/isaacgymenvs/tasks/fabrica/fabrica_task_assemble.py`, `_preprocess_actions` and `_apply_actions_as_ctrl_targets`. Path-aligned residual is transformed/scaled, added to a normalized assembly direction, then applied as a displacement from current fingertip. This differs from a full MPLib terminal-target replan. Actual object attachment drift is retained in our task; do not import fixed-attachment assumptions or claim a complete Fabrica reproduction.
- **Policy Decorator**, https://arxiv.org/html/2412.13630 and https://github.com/tongzhoumu/policy_decorator . Code pinned92ba9ba442587ae5989c286355ace2200a0537fb, `online/pi_dec_bet_maniskill2.py`: base action plus scaled residual, progressive exploration, SAC updates. These protect a base policy through mature techniques. Our uniform15% exploration and successful BC are not a faithful full Policy Decorator reproduction.

## Prepared standard comparator

Two levels must not be conflated. A minimal same-planner comparator is a **nonaccumulating phase-target residual**: preserve the phase's original requested endpoint T_phase and request T_phase+R_goal δ on every local correction, with the same explicit refresh/continue actions and full planner. Compare against incremental T_active+R_goal δ; otherwise identical action IDs, bounds and observation. This tests standard anchored versus incremental target adaptation without rebuilding low-level control. It is a standard reference, not an invention.

A control-level rLfD/Fabrica-like reference would apply bounded Cartesian displacement relative to the current nominal waypoint through a validated IK/controller interface. That changes execution authority and requires separate calibration, joint/dynamic checks and matched baselines; it cannot be labeled already tested from the current endpoint study. Neither comparator is claimed trained in the present first factorial until raw results exist.

## Falsifiable decomposition

For measured-reference endpoint action, Δp = [p_refresh(x)−p_active] + R_goal δ. A nominal3mm parameter constrains only the second term. For relative-active-target action, requested Δp=R_goal δ, but IK branch selection, retiming and contact can still cause discontinuous executed paths. Therefore separating reference refresh removes one confound; it does **not** prove a globally Lipschitz physical action map or stable RL.

The current actor sees queue summaries, not the entire queue/endpoint. Partial observability and compounded incremental offsets remain competing explanations if representation clarity fails to improve learned behavior. Changes to actor observation or reset semantics must be separate experiments, not silently added to the factorial.

Implementation detail: Policy Decorator sets residual-enable probability to min(global_step/H,1), zeros disabled residuals, scales enabled residuals by alpha, and adds them to the base action. Its defaultH=2,000,000 and alpha=.2 are code defaults, not values adopted here. The code also uses a different reward/termination bootstrap convention; simply importing its loss would not be a controlled comparison.
