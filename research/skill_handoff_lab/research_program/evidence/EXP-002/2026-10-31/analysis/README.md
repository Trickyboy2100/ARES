# EXP-002 automated evidence audit

{
  "saved_complete_episodes": 426,
  "saved_physics_steps": 85200,
  "saved_prefix_steps": 36069,
  "saved_skill_decisions": 9232,
  "optimizer_updates": 4480,
  "finalized_run_optimizer_updates": 4480,
  "unfinalized_training_runs": [],
  "excludes_live_unfinished_episodes": true,
  "training_tail_episodes": 5
}

QA issues: 0; exact recorded prefix pairs 1358/1358.
Per-run scores retain snapshot and stop-rule identity. Prefix comparisons may cross source hashes, but require the same seed, bias, physics and stop rule; source changes are explicitly recorded. A matching initial state hash does not imply matching grasp/contact boundary.
Source RGB/video replay adds no physics. Physical readback verifies constructed component values, not system-identification estimates.
Training tails are counted in cost but excluded from policy terminal success denominators. Live unfinished runs are not included until saved.
Effect activation/constant-feature diagnostics use saved source snapshots and only 21-dimensional observations. Older 19-dimensional records are skipped.

## Per-run scores

- adaptive_smoke / fixed_minus: 1/1; UNKNOWN_STOP_RULE; source a9248aea6982; diagnostic.
- adaptive_smoke / fixed_plus: 0/1; UNKNOWN_STOP_RULE; source a9248aea6982; diagnostic.
- adaptive_smoke / geometry: 1/1; UNKNOWN_STOP_RULE; source a9248aea6982; diagnostic.
- adaptive_smoke / rls: 0/1; UNKNOWN_STOP_RULE; source a9248aea6982; diagnostic.
- adaptive_smoke / search: 1/1; UNKNOWN_STOP_RULE; source a9248aea6982; diagnostic.
- bias_minus12_v2 / fixed_minus: 0/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_minus12_v2 / fixed_plus: 1/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_minus12_v2 / geometry: 0/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_minus12_v2 / rls: 0/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_minus12_v2 / search: 0/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_plus12 / fixed_minus: 2/2; UNKNOWN_STOP_RULE; source a837fd85d7e5; diagnostic.
- bias_plus12 / fixed_plus: 0/2; UNKNOWN_STOP_RULE; source a837fd85d7e5; diagnostic.
- bias_plus12 / geometry: 1/2; UNKNOWN_STOP_RULE; source a837fd85d7e5; diagnostic.
- bias_plus12 / rls: 1/2; UNKNOWN_STOP_RULE; source a837fd85d7e5; diagnostic.
- bias_plus12 / search: 2/2; UNKNOWN_STOP_RULE; source a837fd85d7e5; diagnostic.
- bias_plus12_v2 / fixed_minus: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_plus12_v2 / fixed_plus: 0/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_plus12_v2 / geometry: 1/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_plus12_v2 / rls: 1/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- bias_plus12_v2 / search: 1/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- effect_ln_sac_alpha002 / effect_ln_sac: 17/44; OBSERVED_GOAL_STOP; source 2681604062fd; train.
- effect_sac_pilot_1024_retry / effect_sac: 15/43; OBSERVED_GOAL_STOP; source 2681604062fd; train.
- eval_controls_-1.2 / geometry: 0/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_-1.2 / rls: 2/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_-1.2 / search: 4/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_0 / geometry: 4/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_0 / rls: 4/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_0 / search: 4/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_1.2 / geometry: 0/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_1.2 / rls: 0/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_controls_1.2 / search: 2/4; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_effect_ln_prior_only / effect_ln_sac: 2/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_effect_ln_sac / effect_ln_sac: 2/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_effect_prior_only / effect_sac: 3/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_effect_sac / effect_sac: 3/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_ln_sac / ln_sac: 4/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_ln_sac_alpha002 / ln_sac: 10/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_sac / sac: 1/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- eval_sac_zero_z / sac: 7/12; OBSERVED_GOAL_STOP; source 2681604062fd; eval.
- heavy_highfriction / geometry: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- heavy_highfriction / rls: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- heavy_highfriction / search: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- light_lowfriction / geometry: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- light_lowfriction / rls: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- light_lowfriction / search: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source 18e3563eb693; diagnostic.
- ln_sac_alpha002 / ln_sac: 22/47; OBSERVED_GOAL_STOP; source 2681604062fd; train.
- ln_sac_pilot_1024 / ln_sac: 18/44; OBSERVED_GOAL_STOP; source 2681604062fd; train.
- native_727000_003 / servo_local_hold_stop: 3/4; NATIVE_TRUE_GOAL_STOP; source f951b172e6d0; diagnostic.
- repeat_controls_-1.2 / geometry: 0/1; OBSERVED_GOAL_STOP; source 2681604062fd; diagnostic.
- repeat_controls_-1.2 / search: 1/1; OBSERVED_GOAL_STOP; source 2681604062fd; diagnostic.
- repeat_controls_1.2 / geometry: 0/1; OBSERVED_GOAL_STOP; source 2681604062fd; diagnostic.
- repeat_controls_1.2 / search: 0/1; OBSERVED_GOAL_STOP; source 2681604062fd; diagnostic.
- sac_pilot_1024 / sac: 1/4; OBSERVED_GOAL_STOP; source 0dab2da19d2d; train.
- sac_pilot_1024_retry / sac: 16/44; OBSERVED_GOAL_STOP; source 2681604062fd; train.
- search_v3_-1 / rls: 1/2; PRIVILEGED_NATIVE_GOAL_STOP; source b7532a855643; diagnostic.
- search_v3_-1 / search: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source b7532a855643; diagnostic.
- search_v3_1 / rls: 1/2; PRIVILEGED_NATIVE_GOAL_STOP; source b7532a855643; diagnostic.
- search_v3_1 / search: 2/2; PRIVILEGED_NATIVE_GOAL_STOP; source b7532a855643; diagnostic.
