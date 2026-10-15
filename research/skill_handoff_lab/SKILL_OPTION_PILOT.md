# Qualified skill-Option pilot

Research-only extension of the existing CASH/Gym interface; no hardware API.
Historical experiments and the production ARES-R repository are not modified.

## Evidence boundaries

- `qualified_planning.py`: C2 time parameterization with analytic polynomial
  derivative bounds, followed by independent proxy-world, self, attachment,
  reference soft-limit, orientation and scene-binding checks. This is not CAD
  collision certification. Spline paths are revalidated, not presumed safe.
- `prepare_skill_rl_pilot.py` and `build_skill_planning_graph.py`: bounded IK
  exploration and declared station-derived tasks. Derived scenes are not new
  observations. Keep original field snapshots and all rejected attempts.
- `assemble_skill_dataset.py`: binds each suffix to the exact preceding
  planner terminal. A nominal target is never substituted for an observed
  terminal. Cache lookup refuses different starts, scenes or source hashes.
- `option_backend.py`: planning-only two-stage state transitions, not physical
  execution. An unsuccessful attempt consumes one logical request budget unit
  but leaves q unchanged. Budget failure does not establish geometric
  impossibility. Actor features/masks never contain cached success labels.
- `train_skill_options.py`: standard MaskablePPO, candidate-order permutation,
  three seeds by default, four non-RL rules and frozen test logs. `gamma=1` and
  `lambda=1` define a finite-budget undiscounted SMDP objective with explicit
  solve/time/call costs. Exhausted task budget is terminal, not a time-limit
  bootstrap. FAST/FALLBACK internal attempt caps are reported separately from
  logical requests; do not call these observed internal attempt counts.
- `replay_controlled_site.py`: real Articulation drive/PhysX steps; only one
  reset write. Optional live Oracle continuation uses actual terminal q and a
  C2 drive-command bridge. It is a scripted integration test, NOT online RL.
- `qualify_physx_options.py`: independent checks on actual measured q samples,
  including command and measured acceleration across Option boundaries.

## Reproducibility and confidentiality

Run all scene generation, planning, datasets, checkpoints, plots and mentor
materials in an approved **non-Git controlled directory**. Requests include
sensitive geometry and robot state. Never commit generated artifacts, derived
field figures, USDs or cloud data. Source artifact hashes, scene groups, seeds,
software revision and full attempts must accompany private results.

The first pilot is deliberately small: a successful PPO run does not establish
new-workstation generalization, superiority to Direct, or a novel method.
Production ARES-R Beam remains distinct from the simple local greedy and
equal-budget retry baselines; do not relabel those as production Beam.

An inherited training USD can encode historical soft limits as physical joint
stops. `build_controlled_site.py` records old and corrected stops based on the
frozen reference configuration. These simulated corrections neither change
production limits nor grant permission for robot execution. Payload inertia,
tray CAD, inactive-arm geometry and contact instrumentation remain scoped
research approximations.
