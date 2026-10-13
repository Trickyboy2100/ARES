# P2.1 science decision

## Decision

**Priority A — deterministic planning/full enumeration.** Highest priority.
With <=12 fixed candidates per scene, exhaustive candidate generation plus
cuRobo validation is the strongest current method. On every V3 scene where a
legal endpoint existed, the ARES-R local `_edge_cost` choice also minimized the
CPU full-chain proxy (observed regret 0/5 scenes). This is not proof of global
optimality, but it removes a present need for RL.

**Priority B — supervised candidate value/risk model.** Conditional second.
Use only after independent cuRobo/Isaac executions establish either (i) mixed
downstream outcomes among legal terminals or (ii) meaningful planning-time,
clearance, limit-margin, or execution-time regret that the heuristic misses.
Train/test splits must be by scene revision. A model should predict downstream
success/cost, not reproduce `_edge_cost` labels.

**Priority C — CASH contextual/SMDP RL.** Implemented as V0, but scientific
promotion is deferred. RL becomes justified only if choices have delayed,
history-dependent effects that exhaustive one-step ranking or a supervised
ranker cannot capture. Current data have no such demonstration.

## Current learnable space

- Candidate legality varies substantially with symmetry and center yaw:
  52/120 fixed slots were legal. This is useful for action masking, not an RL
  reward.
- In feasible Cabinet scenes, CPU proxy full-chain costs vary by up to roughly
  4x. This may support cost ranking after cuRobo confirms it.
- No within-scene legal-candidate Boolean sensitivity was observed.
- The heuristic has zero observed proxy regret in the five feasible scenes.

## Minimum next empirical experiment

After the driver is repaired and one persistent-cuRobo request passes:

1. Use the frozen V3 candidate list; do not generate new seeds.
2. Select one Groove and one Cabinet scene by diagnostic coverage, not outcome.
3. Validate all legal candidates in each selected scene, estimated 6--12 plans
   per scene, with attached collision, independent dense validation, and no
   interpolation/teleport fallback.
4. Record success, planner time, trajectory duration, minimum clearance,
   singularity/limit margin, and failure reason.
5. Expand to 48--96 plans only if the first bounded batch shows actual
   candidate-dependent outcomes or heuristic regret.

If all legal candidates remain uniformly feasible/infeasible and the heuristic
remains optimal, stop method escalation and fix/measure the industrial task
geometry rather than training CASH.

## Relation to prior work

- STAP jointly evaluates geometric dependencies across sequences using learned
  skill Q-values and planning; CASH must beat exhaustive planning in this small
  action set, not merely rediscover it. Paper/code:
  https://arxiv.org/abs/2210.12250 and https://github.com/agiachris/STAP
- ViSkill learns terminal-state suitability for all subsequent skills. CASH's
  successor-effect head is analogous in purpose, but our actions are explicit
  industrial terminal configurations and strict planner/execution evidence is
  required. Paper/code: https://arxiv.org/abs/2307.16503 and
  https://github.com/med-air/ViSkill
- Adversarial Skill Chaining regularizes terminal distributions so the next
  policy can start. It is relevant only if our terminal distribution—not a bad
  fixture—is the bottleneck: https://proceedings.mlr.press/v164/lee22a.html
- SkiMo predicts skill outcomes for skill-space planning. That is a later model-
  based route if multi-stage outcomes show history dependence:
  https://arxiv.org/abs/2207.07560
- Curriculum-guided skill learning (2025) learns transition policies from a
  demonstration curriculum. It addresses bridging continuous skills; P2.1 is
  currently a discrete terminal-selection problem and should not import that
  machinery before the basic sensitivity exists.

