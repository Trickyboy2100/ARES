# CASH method V0

CASH V0 is a constrained actor-critic for **C**ontext-aware **A**ction
**S**election at skill **H**andoffs. This is an implementation milestone, not a
positive method result.

## SMDP

One episode has three decision boundaries:

1. `GRASP_BRANCH`: select a physically defined grasp symmetry;
2. execute and re-observe, then `CENTER_TERMINAL`: select center yaw/IK branch;
3. station/scene revision and re-observation, then `PREPLACE_PLAN`: select a
   downstream approach plan and observe the executed downstream result.

Observation = robot state + remaining task goal + next-skill constraints +
revisioned history. Actions carry explicit legality masks. The backend is the
only authority for transition and reward. Strict mode accepts rewards only from
`CUROBO_ISAAC_EXECUTED` or `REAL_ROBOT_EXECUTED` transitions; CPU endpoint
proxies, hand scores, timeout, and `_edge_cost` are rejected as truth labels.

## Model

`cash/model.py` implements a candidate-set actor-critic: a state encoder and
action encoder produce masked policy logits; a value head estimates return; an
auxiliary successor-effect head predicts post-action effects. `cash/trainer.py`
contains one clipped actor-critic update and an effect-regression loss. The
auxiliary head is not itself the reward.

## Baselines

- ARES-R heuristic/Beam: same candidate/action mask, scored by `_edge_cost`.
- supervised Ranker: same features, trained only on executed outcome evidence;
  the implementation rejects proxy/hand-authored labels.
- ordinary MaskablePPO: specified through `sb3_contrib.MaskablePPO`; unavailable
  in the current environment because Gymnasium/SB3-contrib are not installed.
- CASH: identical environment and candidate budget, plus successor-effect head.

No new packages were installed. No Qwen/VLA/WebUI dependency was introduced.

## Current artifact

`results/cash_v0/CASH_V0_UNTRAINED.pt` is a deterministic architecture
initialization, SHA-256 recorded beside it. It is deliberately marked
`UNTRAINED_ARCHITECTURE_INITIALIZATION_ONLY`; it is not a learned policy.

## Promotion conditions

CASH must be compared on identical frozen scenes, candidate lists, oracle
budgets, and executed outcomes. It must show value beyond exhaustive search or
the supervised ranker. A learning curve cannot be produced honestly until the
GPU stack supplies repeatable executed transitions; a synthetic or CPU-proxy
curve would answer a different question.

