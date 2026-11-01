# Two-skill parameter RL: prior-art boundary and testable mechanism

2026-11-01. **Literature/source review only; no new physics or training in this document. Proposed mechanism remains UNVERIFIED.** Scope follows the new PAPER_CONTRACT and EXP-003/004; local servo improvements are shared infrastructure, not the paper contribution.

## Verified primary sources

| Work | Actual review / fixed source | Consequence for ARES |
|---|---|---|
| RAPS | [NeurIPS2021 paper record](https://proceedings.neurips.cc/paper_files/paper/2021/hash/b6846b0186a035fcc76b1b1d26fd42fa-Abstract.html); official code SHA `1c56b0fe367cc32d3ee548f407ac3a9af9e3ce2e`, `rlkit/rlkit/envs/primitives_wrappers.py`; top-level MIT read | Wrapper selects primitive by argmax of action prefix, splits bounded parameter arguments, executes the selected primitive. Learning skill choice/parameters and gaining from an action interface is prior work. |
| STAP | Reuse Oct31 verified SHA `0fe0dcbd5ede5ce4ee3d952642f70b5550a0813c`, MIT; [CEM source](https://github.com/agiachris/STAP/blob/0fe0dcbd5ede5ce4ee3d952642f70b5550a0813c/stap/planners/cem.py) | Learned skill dynamics/value and candidate sequence search already address downstream dependencies. Calling a second skill Q a contract is no distinction. |
| Foresight Residual RL | [Full primary manuscript](https://arxiv.org/html/2607.16506v1), algorithms1/2 reviewed Oct31; official project URL unavailable then, no verified code | Downstream rollout success labels, visual foresight prediction, backward residual-policy training and terminal reward modulation already exist. Do not present these as new. |
| Fabrica | [CoRL2025 primary paper](https://proceedings.mlr.press/v305/tian25a.html); official SHA `215a30fe51e59299588a5b5e417a9cb934fb393e`; `learning/isaacgymenvs/tasks/fabrica/fabrica_task_assemble.py` lines157,427–471 inspected; root MIT and learning BSD-3-Clause read | Code preprocesses path-frame actions and combines plan-derived correction with policy action before controller targets. Task-frame residuals, planning for downstream stability and assembly generalization are established competitors. |
| SCOUT | Reuse verified SHA `cdb079ccf9f40c5ed120261f665945aeff3a6473`; [official project](https://liy1shu.github.io/SCOUT/), inner-loop and trainer reviewed Oct31 | Shared belief adapted from action-outcome prediction loss already exists. No verified top-level code license at checked SHA; third-party license is not whole-project permission. |
| DAWN | Reuse verified SHA `71122b7fa89568bc2d49831fed8cb1b01e5a91e8`; [official publication](https://proceedings.mlr.press/v306/ma26s.html); critic/trainer reviewed Oct31 | Nominal-data anchoring and critic LayerNorm are baseline techniques. Checked LICENSE was0 bytes. ARES independent LN implementation is not DAWN reproduction. |

RAPS/Fabrica HEADs were resolved through GitHub API and files fetched by immutable SHA on Nov1. No external code was installed/executed/copied; no reproduced paper score is claimed. Earlier checks are documented in [Oct31 review](../../EXP-002/2026-10-31/PRIOR_ART_REVIEW.md). RAPS code: [fixed wrapper](https://github.com/mihdalal/raps/blob/1c56b0fe367cc32d3ee548f407ac3a9af9e3ce2e/rlkit/rlkit/envs/primitives_wrappers.py); Fabrica: [fixed task](https://github.com/yunshengtian/Fabrica/blob/215a30fe51e59299588a5b5e417a9cb934fb393e/learning/isaacgymenvs/tasks/fabrica/fabrica_task_assemble.py).

## Minimal scientifically correct SMDP

At skill boundary k, expose state `s_k=(current observable geometry/velocity, task, skill, remaining physical budget, successor contract)`. Select legal parameters `xi_k`. Execute the unchanged skill controller for actual duration `tau_k` and return **measured** SkillResult `r_k`, next observable state `s_(k+1)`, and consumed budget. The next parameters must be chosen after this physical result exists.

A two-boundary example is Pick/GraspTool→measured grasp-and-lift completion→Hook/Pull decision. Merely dividing a planned trajectory at two waypoints is not two-skill adaptation. Stack's Release/Verify boundary is valid only after actual opening, physical settling and state measurement. Failures remain full episodes, not silently reset away.

A standard baseline uses duration-correct target:

`y_k = sum_(j=0..tau_k-1) gamma^j reward_(t+j) + gamma^tau_k (1-done) V(s_(k+1))`.

This is ordinary Option/SMDP value learning, not a contribution. If an implementation discounts once per skill instead, report a deliberate different objective; do not claim equal physical-time credit. Use same entropy convention and action dimensionality in matched RL comparisons.

## One bounded candidate: contract-relabeled physical-result supervision

**Hypothesis**, not established novelty: with limited real successor rollouts, predicting a reusable distribution of physically measured skill results and evaluating it under different successor contracts supplies more useful cross-task supervision than a binary downstream-success head. It must help parameter decisions beyond same-input/same-data strong RL and supervised dynamics+search.

1. Predict from pre-skill observable state and candidate parameters a distribution over **measurable handoff quantities**: tool-to-gripper transform, slip/velocity, stable object pose and elapsed time. Do not feed the actual future SkillResult to a pre-skill actor/critic. At boundary2, the actual result is legitimate current observation.
2. The successor contract maps a result to a vector of geometric/kinematic/budget margins, e.g. hook reach/approach clearance, stable support region, or remaining feasible motion budget. Contracts are specified before evaluating the rollout. Their truth is not native task success.
3. One recorded physical handoff can be evaluated under alternative **declarative** successor requirements without new physics, giving margin labels. This relabeling must not invent counterfactual successor success/reward. Dynamic outcomes still require actual successor execution and remain separately supervised.
4. Train ordinary SMDP return together with a contract-margin distribution loss. The actor maximizes the same full-task critic as the baseline; a learned margin representation is a proposed data-sharing mechanism, not an extra hidden mask or reward advantage. Do not use an auxiliary margin to prohibit known physically valid actions unless every comparator receives the same projection.

Potential difference from a fixed next-skill success head is reuse of actual physical result labels across changing explicit successor requirements, including unseen requirements; difference from generic STAP dynamics is **not automatic**. If an equally sized full-result dynamics model plus contract evaluation performs the same, the specialized representation has no independent increment. If contract relabeling equals ordinary multi-goal auxiliary learning, credit that literature and do not claim first use. No originality claim is currently warranted.

## Required alternatives and falsifiers

| Comparator / ablation | What it rules out |
|---|---|
| Strong measured-state geometry/replanning and equal-budget parameter search | Gains from correcting stale geometry or spending more queries |
| Mature same-interface LN-SAC/PPO with identical task/contract/SkillResult inputs | More information, easier action authority, weak learning baseline |
| Same-data supervised parameter ranker | Pure supervised selection is sufficient |
| STAP-style result model + parameter search | Generic model-based planning explains the result |
| Foresight-style actual downstream return predictor | Ordinary successor-value shaping explains the result |
| Full-result auxiliary prediction without contract relabeling | Any representation learning explains gain |
| Same contract labels supplied directly to the baseline auxiliary head | Extra labels/data explain gain |
| Shuffled or irrelevant successor contracts, no relabeling, equal-capacity features | Claimed successor specificity is spurious |

A physical matrix must first show conditional action rankings: changing predecessor parameters changes actual handoff, and **different successor requirements change which handoff is useful**. Rescue/regression is evaluated on complete trajectories. A best post-hoc parameter union is only an opportunity upper bound.

Reject the candidate if strong geometry already captures all differences, if no two effective decisions exist, if one fixed parameter dominates, if the auxiliary loss only predicts reachability but not complete-task choices, or if gain disappears under equal labels/data. Do not force POMDP: measurable attachment/velocity belongs in the common observation. Only genuine observable aliasing with useful action-result probes justifies hidden-state belief and SCOUT/RLS/history comparisons.

## Practical decision for the current session

Collect the smallest informative real two-boundary paired matrix first, then choose one mechanism based on it. Log initial state hash, requested/applied parameters, actual SkillResult, both durations, declared successor contract, second decision, full native success, and search/learning costs. This permits supervised/standard RL/value baselines without re-labeling old physics as new experiments. Shared stop fixes and improved LN baselines stay separate from the method claim.
