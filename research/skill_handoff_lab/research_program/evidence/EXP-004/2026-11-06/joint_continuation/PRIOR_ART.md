# Joint physical continuation: algorithm boundary

Primary papers and inherited pinned implementations read on2026-11-06:

- [STAP, ICRA2023](https://sites.google.com/stanford.edu/stap): product of predicted skill feasibility with learned dynamics, uncertainty and sampling-based plan optimization. Local pinned `external/STAP-0fe0dcbd/stap/agents/sac.py` computes reward + discount * target value, and offers mean/min/median actor-Q aggregation. Our experiment uses actual transitions and online policy iteration, without claiming to reproduce STAP's complete model-based search. Mean-Q is inherited mature machinery.
- [RecoveryChaining](https://arxiv.org/html/2410.13979): nominal controllers are extended actions available to a learned recovery policy, with learned handback. Our finite set retains fixed physically executable proposal Options and fixed release conditions. Repeated release/redecision is an interface foundation, not a novel handback algorithm.
- [Policy Decorator, ICLR2025](https://policydecorator.github.io/): base-policy refinement with bounded residual action and progressive exploration. Its inherited code `external/atomic_handoff_prior_20261105/policy_decorator.py` is retained. Our zero-initialized logit correction and controlled categorical exploration are standard base-policy refinement ingredients; they are not motor residuals or a formal reproduction of its continuous policy.
- [SPIBB, ICML2019](https://proceedings.mlr.press/v97/laroche19a.html): inspected pinned `external/rule_option_spibb_20261105/spibb.py`, count support, frozen baseline probability mass and exact policy evaluation. No count-based confidence set or SPIBB neural safety guarantee is present here.
- [ThriftyDAgger, CoRL2022](https://proceedings.mlr.press/v164/hoque22a.html): novelty/risk-based supervisor intervention addresses when to seek corrections. Learning switching alone is not a new contribution. We do not use human supervisory queries.
- Inherited Option-Critic implementation remains a reference for learned termination. This experiment fixes Option release, rather than training an Option-Critic termination gradient.

## Why the previous classifier was sufficient

With a fixed predecessor, at most one adopted deviation and a fixed continuation,
its terminal advantage is a conditional binary-return difference. A supervised
cost-sensitive scorer can therefore account for improved deployment. Previous
online and initializer outcomes being identical is consistent with that model.

Changing the real grasp changes the downstream state distribution. After a
completed Option, opening another actual decision makes its return depend on the
later policy. A fixed pair label collected with a permanently frozen suffix is
then not Q of the new deployment policy. This is a distinction in executable
policy classes, not a claim of a novel Bellman equation.

## Implemented comparison

Both arms inherit the same learned front logits, paired-data scorer, strong
planner, observations, proposal generators and completed-Option runtime. A
zero-output residual actor changes all three actual grasp probabilities and
rear rule/proposal probabilities. Fresh twin critics avoid transplanting the
old incompatible Q function. Standard finite discrete SAC, equal front/rear
sampling and success replay are shared. The rule branch executes one feedback
chunk; a proposal executes its inherited complete Option. Policies reopen at
actual Option release. One-shot warm-start and frozen rule remain separate
controls; they are not falsely described as the identical recurrent interface.

Native finite horizon uses gamma=1. Summing actual transition rewards and
physical durations preserves reward + (1-done) V(next); with gamma<1 each
within-Option reward and bootstrap would require its physical-step exponent.
Potential shaping telescopes to native terminal return plus an initial-state
constant. Remaining budget is observable. No additional native steps are used.

The minimal return-mix probe changes only predecessor targets: recent actual
suffix return contributes one half, standard TD the other half. Rear updates,
actor loss, legal actions, networks and per-episode update counts are identical.
This is an empirical multistep/Monte-Carlo mixture, not an originality claim.
Recent collected returns still come from behavior policies; recency does not
prove current-policy consistency. A positive result would require separation
from ordinary multistep learning, data support and controller improvements.

All learned comparisons collect their own episodes and use the same fixed
final checkpoint rule. All historical failed candidates and warm scores stay
unchanged. Independent method advantage requires actual new task outcomes,
not changed logits, fitting error or additional decisions.

## Evidence-driven second iteration: action identity

The first standard SAC's real replay contains four states where two legal
proposal labels produce the same executable Option, yet its Q gaps range from
0.018 to0.261. More generally selector1/2 each produces multiple different
physical corrections across visited states. This does not make their MDP
semantics invalid: the frozen proposal generators are deterministic functions
of state. It is a concrete representation/generalization burden, not proof
that it caused every observed failure.

The next controlled experiment gives **both** critics all three proposal
physical IDs and execution types. A flat125-input,72-hidden critic predicts
three values. A shared critic evaluates each selected executable identity from
139 inputs with64 hidden units. Twin counts are18582 and18050 respectively;
the baseline has slightly greater capacity. Actors, SAC/BC updates, rollout
policies, native budgets and base initialization are identical. The shared
form enforces equality for executable aliases; it must improve actual policy
learning to justify anything beyond that algebraic property.

[MP-DQN](https://arxiv.org/abs/1905.04388) and
[parameterized-action deep RL](https://arxiv.org/abs/1511.04143) establish
strong precedents for action-parameter-conditioned value networks. MP-DQN
studies unselected continuous-parameter dependencies; our finite proposal
identity experiment is not an MP-DQN reproduction and cannot claim that
standard parameterized Q is new. Current continuation, actual duration and
joint physical grasp execution remain the task-specific foundations.

Before either second-iteration arm started, a disclosed common-data amendment
added the96 new design episodes to both replays. First-cycle training had few
successful grasp2 examples despite actual design evidence. Logged executor
origins distinguish abstract rule choice from physical action0; compatible
joint transitions are reconstructed from those origins and actual rewards,
not relabeled from an incompatible old replay. Each arm reexecutes764008/grasp2
and764004/grasp1 and verifies all100 recorded physical steps before training.
Both passed. These four new episodes and the96 shared episodes are charged.
The second iteration cannot be compared with the first as an isolated critic
change: data and fresh train/dev distributions changed together. Its internal
flat-versus-shared comparison holds these factors fixed.

## Third iteration: full-policy return versus bootstrapped ranking

The action-shared critic's midpoint losses came from changed grasps, not
failure to enforce aliases. We therefore compare an ordinary same-data SAC
with a complete-joint-policy score-function update. Both keep the same actual
recurrent interface, start from the strong warm actor, and have32 own native
training episodes, the96 design traces and32 separately executed frozen-joint
reference traces. The latter are charged; no simulator snapshot is restored.

If a frozen reference return b(s0) is independent of the sampled actions,
E[(R-b(s0)) sum_k grad log pi_theta(a_k|o_k)] is the ordinary episodic policy
gradient. Here the sum includes the real grasp and all reopened successor
decisions; intra-Option continuation is frozen. This subtracts a root baseline
from the **actual current trajectory**. It does not assign an old fixed-suffix
intervention label to one action of a different current continuation. Native
gamma=1 and the full100-step task endpoint are unchanged.

The implementation differentiates the exact0.9 policy+0.1 uniform-legal
sampling mixture and uses a clipped whole-sequence likelihood ratio. Forty
updates are allowed on nonzero paired returns; zero differences perform no
optimizer step, avoiding Adam-momentum drift without evidence. The multiple
clipped epochs are a familiar local surrogate, not an unbiased-estimator or
monotonic-improvement theorem. SAC gets40 actor/critic iterations every episode;
actual optimizer costs differ and are disclosed. Both can access the same
physical data; SAC also consumes reference trajectories as legal off-policy
experience. Deployment is greedy, whereas training samples, so improved
training return alone does not establish deployment improvement.

Baseline subtraction, common-initial pairing, REINFORCE/PPO-style clipping
and nominal retention are established ideas. Any prospective contribution
must be in their demonstrably useful executable joint-skill combination;
no independent originality follows from this estimator being implemented.

Scope limitation: the joint policies choose the rule or two frozen, state-
dependent proposal generators. The executor retains its12 physical primitives,
but the learner does **not** freely choose every primitive in every state.
All paired algorithms share this restriction. Recorded visits omit some
physical proposals, so absence of a learned rescue cannot establish that the
full legal primitive library lacks a rescue. Grasp effects are real; neither
the small design union nor a proposal-selector failure is a global oracle.
