# Baseline-relative Option adoption: prior-art boundary

Primary sources checked during this cycle:

- [SPIBB](https://arxiv.org/html/1712.06924), [official implementation](https://github.com/RomainLaroche/SPIBB), MIT, pinned `833f32b642759111907d2c4d086d1ae1ce9d84cb`. Inspected `spibb.py`: count-based support mask, preservation of baseline mass on bootstrapped actions, constrained redistribution on supported actions and model-based policy evaluation. This cycle has neither this finite-state confidence set nor its guarantee. A negative initial gate bias is not SPIBB.
- [RecoveryChaining](https://arxiv.org/html/2410.13979): learned recovery with nominal controllers as temporally extended actions, including learned handback. Current experiment instead freezes existing Options and their release rules and learns when to make at most one deviation. This is a restricted empirical comparison, not a claim to invent recovery/handback.
- [RAPS](https://mihdalal.github.io/raps/): parameterized action primitives are an established interface. Our grasp and closed-loop offset Options reuse this general foundation.
- [Policy Decorator](https://policydecorator.github.io/): base-policy protection and controlled exploration are mature ideas. The experiment uses the same bounded exploratory mixture in its supervised and policy-gradient arms; no protection theorem is inferred.
- Option-Critic source and Policy Decorator source pinned/reviewed in the preceding atomic_handoff cycle remain inherited references. The current Option termination is fixed, not learned by an Option-Critic termination gradient.

## What is actually trained

A shared 107-input MLP scores each frozen proposal from the 83-dimensional actual execution observation and the rule/proposal action identities. The finite policy chooses wait or one of two complete Options. It returns permanently to the frozen rule after that Option. No network predicts an extra future state or hidden physical variable.

The candidate applies measured terminal rule-relative advantage to the log probability of the complete waiting/adoption sequence. The supervision control classifies actually observed positive versus nonpositive Option effects. Both use common paired initialization, then their own complete native episodes. The likelihood-ratio estimator and clipped updates are standard components. An independent contribution would require reproducible benefit from continuation-bound credit assignment beyond these controls; it is not established merely by this implementation.

## Conditional identity, not a safety theorem

For a fixed initial distribution, unchanged predecessor, fixed Option executor and fixed rule continuation, let tau be the first selected deviation and z_tau the complete execution history. Then the expected full-task return difference equals the expectation of the corresponding single-deviation advantage at tau (zero if never adopted). Before tau the physical controller follows the rule. This identity does not identify unobserved Option values and does not make the 83-dimensional observation Markov. It also does not carry over unchanged to multiple deviations or changed Option termination.

For the sampled complete decision sequence, the score-function gradient includes all wait choices before adoption. Omitting those terms is a specific ablation, not an equivalent estimator. The experiment uses twofold regression cost: for binary returns this reweights baseline-success initial conditions; it is distinct from unweighted success maximization and is shared in comparator labels. Deployment argmax is also distinct from the stochastic training policy and must be assessed in real physics.
