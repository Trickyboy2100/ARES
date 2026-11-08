# Version-bound continuation evidence: working hypothesis

The retained executor, 83 observations, three grasp choices, rear13 legal selectors and native100 horizon are unchanged. No new physical ability or safety guarantee is claimed. Initial study fixes the warm predecessor to isolate rear improvement; a separate joint SAC control retains predecessor learning. The frozen final checkpoint is used for the primary developmental comparison, with all intermediate scores retained.

A reference policy is frozen for physical branch collection. A reachable boundary and two different legal choices are selected by scene-index cyclic sampling before outcomes; this is systematic action intervention, not an uncertainty-based acquisition claim. Every branch starts from the original native reset. Equality is checked on the complete recorded prefix, actual observation, queue, cursor, previous action, active reference and budget. This is not a claim of serializing every hidden PhysX solver variable. The common suffix is the same frozen recurrent policy after inherited Option execution. No simulator state restoration occurs.

Each measured pair stores its policy version, real duration, complete trace paths, queue and prefix hashes, and signed terminal outcome difference. Ties stay in transition replay but supply no positive preference. Non-ties define a logistic ranking loss between the executed alternatives. Their actual transitions remain usable for ordinary SMDP learning even after preference labels expire.

Three same-interface arms receive the same initial physical data and equal own-policy and refresh budgets:

- Flat SMDP-SAC with the inherited success-BC term.
- Terminal-outcome supervised scoring plus physical non-tie ranking; no Q-derived Actor gradient.
- The same SAC plus physical preference improvement and an empirical continuation trust region.

The candidate computes KL from the frozen reference's soft distribution on successful **post-intervention** suffix states to the current policy. The intervention state is excluded, so retaining a suffix does not directly forbid the new first choice. The pair gradient is enabled while the minibatch mean suffix KL is below0.05; otherwise only the restoring KL gradient remains alongside SAC. Both ranking and successful-suffix retention are standard ingredients. This particular empirical composition is a falsifiable candidate, not a proven novel algorithm or monotone policy improvement theorem. Soft KL does not ensure identical argmax execution, nor protect unseen states.

At64 own episodes, each arm collects fresh original-reset pairs under its own checkpoint on the same new16-scene training distribution. It retires the old preference/anchor set, keeps compatible transitions, and continues own-policy training. Initial branches compare the warm reference choice; refresh compares the current frozen reference choice, not necessarily the fixed warm action. Thus the refresh estimator is a reference-policy local improvement contrast, rather than an exact implementation of fixed-baseline advantage throughout.

Gamma=1 respects the finite undiscounted native objective. Actual Option duration changes the remaining physical budget; decision entropy is not physically time-invariant. Recorded complete success, rescue and regression determine efficacy. Ranking fit, lower KL, altered logits, or more interventions are not positive method results.

## Pre-training candidate revision from actual SAC64 behavior

Both ordinary seeds fall from10/16 to8/16: one rescue and three regressions. In all four changed scenes the first changed selector is a warm nonzero Option replaced by rule0. Thus retention must preserve necessary nonzero behavior, not merely suppress extra corrections.

Before any retention-arm training, anchor construction was revised: retain **all rear decisions of successful reference trajectories**, plus strictly post-intervention states of successful alternative branches. The initial anchor set grows from260 to412 using exactly the same189 physical episodes and identical1366 transitions/125 pairs; no new labels are inferred at unexecuted states. This corrects coverage of earlier successful reference decisions. The prototype suffix-only scope remains reproducible, but was not physically trained and is not called a negative model. Actual baseline training uses the original source at `f3f57c8`; its SAC update ignores anchor annotations, so this candidate-only derived-supervision revision does not change baseline mathematics or physics. Current source and per-run hashes identify the revised candidate.
