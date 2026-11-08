# Continuation likelihood and policy-versioned Option evaluation

This cycle resumes the strong `action_frontier/sac_32202` actor, both critics,
optimizers and 2218 inherited transitions. It does not restart from warm. Actual
Peg100 grasp/Option executor and official success are unchanged. The warm
predecessor is shared to isolate rear learning; this is not newly learned joint
skill synergy. Every arm collects its own native episodes.

## Competing explanations

1. Historical successful outcomes belong to their executed suffix, not a changing
actor. One-step TD propagates sparse terminal evidence slowly; naive complete
returns misattribute changed suffixes.
2. This mismatch is already addressed by mature off-policy multi-step learning.
A special continuation mechanism might add nothing beyond standard Retrace.
3. Periodically fixing the evaluated policy can reduce moving-target bias, but
can instead produce stale values and hurt deployment. This is a falsifiable
candidate, not an established independent contribution.

## Exact update

For an actual chain of completed Options, use native-duration reward sums and
finite-horizon gamma=1. Let `p` be the evaluated policy, `mu` the measured sampling
probability, and target Q the inherited exponentially averaged critics:

```
V_p(s) = sum_a p(a|s) [Q_target(s,a) - .01 log p(a|s)]
c_j = .9 min(1, p(a_j|s_j)/mu_j)
G_j = r_j + (1-done_j) [V_p(s_next) + c_(j+1)(G_(j+1)-Q_target(s_next,a_(j+1)))]
```

The correction uses the NEXT executed Option. Truncate at eight Options,
terminal100, a missing actual successor link, or missing behavior probability.
Old replay does not have `mu`: keep it as one-step evidence rather than infer it
from current networks. Every fresh action stores its actual mixture probability
(.9 Actor + .1 uniform legal actions); deterministic common front has mu=1.
No physical-state reset, hypothetical branch reward or future observation enters
the Actor. Explicit next-observation equality checks protect episode links.

Three matched arms: SAC uses one-step targets; trace uses current-policy soft
Retrace; lagged uses the same trace with an Actor snapshot refreshed each eight
own episodes. All share the same Actor objective (soft-Q improvement plus .1
success BC), Q/Actor capacity, action set, old/own 50:50 replay mixture and40
updates per own episode. Lagged does not lock physical control, restrict actions,
add a head or impose a deployment gate. Only the evaluation-policy version
changes. Standard Retrace and periodic approximate policy iteration are mature
components, not claimed novel.

## Boundaries and interpretation

Action0 at physical level continues the CURRENT queue, never restores an old
path. Selector0 is the existing rule chunk. The preserved AtomicMacro releases
on its actual inherited event; this cycle does not alter termination. Historical
757300 already has a recorded100-step witness reproduction under this interface
in `continuation_option/WITNESS_REPRODUCTION.json`; do not count it again here.

83 features do not establish a fully Markov state: complete queue and hidden
PhysX state are not provided. No global convergence/safe-improvement guarantee
follows from a tabular Retrace theorem. Soft stochastic targets also differ from
argmax deployment. Report physical deployment outcomes, not value fit as gain.
Frozen-reference lag is measured in training episodes, NOT physical commitment
steps. Continuation seeds share pretrained weights; they are not independent
network initializations.
