# Nearest methods inspected this cycle

Primary papers and official code were read; external implementations are read-only,
not installed or copied into the learner. Exact downloaded-source SHA and pinned
ACPPO/RecoveryChaining commits are in the run's prior-art metadata.

| Work | Mechanism actually checked | Consequence for this experiment |
|---|---|---|
| [Option-Critic](https://arxiv.org/abs/1609.05140), [author code](https://github.com/jeanharb/option_critic) | `neural_net.py`: beta-weighted continuation/termination bootstrap and termination advantage gradient | Our event-defined fixed internal Options are NOT learned Option-Critic termination or a new temporal abstraction. |
| [RecoveryChaining](https://arxiv.org/abs/2410.13979), [code](https://github.com/shivamvats/recovery-chaining) | `recovery_skill.py`, `on_policy_algorithm.py`, commit4894acc: recovery training around nominal policies; nominal switch actions/preconditions | Handing control back to a nominal suffix is prior art. No claim of reproducing its original task scores. |
| [BCP](https://arxiv.org/html/2608.03483v1) | Eq4 ordinal continuation probabilities; Eq5–8 group-relative trajectory update with fixed-horizon reference and replanning-efficiency reward | A fixed reference trajectory or continuation head is not novel. Our study retains event Options and native reward; it does not reproduce a VLA benchmark. Public paper read; no official BCP training implementation pinned. |
| [ACPPO-Corr](https://arxiv.org/html/2609.36250v1), [code](https://github.com/hshhahn/ACPPO) | `acppo_res.py` at1007a44: chunk-start policy, current-step correction, valid-prefix masks and chunked likelihoods; paper Algorithm1 separates planner/corrector updates | Correct probabilities must describe the executed closed-loop policy. Our discrete actual Option likelihoods are not its Gaussian motor-action residual. |
| [CA3C](https://arxiv.org/html/2610.07949v1) | Consequence agreement from imagined candidate futures chooses execution commitment without retraining base policy | This experiment neither uses a world model nor claims imagined agreement. We measure executed trajectories and train value/Actor updates. |
| [Retrace](https://arxiv.org/abs/1606.02647), [RLax implementation](https://github.com/google-deepmind/rlax/blob/main/rlax/_src/value_learning.py) | `retrace` clipped pi/mu coefficient and `general_off_policy_returns_from_q_and_v` recursion | Mature comparator. Its tabular convergence is not a theorem for our partial-observation neural soft-Q experiment. |

Prior AWAC, Q-filter, SPIBB and Policy Decorator comparisons are retained in
`../continuation_retention/PRIOR_ART.md` and prior cycles. No repeated training of
all historical algorithms is needed to identify this one update difference.

Candidate difference is narrowly scoped: freeze the evaluated successor policy
for eight collection episodes while continuing actual Actor improvement, with
measured Option likelihood correction and inherited strong replay. This is a
periodic approximate-policy-iteration hypothesis; originality is UNVERIFIED.
If standard trace matches it, retain the mature baseline and reject a special
continuation-version contribution. Mean Q aggregation, success BC and finite
reference snapshots are all shared or standard components.
