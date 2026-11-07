# What the candidate does — and does not change

Let the shared warm logits be b(s). The flat Actor uses b(s)+z(s); the candidate uses b(s)+M z(s), with rank(M)=13 and unit row norms. Both can represent any13-way successor logit vector. The extra null coordinate in the candidate matches the otherwise unused output of the flat network; total stored parameter counts are equal. The three predecessor logits are separate outputs of the same hidden representation.

Under a local SGD approximation with fixed features, a logit-space gradient is propagated through M M^T. Thus shared coordinates change optimization coupling, not action authority or representable policies. The actual optimizer is Adam, so this expression is an explanatory approximation rather than an exact update identity. Same-signed offsets across active/refreshed references have positive coupling1/sqrt(6); distinct active-reference offsets share their reference coordinate (coupling1/2). Reference-specific interactions permit cancellation; they do not guarantee learning it from sparse physical data.

The physical hypothesis therefore requires useful shared gradient structure, not merely similar action names. The measured reference-dependent reversals are a concrete competing explanation. In particular,10/16 Y-plus pairs and12/16 Z-minus pairs have opposite native outcomes in the finite design set. Those outcomes include the common physical suffix and cannot be read as a local force derivative or as current-policy Q labels.

A true method benefit would require improved complete-task behavior at equal physical/data budget, beyond flat SAC, supervised outcome selection and frozen feedback. A different neural output, lower Q loss, successful catalogue union, or success on a hand-picked intervention is insufficient. If the two coordinates tie, the full executable interface remains useful infrastructure but the sharing hypothesis has no demonstrated independent benefit.

## SMDP target and temporal scope

Each sampled transition is an actual completed Option with duration tau, measured shaped reward and observed successor. Gamma=1 preserves the native finite-horizon undiscounted objective; no duration discount is claimed. The entropy term is applied per policy decision. Holding the shared executor fixed makes the current comparison matched, but changing Option duration would change this regularizer; temporal subdivision invariance is outside this candidate claim. Native final success remains the external evaluation metric.

## Causal scope of design pairs

Pairs use original-reset full physical reexecution, identical recorded pre-intervention physics/actions and a common recurrent suffix. They do not use approximate restored PhysX states. The finite union reports measured existence of a successful executable branch, not an available test-time oracle. A different grasp that independently succeeds is not evidence that coordinated changes at both boundaries are necessary.


## Revision after matched128-episode results

Flat SAC reaches21/24 and fixed reference/offset coordinates20/24. The candidate is therefore revised, not promoted: derive each replanning action's requested target displacement from the existing observed refreshed-minus-active translation/rotation and legal3mm offset. The derivation was checked against413 saved actual replan targets (maximum coordinate error4.36e-8m); one attempted replan without a returned target is counted separately. This is a geometry/interface check, not a claim that the ensuing joint paths are close.

For nine replanning selectors, define features with3mm translation units,5degree rotation units, and a unit categorical indicator for atomic refreshed continuation. Set `K_ij(s)=exp(-||phi_i(s)-phi_j(s)||²/2)` and row-normalize `.75 I + .25 K(s)`. Rule/continue/withdraw/hold remain isolated; an already-active repeated correction that the executor treats as queue continuation is isolated too. Multiply the13 flat residual logits by this matrix. No learned head, future outcome, extra planner query, reward change or extra action is introduced. The Critic and inherited SAC/BC/exploration update are unchanged.

The matrix is invertible at each state. Unlike the first constant matrix, state-dependent fixed preprocessing can change the finite network's function class; only trainable parameter count and pointwise output dimension are matched. It adds decision computation, recorded by the runtime. A static-reference chart (same kernel, treating reference displacement as zero) is the mechanism ablation if improvement appears. These are fixed predeclared scales tied to action units, not tuned on development scores.

This revision tests whether **measured target geometry**, rather than an action name, offers useful parameter sharing. It remains a prior-informed action-relation mechanism with close precedents (AGILE and action representation learning). Similar target poses are not identical Options: future reference updates, continuation, actual attachment and contact can still differ. Native complete-task outcomes, not the413 geometry checks, decide its value.


## Observation and objective limitations

An executable Option process with measured duration does not prove that the83-dimensional learner observation is Markov. It includes controller summaries and target/reference features, but not the entire physical simulator or every active queue waypoint. No tabular policy-improvement or safety theorem is transferred to these function-approximation results. The current Actor also uses successful-behavior anchoring, whereas greedy-Q extraction ignores that constraint; their deployment difference is not solely a measurement of Actor optimization error. Likewise the shaped/entropy objective is not identical to binary final native success. These are competing explanations to be distinguished by physical comparisons, not reasons to rename a new auxiliary head.
