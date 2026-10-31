# Development comparison: what the completed pilots establish

Snapshot: the first six policies on 12 conditions (seeds 729000–729003 ×
socket-observation lateral bias −1.2/0/+1.2 mm), before the zero-Z and LayerNorm
follow-up results. These conditions are now inspected **development** data.
The native geometry, success test and 200-step horizon remain; the observation
distribution is custom, not an official native PlugCharger score.

| Policy | Complete terminal success | First native-criteria step, successful cases only |
|---|---:|---:|
| Geometric depth-hold | 4/12 | 128.50 |
| Online RLS controller | 6/12 | 140.67 |
| Progress-based search | 10/12 | 141.70 |
| Residual SAC pilot | 1/12 | 149.00 |
| Effect-feature SAC pilot | 3/12 | 167.33 |
| Same effect policy with prior-only features | 3/12 | 168.00 |

All six policies have zero prefix failures on this group. Every cross-policy
pair has the same recorded physical prefix boundary on all 12 conditions,
including q/qvel, tool, TCP, socket and velocities. This verifies recorded
boundary equality, not inaccessible internal PhysX solver-state equality.
No sources or privileged native-stop results are pooled into these scores.

Compared with geometric control, search rescues 6 and regresses 0. Compared
with search, effect SAC rescues 1 but regresses 8; ordinary SAC rescues 0 and
regresses 9. Effect SAC versus ordinary SAC rescues 3 and regresses 1, but
this small difference does **not** establish the proposed mechanism: replacing
the effect inputs with the unchanged prior gives the identical 12 binary
outcomes (zero rescue, zero regression).

The effect posterior is not universally dormant. Exact saved-source replay
finds nonprior features on 132/316 effect-evaluation decisions (41.77%) and
110/1024 effect-training decisions (10.74%). Nevertheless, activation alone
has no demonstrated terminal benefit. In the prior-only policy, the 139/316
potentially nonprior computations are counterfactual diagnostics: those
computed updates were **not fed to the policy**. Ordinary SAC receives no
effect features either. The analyzer distinguishes this from actual policy
input use.

The effect/prior-only successes are exactly the same three conditions:
729002/+1.2 mm, 729003/0 mm, and 729003/+1.2 mm. First native-criteria steps
are respectively 134/195/173 for effect and 134/196/174 for prior-only. Two
one-step differences do not constitute a meaningful method gain.

First criterion satisfaction is auxiliary: geometric control has two cases
and ordinary SAC has two cases that satisfy the native criterion transiently
then fail at step 200; search has one. All remain counted as failures. No
early transient goal is substituted for the full-task terminal measure.

## Next discriminating analyses

* Zero-Z SAC evaluation tests whether an unnecessary lateral degree of freedom
  causes avoidable degradation. It is an action-interface ablation of an
  already trained policy, not evidence that an equally trained reduced-action
  policy would succeed.
* The bounded LayerNorm SAC baseline tests a known optimization improvement.
  If it helps, this strengthens the baseline and cannot be claimed as a new
  ARES RL mechanism. The pilot must still be labeled limited-budget and
  single-seed, rather than a converged comparison.
* Before expanding training, compare policies on the search failures and on
  the observed transient-success losses. A useful adaptive mechanism must
  preserve solved cases and exploit actual contact information in the few
  residual cases; it cannot justify eight regressions by highlighting one
  rescue.

The final analyzer automatically includes subsequent zero-Z and LayerNorm
runs. This note deliberately freezes the first comparison; updated exact
counts belong in `EVAL_COMPARISON.json`, `SUMMARY.json` and later notes.

Source: SHA-verified episode trajectories and source snapshots indexed in
`RUN_INDEX.json`/`EPISODES.json`; pairing in `PREFIX_PAIRS.json`. Crash recovery
keeps all four old training episodes (800 physical steps), deduplicating the
mistaken recovery of an already indexed episode. Neither rendering nor this
CPU analysis adds simulator interactions or optimizer updates.

## Zero-Z diagnostic follow-up

The same frozen ordinary SAC policy with its Z residual forcibly set to zero
scores **7/12**, rescuing six original SAC failures without regressing its one
success. Prefix records match exactly for all 12 conditions. Against search
it rescues two cases but regresses five (7/12 versus 10/12). Successful cases
first meet native criteria at mean step 139.43; no transient-success terminal
loss occurs in this diagnostic.

This supports avoidable degradation from the policy's unconstrained Z output
on the Y-only observation-bias family. It does not establish that Z adaptation
is universally unnecessary or that restricted-action training would reach
the same result. The intervention changes available actions at evaluation;
it is reported as an explicit ablation, not a like-for-like new RL method.

## LayerNorm protocol deviation retained

The first `ln_sac_pilot_1024` invocation omitted the entropy override, and its
saved manifest records alpha **0.1**, whereas ordinary/effect SAC used **0.02**.
It is therefore a mixed LayerNorm-plus-entropy intervention, not a pure
LayerNorm ablation. Its data and compute remain counted. A separately named
`ln_sac_alpha002` run is planned with explicit alpha 0.02; both runs must be
retained and reported. Evaluation alpha is recovered from the checkpoint's
training manifest, never inferred from an unused evaluation CLI default.
No best-checkpoint/model-selection claim follows from these development runs.

## Completed LayerNorm controls and repeatability (interim)

The first mixed LayerNorm/alpha-0.1 pilot scores **4/12**. The separately
trained alpha-0.02 LayerNorm baseline scores **10/12**. Against the ordinary
alpha-0.02 SAC pilot (1/12), matched-alpha LayerNorm rescues nine conditions
and regresses none. Against the strongest search controller (10/12), it
rescues two and regresses two, yielding **no net success advantage**. All
12 paired prefixes are exactly equal in the recorded state variables.
Changing LayerNorm alpha from 0.1 to 0.02 rescues seven and regresses one;
this comparison is retained as an entropy sensitivity result.

This is evidence that a known optimization baseline materially changes this
small-budget result. It is not an independent ARES method contribution,
not a converged RL baseline, and not a multi-seed generalization conclusion.
The effect mechanism lacks a positive ablation result under the original
optimizer; a fair same-LayerNorm effect-versus-prior comparison is now
warranted before generalizing that negative result to improved optimization.

Matched-alpha LayerNorm's two terminal failures are both seed 729001, with
bias 0 and +1.2 mm. Both establish contact and transiently satisfy the native
criterion (steps 154 and 161), then fail the step-200 test. Peak recorded
tool/socket force norms are respectively 43.16 and 49.48 N. These observations
identify terminal retention/contact control as a useful next diagnostic;
they do not establish a force limit violation or unique causal explanation.
The success-only mean first-criterion step is 146.4. Search instead fails
729000/+1.2 and 729002/+1.2: complementarity exists, but an oracle union is
not a deployable policy score.

Four new cold repeats (seed 729000, biases ±1.2 mm, geometry and search) match
the original full 200-step trajectories exactly for all audited q, qvel,
TCP/tool/socket poses, attachment, velocities, action, force norm and success
flags. They confirm these particular control outcomes are repeatable; they
do not establish global simulator determinism. See `REPEATABILITY.json`.

The interim snapshot before the effect-plus-LayerNorm pilot totals **357
complete episodes, 71,400 physical steps, 30,254 prefix steps, 7,606 logged
skill transitions and 3,584 optimizer update iterations**. Four training-tail
episodes remain in physical cost and outside policy-success denominators.
The four episodes of the failed serialization run contribute 800 steps and
zero optimizer updates; the duplicate recovery contributes none. Each SAC
update iteration includes separate actor and critic optimizer steps, not one
combined optimizer call. No GPU/rendering work was performed by this analyzer.

### LN terminal-failure timing: no hold was entered

All rows below use seed 729001. Step numbers count recorded physical control
steps, starting at one. Final force is the measured tool/socket force norm.

| Policy / Y bias | First / last native success step | First terminal-hold step | Final position / angle error | Final force |
|---|---:|---:|---:|---:|
| LN SAC / 0 mm | 154 / 189 | none | 13.245 mm / 0.03677 rad | 9.547 N |
| LN SAC / +1.2 mm | 161 / 172 | none | 10.199 mm / 0.00820 rad | 30.783 N |
| Search / 0 mm | 122 / 200 | 123 | 3.525 mm / 0.00462 rad | 2.125 N |
| Search / +1.2 mm | 156 / 200 | 159 | 3.338 mm / 0.00717 rad | 9.259 N |

The LN trajectories remain in CONTACT_SERVO through step 200. Thus these two
failures are neither premature terminal-hold entry nor slipping during a
fixed-joint hold. The controller continues to modify the tool after transient
native success. The observed stopping contract is ≤4 mm and ≤0.03 rad; the
native criterion permits ≤5 mm and ≤0.2 rad. Reconstructing the biased task
frame from every recorded tool/socket pose finds **zero** observed-contract
matches in either LN episode. During their native-success periods,
orientation errors are approximately 0.037–0.143 rad (0 mm bias) and
0.058–0.101 rad (+1.2 mm), so the stricter observed orientation requirement
rejects them even when insertion position is close.

This isolates a concrete contract-versus-terminal-objective mismatch and
continued control in the recorded failures. It does **not** prove that holding
at the transient native success would retain the plug, or authorize a
privileged true-goal stop. A next intervention should use an explicitly
observable dwell/retention contract, applied equally to search and learned
policies, with the full 200-step physical outcome measured again.

## Final matched-optimization effect test

The effect-feature policy was also trained with the same critic LayerNorm,
alpha 0.02, 90-input interface and 1,024 decision budget as the stronger
baseline. It scores **2/12**, versus matched plain LN SAC **10/12**: zero
rescues and eight regressions. Its prior-only input ablation also scores
**2/12**, with the same two successful conditions and zero paired outcome
changes. Against search, effect LN rescues one and regresses nine.
Recorded physical prefixes remain exactly paired on all 12 conditions.

This controlled result removes the original weak-optimizer explanation for
the feature result **within this one training seed and limited budget**.
It does not show that every execution-effect model, other feature design,
training seed or longer training run would fail. It does show that this
particular compatibility feature has not earned a positive method claim;
the strongest established result remains the known LN SAC/search baselines.

The feature is active on 88/1,024 training decisions and 96/301 evaluation
decisions. The prior-only ablation computes potentially informative features
on 100/301 decisions for offline diagnostics but feeds **zero** nonprior
updates into the policy. Activation is therefore real, yet terminal gain
from supplying these updates remains absent.

### Does the feature estimate the hidden offset usefully?

CPU-only diagnostic truth is the hole offset relative to biased observed
geometry, `-bias_mm`. It is never supplied to policies or training. The saved
feature mean is multiplied by 1.5 mm and compared with this truth only on
activated decisions. Errors below are decision-weighted 2D-vector RMSE;
sign accuracy excludes zero-bias components. Adjacent decisions are correlated.

| Source trajectories | Activated decisions | Feature RMSE | Zero-prior RMSE | Correct nonzero-offset sign |
|---|---:|---:|---:|---:|
| Effect LN training | 88 | 0.995 mm | 1.023 mm | 36/64 |
| Effect LN development evaluation | 96 | 1.016 mm | 0.995 mm | 21/66 |
| Prior-only evaluation, counterfactual feature replay | 100 | 1.002 mm | 0.982 mm | 26/67 |

The development feature estimate does not improve over its zero prior on
this diagnostic. Its broad mean ±1.96 standard-deviation boxes contain truth
for all activated decisions; this is **not probabilistic calibration**:
the weights are heuristic geometric compatibility scores and the broad boxes
do not demonstrate accurate concentration. Exact per-episode numbers and
source hashes are in `QA.json`; pooled diagnostics are in
`EFFECT_INFERENCE.json`. The feature-inference weakness and terminal-prior
ablation together motivate revising the observable contact/retention contract
before adding another network or expanding this feature experiment.

Final physical accounting after this matched test: **426 complete episodes,
85,200 physical steps, 36,069 prefix steps, 9,232 logged skill transitions,
and 4,480 optimizer update iterations**. Five training-tail episodes are
retained in cost. All source/hash/action/readback checks pass; four selected
cold repeats have identical complete recorded trajectories. Zero additional
physical or optimizer steps were introduced by evidence analysis or rendering.
