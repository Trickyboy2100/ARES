# Continuation-explicit Options: three actual learning iterations

**Independent method gain: UNVERIFIED.** [Open review](REVIEW.html) · [raw-recomputed results and cost](SUMMARY.json) · [model SHA](MODEL_SHA.json).

From `c4219f2`, implementation `758ab2c`. PegInsertionSide-v1, unchanged official100 steps and success, shared strong planner, real grasp1, one RTX3080, network/device-isolated simulation. No ARES-R edits, hardware, sealed tests or restricted site captures.

## What is established

The learned first intervention in757300 succeeds when its continuation is preserved. Native reexecution with the new event-terminated Option exactly reproduces **all100 physical steps** of the previous successful witness: action, q, qvel, TCP and peg pose. Same actor with repeated decisions fails; phase-only commitment also fails. This is a shared **temporal-interface/control improvement**, not original RL. See [exact comparison](WITNESS_REPRODUCTION.json) and [RAW/English synchronized video](video_mechanism/SYNC_VIEWER.html).

Action0 continues the current modified queue. It does not undo an intervention. The Option executes the selected nonzero physical primitive and then a frozen reference actor; it reopens the learned controller on observed grasp loss or an exhausted INSERT queue. Refinement phase changes alone do not terminate it. Observations are sampled at inherited feedback boundaries, up to4 physical steps apart. Native100 always terminates. The reference can itself use the same legal primitives; it is not an unmodified trajectory replay. Episode-long and phase-only variants are diagnostics. The new interface is available to ordinary SAC, BC, candidates and the frozen-rule control.

SMDP transitions aggregate actual reward, next state and duration only between policy-over-Option boundaries. Gamma remains the inherited finite-horizon1. Old primitive replay is **not** relabeled as new Option data. Inherited neural weights remain initialization, and frozen continuation model SHA is recorded. Full active queues remain raw evidence; the shared Actor sees the inherited compact64-dimensional state, not future outcomes.

## New natural development, not sealed testing

758200–758215 were selected before outcomes. They are now seen development. Final predeclared checkpoints, not best checkpoint selection:

| Method | Round1 | Round2 | Round3 |
|---|---:|---:|---:|
| Nominal continue | 7/16 | shared | shared |
| Frozen original feedback rule | 8/16 (1 rescue,0 regression) | shared | shared |
| Frozen rule through new Option interface | 7/16 | shared | shared |
| Standard Option-SAC | 7/16 | 7/16 | 6/16 |
| Successful-witness BC + SAC | 6/16 | 6/16 | — |
| Local measured preference + suffix KL | 4/16 | 4/16 | 6/16 |
| Joint suffix-likelihood revision | — | 3/16 | — |
| Termination-consistent SMDP credit revision | — | — | 4/16 |

All learned final variants have **zero development rescues**; their deficits are regressions relative to nominal. Ordinary primitive SAC and untrained inherited Option actor both7/16. Therefore the known witness repair does not establish distribution-level improvement, and the event contract can remove the frozen rule's one rescue. Do not claim this interface universally dominates repeated feedback.

## Actual algorithm changes and falsification

1. Shared-data native online Option-SAC, success-BC, and signed measured intervention preference with reference-KL on the winning continuation's later Option boundaries.80 Actor/Critic iterations plus40 Actor-only iterations per round for every method. BC uses all real successful replay, not only the paired winner.
2. From the same round1 ordinary checkpoint, compare local version against **joint continuation likelihood ratios** along both actual branch suffixes. This changes the Actor gradient, adds no head, and keeps identical new physical data. It does not improve generalization.
3. From the same round2 ordinary checkpoint, replace full frozen-rollout credit with the difference between each actual Option's shaped return plus current soft tail value at its own termination. This removes the semantic substitution of a reference-policy complete return for current-policy one-step advantage, but leaves learned tail-value error. The final4/16 is below ordinary6/16.

Round1 uses758000–007, round2 uses758008–015, round3 uses758016–023. Each round fully reexecutes both native prefixes; initial state, pre-action physical sequence, observation and complete active queue must match exactly. No PhysX state restoration. There are24 such pairs:3 rescues,4 harmful interventions,17 ties, under the actual behavior continuations. These are **not** an exhaustive action-success union or a task ceiling.

All arms within a round share the exact buffer, physical budget and80+40 update schedule. Later rounds inherit accumulated compatible Option data and start all arms from the preceding ordinary checkpoint, preserving failed models separately. Additional candidate objectives mix half base SAC and half evidence loss; ordinary uses the full base SAC loss. BC adds its declared0.2 term. Thus local-vs-joint is the cleaner same-form ablation; no inference of an advantage from optimizer counts alone. The same legacy auxiliary networks are updated for compatibility but never select actions or supply the ordinary policy backup.

**Concrete learning, but overgeneralization:** round1 candidate learns actions3 and5 at758002/758004 (probabilities0.89/0.99). Full native retakes succeed on both; BC also succeeds. On selected diagnostics758000/002/004/005/757300, candidate5/5, BC4/5, ordinary2/5. This selected training diagnosis is NOT generalization. On development the candidate chooses physical action5 first in12/16 scenes and loses758204/205/214; see [behavior evidence](INTERVENTION_BEHAVIOR.json). Continued training cannot be judged by its ability to fit those few positive interventions.

SMDP target logs retain actual measured native delta separately from estimated advantage. Even tied complete outcomes receive nonzero estimated differences through current tail values. These estimates are not new physical success evidence. No theoretical monotonic-improvement guarantee is established for any candidate.

## Costs and retained assets

484 complete native episodes /48,400 physical control steps;48 paired training episodes on24 natural starts,436 control/evaluation/diagnostic episodes.6,400 primary Critic iterations,9,600 Actor iterations including3,200 extra Actor-only iterations;6,400 inherited auxiliary iterations separately.90 checkpoint/model files.3,847 planner calls;18.058s summed planner-query time;2,304.698s summed episode wall time (overlapping jobs, **not** elapsed wall-clock). One inherited neural initialization31501; no multi-seed or convergence claim. GPU peak memory was not instrumented.

292 accumulated training Option transitions include48 predecessor and244 successor decisions. The predecessor is truly executed but fixed to isolate successor learning; joint predecessor training and a second task remain unfinished. Current buffer coverage is too limited to call the new Option learner converged.

Permanent assets: shared continuation runtime, reward/duration contract checks, three actual trainable update variants, same-data baselines, strict native-pair acquisition, all intermediate weights and failures. Method novelty and generalization remain open. Do not expand the failed joint-preference loss unchanged or revert to fixed branch scoring. The next concrete issue is decision timing/state generalization and reliable current-policy tail evaluation under these preserved Option semantics.

## Evidence and reproduction

- [Results figure](figures/RESULTS.png), [fixed-checkpoint learning curves](figures/LEARNING.png).
- [Known-witness RAW/English video and synchronized physical curves](video_mechanism/SYNC_VIEWER.html).
- [New learned training rescue RAW/English video](video_training/SYNC_VIEWER.html). BC also succeeds; no independent-method claim.
- Videos are101-frame measured-state playback,20fps,0 additional physics; maximum pose readback1.1921e-7. Raw simulation, not playback, supplies success labels. Figures and final frames were visually inspected; interactive browser QA was not performed.
- [Commands](REPRODUCE.sh), [prior-art boundaries](PRIOR_ART.md), [archive manifest](RAW_MANIFEST.json). Each run saves exact launch-time source, protocol, raw100-step traces and models. Prior third-party source copies remain local; source URLs/revisions/SHA are recorded, without republishing unlicensed code.


## Separate shared-runtime correction after the learning study

Raw audit found4 intermediate checkpoint rollouts with `IndexError` from a planner `Success` carrying an empty joint path, plus1 actual planning failure. These remain in the archives and the original curves; no final checkpoint had the empty-path exception. The shared runtime now executes one measured-current-q hold step for a successful empty path, records it and continues under the same100-step budget. A separate native reexecution of758204 changes failure to success, with no Actor update. This is a controller/runtime repair, **not RL gain**; no historical scores were rewritten. Controller ID changes to `empty_success_hold_v1`. [Exception list](EXECUTION_ERRORS.json), [separate physical comparison](EMPTY_PLAN_FIX.json), and `empty_plan_fix/RAW_MANIFEST.json`. Add1 validation episode/100steps to the484-episode main study. Reproduce historical scores with commit758ab2c or archived launch source; latest runtime includes this later correction.
