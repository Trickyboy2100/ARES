# Successor retention and controlled exploration

**Independent RL method gain: UNVERIFIED.** Continued from `a2fa4a6`, same native Peg100 and persistent feedback runtime. [Open review](REVIEW.html) · [exact results](SUMMARY.json) · [costs](COST.json) · [declared comparisons and prior art](PROTOCOL.md).

## What changed, and what was actually learned

Online success BC and CQL were previously removed together. They are now independent switches; historical defaults produce exactly equal parameter updates on real replay fixtures. This round fixes the actual predecessor to grasp1 and trains only successor samples. Every evaluation starts from the native initial state and physically executes the whole prefix. No handoff reset, extra horizon, new action or weakened planner. The original task-conditioned predecessor, all earlier weights, and all negative evidence remain intact; these results do **not** establish a jointly improved predecessor/successor policy.

Four ordinary finite-SAC branches each execute32 new native training episodes and1280 actor/critic iterations. They share the same inherited checkpoint/replay and32 training initial states756000–031. Each then receives4000 additional off-policy iterations on its own actual replay, with no new physical training. One inherited neural initialization31501; behavior seed31601. All16 evaluation states753200–215 are previously seen development, not sealed or independent tests.

| Same-authority successor | After32 online episodes | After4000 further updates | Rescue / regression versus nominal9/16 |
|---|---:|---:|---:|
| BC removed, original sampling | 9/16 | 8/16 | 0 / 1 |
| Persistent successful-trajectory BC0.2 | 9/16 | 9/16 | 1 / 1 |
| Controlled sampling, BC removed | 9/16 | 10/16 | 2 / 1 |
| Controlled sampling + persistent BC0.2 | 9/16 | 9/16 | 1 / 1 |
| Posthoc phase/budget feedback rule | — | **12/16** | **3 / 0** |

Controlled sampling means epsilon0.1 uniform actions, otherwise deterministic Actor; original sampling uses the stochastic Actor. Deployment is deterministic throughout. This is a mature exploration control, **not** a reproduction or renaming of Policy Decorator. The posthoc rule selects existing action4 during refinement when≤28steps remain, otherwise continues the current queue. It uses the same state/actions/planner and was derived from seen physical evidence. Its12/16 is a reusable development control result, **not RL or unseen generalization**. No learning method exceeds it.

## Further method revision: successful is not necessarily nominal-preserving

Persistent BC can encourage corrections because all actions in a successful episode inherit the same outcome. On119 nominal-success visited states, inherited96-episode model mean replan probability is0.000154; after the persistent-BC online branch it is0.054716. This is a frozen-policy probe on actual support, not measured intervention success. [All checkpoint predictions](MODEL_PROBE.json).

We therefore trained additional strict common-model/common-replay comparisons, changing only the source of a64-row successful constraint batch (BC weight0.2). The RL replay always retains all successes and failures. Nominal anchors require a **complete successful episode with every successor action0**, not an isolated zero action after a correction.

| Real replay used | All-success anchors | Whole-episode nominal-success anchors |
|---|---:|---:|
| Original-sampling replay | 8/16, 0 rescue /1 regression | 9/16, 0 /0 |
| Controlled-sampling replay | 10/16, 2 /1 | 9/16, 0 /0 |

One bounded follow-up lowers only nominal-anchor weight0.2→0.05 on the controlled replay: **9/16, 1 rescue /1 regression**. Thus stronger anchoring restores nominal performance but suppresses recovery; weakening it reintroduces degradation. Neither continued BC alone nor this coefficient revision solves the retention/recovery tradeoff. These are mature BC/data-anchor controls, not an independent algorithm.

## Physical evidence and execution semantics

- Nominal failures in this fixed-prefix panel hand off at73–86steps, versus46–62 for nominal successes. The remaining budget is real, not extended by training.
- Controlled SAC rescues753204 and753212, but loses753209. Its successes are genuine continuous native executions. The strong feedback rule rescues753201,753212,753213 without losing nominal successes; it still fails753204. This leaves an observed conditional action/timing difference beyond one fixed feedback schedule, not proof of a new learning method.
- On753204, persistent BC uses one action4 at step86; REFINE3 ends90 rather than91 and the native result changes. Geometry and duration both change; the current evidence does not isolate a purely timing-based explanation.
- On753209, a learned action5 replan target is about160mm from the current TCP despite the parameter being named a3mm offset. **Target-offset planning is not a small bounded motor residual. Action0 continues the currently active queue; it does not restore the original plan.** Later reselecting a parameter after action0 can initiate another replan under the unchanged runtime.
- Nine additional native reruns (three selected scenes × frozen SAC/zero/strong rule) exactly reproduce all100 steps of action,q,qvel,TCP and peg state. [Repeat audit](REPEAT_CHECK.json). These confirm reproducibility of selected mechanisms, not an extra overall success estimate. All development prefix comparisons also have max measured error0.

[RAW / English rescue video and synchronized curves](video_rescue/SYNC_VIEWER.html):753204, strong rule fails while SAC succeeds. [RAW / English regression video](video_regression/SYNC_VIEWER.html):753209, strong rule succeeds while SAC fails. Both show101 measured-state frames, zero new physics, source SHA and pose readback checks. No fabricated rollout or omitted failure.

## Costs, reproduction and retained assets

**505 complete native episodes /50,500 physical steps**, comprising128 training episodes /12,800steps and377 evaluation/repeat episodes. Only48 distinct initial states:32 training +16 seen development. **41,120 new primary actor/critic iteration pairs**, including36,000 fixed-replay iterations;1336 effective training successor decisions.3841 actual planner queries,18.762s summed planner-query wall time (not end-to-end runtime). Diagnostic fixture updates and rendering are not counted as method training. No hidden prefix savings or simulated transition generation.

Keep the repeated-feedback runtime, strong shared planner, original SAC/STAP/Options, all new and old checkpoints, true nominal-success anchoring endpoint, exploration controls and strong phase/budget rule. Preserve the distinction: **new closed-loop control capability is demonstrated on seen development; learned nominal retention plus added success, and independent method superiority, remain unverified.** No new scene/seed expansion was launched because the learned improvement was not stable.

[Reproduction](REPRODUCE.sh) · [full raw/model archive and per-file SHA](RAW_MANIFEST.json) · [loss equivalence tests](LOSS_CHECK.txt). Archives include the common PARENT model/replay, all new physical traces, optimizer checkpoints, configurations, source snapshots and logs. Prior raw provenance remains in the previous `closed_loop` archives. SHA verification decompresses and checks every archived member. Pinned ManiSkill `62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3`; network-isolated single RTX3080 execution. A temporary third imported process was paused until the preceding evaluation exited; no run failed or was discarded. ARES-R remained read-only, no hardware, restricted field assets or sealed tests.
