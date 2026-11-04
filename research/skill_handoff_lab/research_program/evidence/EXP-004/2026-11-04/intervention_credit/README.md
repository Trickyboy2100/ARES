# Intervention credit and continuation-policy failure

**Independent method gain: UNVERIFIED.** Continued from `cc6774c`; implementation initially committed as `5ba09f8`. Same strong PegInsertionSide-v1 native100 controller, real grasp1, eight feedback actions and64 observations. No hardware, pose-reset evaluations, sealed tests or field data. [Key result figure](figures/RESULTS.png) · [physical mechanism](figures/PHYSICAL_EFFECT.png) · [exact results](SUMMARY.json) · [costs](COST.json).

## Actual algorithm development

1. Added read-only **pre-intervention full joint queue/cursor/last-action/hold-state** evidence. Each contrast reexecutes the entire native prefix and checks measured action, q, qvel, TCP, peg and queue equality. Full queues support auditing; deployed observations remain the existing64 features. No privileged queue information is secretly added to one method.
2. Implemented an Actor update that substitutes a measured intervention-versus-continuation return gap for the tested Critic action gap. No new head, external gate, BC coefficient search or planner change. Ordinary SAC and an **AWAC-style hybrid discrete actor update** share all physical replay and the same additional decision-state batch. This is not a full original AWAC benchmark reproduction. First-step Critic updates are equal while Actor gradients differ; historical SAC/BC defaults remain exact.
3. Replaced fixed zero-continuation labels with **current-Actor continuation**, freezing the Actor across both full-prefix branches, then updating on fresh labels. All methods pay for the acquisition rollout plus both complete native branches. This is a genuine online algorithm change, not state restoration.
4. Tested direct signed pair log-odds training after observing that the effective action initially has probability **1.7976e-6**. Its expected-SAC gradient is correspondingly tiny. Direct preference training changes the actual selected intervention, but does not preserve its successful continuation. Tied terminal outcomes supply no signed preference.
5. Further1000-update, common-model/common-replay consolidation compares mature advantage weighting with the retained measured preference memory. Neither yields a full-task gain. Pairwise preference learning, AWAC, Q filtering and rollout improvement are established ideas; this prototype is **not a claim of originality**.

All legacy auxiliary heads remain updated for checkpoint compatibility and are unused by ordinary policy/backups. They were neither expanded nor used to claim progress. [Detailed protocol and primary references](PROTOCOL.md), [reproduction](REPRODUCE.sh).

## New natural development: complete native tasks

Fixed development seeds757200–223 were new at the start; now seen development, never sealed tests. Final checkpoint is the declared endpoint; intermediate results are diagnostic. One inherited neural initialization31501, not a multi-seed result.

| Comparison group | Method | Success /24 | Rescue / regression vs nominal |
|---|---|---:|---:|
| Frozen references | Nominal queue continuation |17|0 /0|
| Frozen references | Prior phase/budget rule, unchanged |**18**|**1 /0**|
| Same fixed-pair replay,1000 updates +16 online episodes each | Ordinary SAC |10|0 /7|
| Same fixed-pair replay | AWAC-style |17|0 /0|
| Same fixed-pair replay | Measured-gap candidate |16|0 /1|
| Current-policy pairs,8 rounds ×3 full episodes each | Ordinary SAC |17|0 /0|
| Current-policy pairs | AWAC-style |17|0 /0|
| Current-policy pairs | Measured-gap candidate |17|0 /0|
| Current-policy pairs | Direct measured preference |17|0 /0|
| Same preference endpoint/replay,+1000 updates,0 new training episodes | AWAC-style consolidation |17|0 /0|
| Same endpoint/replay | Preference-memory consolidation |17|0 /0|

Budgets are equal **within each group**, not between groups. The fixed-pair SAC10→current-pair SAC17 difference is not an independent-method comparison: training data, decisions and update budgets differ. Online algorithms may collect different states after their policies diverge; equal-budget is not identical-online-data. Fixed-pair offline endpoints were SAC15, AWAC-style16, measured-gap16. Eight paired rounds are exploratory, not a claim that ordinary RL is fully converged. There is no evidence justifying multi-seed superiority or joint predecessor adaptation yet.

## Physical findings and the falsified explanation

- Fixed-prefix training states757000–007:48 initial contrasts gave0 rescues/7 regressions;24 late-refinement contrasts added0/5. Frozen-rule continuation on757008–015 yielded21 valid contrasts,0/6; one scene never reached the requested boundary, and all four charged full executions are retained. These are clustered contrasts, not93 independent tasks.
- Current-policy sampling reaches a genuine positive contrast on757300: at step59, action1 (measured-attachment target) succeeds versus continued queue failure.757302 gives a genuine destructive contrast. All four current-policy learning variants see the same initial positive and negative cases, before later data collection diverges.
- **The direct preference learner actually selects the successful first action on757300, yet its complete rollout fails.** It replans at steps68,71,73 and93; the successful witness mainly continues the queue after step59. The first69 actual physical steps match exactly. This rules out “the actor simply never learned the first intervention” as the explanation for this specific failure. It does not isolate each later replan's individual effect.
- All four frozen current-policy endpoints fail757300 and retain757302 in eight additional selected native reexecutions. These are training-case diagnoses, not new development scores. Memory consolidation remains17/24; it provides no evidence that more repeated local-preference training solves the continuation problem.
- Action0 continues the active modified queue. Repeated action1 within a queue may simply continue it, but entering another refinement can trigger another whole-path replan. A target offset is not a bounded motor residual.

[Full RAW/English video: actual positive intervention](video_training_effect/SYNC_VIEWER.html) · [Full RAW/English video: learned action, failed continuation](video_continuation_failure/SYNC_VIEWER.html). Both are101-frame **MEASURED_STATE_REPLAY**,20fps,0 new physical steps; source SHA and readback error1.2e-7 are recorded. Static frames were inspected and video metadata checked. Interactive browser playback was not verified.

## What can be retained for the paper

**RETAINED:** exact measured-prefix/queue pairing; native online paired-credit trainer; same-interface AWAC-style control; all checkpoints; direct-gradient evidence; continuation-dependent failure with complete physical traces. These are usable implementation and experimental assets.

**LOCAL_NEGATIVE:** tested single-decision gap replacement and preference updates do not exceed mature controls or the frozen rule. Improved Actor selection alone does not establish task improvement. No sample-efficiency or independent algorithm contribution is established.

**OPEN_MECHANISM:** a local rollout advantage is conditional on the continuation policy. Updating a shared Actor changes both the intervention and that continuation. The concrete next method question is whether policy improvement can retain the measured beneficial continuation while adapting other decisions, compared fairly with ordinary policy trust regions and successful-witness imitation. This is not a license to rename state-dependent BC as new RL. Restore predecessor learning only after a real successor gain; multi-skill paper scope is unchanged.

## Costs and reproducibility

**616 complete native episodes /61,600 control steps;4543 measured planner queries;21.627s summed query time** (not end-to-end runtime). **9480 primary Actor/Critic iteration pairs**, plus inherited auxiliary-head iterations; diagnostic fixture updates excluded.24 saved model/checkpoint files.32 unique learning initial states,8 additional training-design probe states and24 development states. Paired collection128 episodes (96 shared learning,32 continuation probes); online training144 episodes; evaluation/selected verification344 episodes. Every paired prefix is charged. GPU peak was not sampled; no fabricated peak-memory claim.

No positive/negative run was discarded. Native planner numerical warnings remain diagnostic failures, not an excuse to modify the shared planner. No extra physical query, longer horizon or broader action set is granted to the candidate. [Full archive hashes](RAW_MANIFEST.json), [parent provenance](PARENT_SOURCE.json), [Actor/Critic probes](POLICY_PROBE.json), [continuation counterexample](CONTINUATION_FAILURE.json). Prior source/weights and all negative experiments remain unchanged.
