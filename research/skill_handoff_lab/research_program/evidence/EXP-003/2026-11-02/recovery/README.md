# Recovery and cross-skill physics — 2026-11-02

**Independent RL method gain: UNVERIFIED.** Continuation of 525d67e. Open REVIEW.html for the key figure, complete videos, synchronized measured curves and raw sources. These are public simulator experiments, not robot deployment evidence.

## Scientific result

PullCubeTool100: with the strongest 87.5 mm grasp and shared controller, continue / rehook / retract-reapproach / press / wait / feedback all achieved 6/8 on the same eight development scenes (48 episodes). Early effective-contact interventions on four difficult cases all achieved 0/4 (16 episodes). No rescue or regression was found. Case 748103 fails TRANSFER planning before a successor decision; 748105 has unproductive tilted contact and unsuccessful rehook planning. A planner failure is not a proof of geometric infeasibility. Freeze these assets rather than expand recovery RL on this outcome-invariant panel.

We then actually ran native PlugCharger-v1, 200 steps, with real grasp pitch (15–45 degrees) followed by repeated executable advance (0.5–4 mm) and angular increments (0.01–0.08 rad). No observation bias or Y/Z offset action was introduced. All methods share the inherited depth-preserving geometric servo, integral correction, IK and joint increment limit. The inherited prefix stride of two is retained; this is not a new native trajectory safety claim.

In the 48-episode design panel, nominal / fast / soft / feedback scored 10/12, 7/12, 9/12, 8/12 across four scenes and three grasps. Some 45-degree planner prefixes differed: these are NOT strict causal pairs. We therefore replayed the saved nominal joint commands through fresh continuous physics, without setting physical states. Measured prefix state error was zero. The 12 matched and 12 factorial episodes distinguish angular collision from insufficient time: on 749101, nominal and faster advance succeed; faster angular alignment fails; very slow rotation or advance exhausts the available budget. Contact can occur while forward advance is disabled. The earlier 'fast advance causes contact' explanation is rejected. Joint clipping coincides with contact but has not been established as its cause.

The key figure's middle panel is the single-factor experiment, not the combined-fast trial. Its 5 mm position line is only one part of the unchanged native success condition. The additional CAUSAL_TRACE shows combined fast/soft controls; do not interpret it alone as an isolated-factor intervention.

## Real ordinary RL, including negative training attempts

A two-boundary ordinary SAC uses actual SkillResult/state, the same legal actions, remaining budget and finite-horizon gamma=1. Stage-zero unused action is masked. A failed grasp terminates the skill decision process while the physical episode still completes its native horizon. No candidate novelty is claimed.

We preserved an initial 97-transition smoke run (zero updates), then a 444-transition run (632 updates) with the inherited contact-only dense reward. The reward audit found that predecessor-dependent initial potential was omitted; contact-only shaping is not whole-task telescoping shaping. This is a baseline objective defect, not an independent method. Its policy behavior cannot be causally attributed to this defect alone from one training seed.

A corrected native-success-only SAC collected 1,024 recorded transitions, 1,792 update iterations, 100 full episodes / 20,000 steps plus 91 partial steps. Training seed 31181, scenes 749120–749135, 128x128 networks, alpha .02, learning rate .0003, batch64, warmup128, two updates per decision. INITIAL and 256/512/768/1024 checkpoints and buffers are retained. All 1,024 transitions were independently reconstructed from physical trajectories: maximum observation error 2.384e-7; action/reward/done errors zero. This is a bounded working baseline, not a convergence or multi-seed claim.

On eight new DEVELOPMENT scenes 749140–749147 (never sealed test):

| Method | Native final success |
|---|---:|
| Strong nominal, 15-degree grasp | 8/8 |
| Geometric contact force guard, same grasp | 8/8 |
| Legacy contact-reward SAC256 (objective diagnostic) | 4/8 |
| Native-success SAC256 | 3/8 |
| Native-success SAC1024 | 7/8 |
| Same 15-degree predecessor + SAC1024 successor | 8/8 |

Full SAC1024 has zero rescues and one regression relative to nominal. Its failure 749143 selects 18.384 degrees and fails to grasp; it is not an insertion failure. The same-prefix successor experiment removes this predecessor difference. No evidence establishes a useful advantage over the strong fixed policy.

We also executed the existing named ARES-AdaptivePlug-v0 construction-time density/friction variant: two scenes 749160/161, mass scales .5/2 and socket friction .2/1. The 15-degree nominal policy scores 8/8 conditions, versus 2/8 at 45 degrees. These are only two independent scene clusters, not eight independent tasks, and are NOT official native benchmark scores. Constructed masses are 14.5536 / 58.2144 g. Friction is configured, not experimentally identified. No additional network was trained on this covered panel.

## Evidence, costs and limits

354 complete episodes, 64,400 control steps, plus 207 saved partial steps. SUMMARY.json contains every source SHA, per-method wall time, planner query count/time, physical steps and training checkpoint budgets. Recorded training transitions total 1,565 and update iterations 2,424 across all three runs; do not sum checkpoint counters. Control selections for fixed policies are not RL optimization. Selected versus recorded decisions can differ at an interrupted transition. Both interrupted attempts, syntax-only failure, dependency snapshots and negative trajectories remain archived.

Charger contact force is a sampled 20 Hz proxy, not a 100 Hz peak measurement. Videos are measured-pose replay, with zero new physics, 201 frames at 20 fps (initial plus 200 steps), maximum pose readback error 1.192e-7. Geometric projected guides are not contact patches. video_contact isolates angular increment on identical 45-degree prefixes; video_learning compares the same initial scene with different learned/nominal grasp decisions. RAW means unannotated measured-pose render, not an original live camera recording. Static figures/video frames and encoded frame counts were checked; browser playback/click interaction was not automatically verified because local-file access was blocked by browser policy.

The tool early no-op event logger originally overwrote its first-event metadata. Analysis reconstructs the first event from force and interval cube movement and verifies prefixes; physical commands and original records remain unchanged. The source now prevents the overwrite. Evaluation source snapshots include an explicitly labeled post-launch dependency snapshot, not a pre-launch attestation.

Existing STAP, log-Q repair, strong controllers and prior evidence are retained. STAP/Foresight have not been newly evaluated on this charger action interface; no novel-method comparison is asserted. Single-seed training, small development panels and remaining predecessor failures preclude publication-level superiority claims.

## Reproduce / inspect

RAW_MANIFEST.json indexes compressed archives and every decompressed member SHA, models, buffers, source snapshots and visible assets. From repository root:

```bash
python research/skill_handoff_lab/tools/restore_post_contact_evidence.py --package research/skill_handoff_lab/research_program/evidence/EXP-003/2026-11-02/recovery --out /tmp/ares-recovery-copy --portable
python research/skill_handoff_lab/tools/analyze_recovery_physics.py --root /tmp/ares-recovery-copy
python research/skill_handoff_lab/tools/report_recovery_loop.py --root /tmp/ares-recovery-copy
```

Use the pinned environment in ENVIRONMENT.json. Physical reruns must use bash research/skill_handoff_lab/tools/handoff_offline.sh and the saved per-run protocol/source; this preserves network isolation and read-only ARES-R. Portable restore relocates JSON paths with original metadata retained, never rewrites physical gzip trajectories or weights.

## Decision

Retain strict physical-prefix replay, contact/action authority instrumentation, actual two-boundary native-success SAC and transition reconstruction. Freeze the tested tool recovery and charger pace panels: current improvements are covered by strong predecessor/control choices. The next informative experiment must expose a genuine changed successor contract or physically justified grasp constraint that leaves repeated full-task residual failures after the strongest shared control, before allocating further RL budget. Do not narrow grasp permissions merely to force failures, expand old yaw networks, or repair one 45-degree example indefinitely. The paper question stays task-conditioned multi-stage executable skill adaptation; independent algorithm gain remains UNVERIFIED.
