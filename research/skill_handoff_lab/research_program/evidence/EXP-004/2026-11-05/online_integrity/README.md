# Own-policy Option learning and target-origin intervention — 2026-11-05

**Independent method gain: UNVERIFIED.** This cycle adds actual acquisition/training and a falsified action-semantics hypothesis, not a new named RL method. Source baseline `a8418f9`; published implementation `0d533b4`; exact launch code is in each run's source snapshot. ARES-R/hardware/sealed tests untouched.

`ARES_SCI_RESEARCH_TASKBOOK.md` was absent from the synchronized branch and searched local workspace/download locations. A path was requested asynchronously. This cycle follows the latest explicit user instruction plus existing AGENTS/STATE/PAPER_CONTRACT; it does **not** claim to have read the missing taskbook or satisfied an unknown packaging schema.

## Native online results

New training759000–759063; development759200–759215; mechanism759500–759503; resume758900–758901. These groups are separate. Development is now seen, not sealed testing. Each arm collects its **own** real episodes and trains only on its own replay. This differs from the previous shared-data continuation experiments. No common-data loss-isolation claim is made here: policy-dependent data are part of the online comparison.

| Method | Initial | After32 train | Fixed final64 | Rescue / regression vs nominal |
|---|---:|---:|---:|---:|
| Nominal continue, actual grasp1 | — | — | 9/16 | 0/0 |
| Frozen historical feedback rule | — | — | 8/16 | 1/2 |
| Ordinary Option-SAC | 9/16 | 10/16 | **6/16** | 1/4 |
| Persistent successful-behavior BC + SAC | 9/16 | 8/16 | **9/16** | 1/1 |

Both learning arms share legal3-grasp/8-feedback actions,64 observations, strong planner, empty-success-path fix, frozen continuation reference, native100, dense shaping and5120 Actor/Critic iterations. Persistent BC is a **mature comparator**, not an original candidate. No new auxiliary head or coefficient sweep. The inherited legacy auxiliary networks still receive equal compatibility updates but do not select actions or change ordinary Bellman targets.

The shared inherited neural initialization is31501;31601 controls replay/exploration. **Not two independent training seeds**, not convergence, not a statistically established method advantage. Final checkpoint was predeclared; SAC's intermediate10/16 is preserved but not selected as the main result.

Actual predecessor decisions are restored: SAC training grasp counts0/1/2 =16/38/10; BC21/35/8. Full physical predecessor execution precedes every successor. In mechanism scene759501, grasps0/1 succeed and grasp2 fails under zero intervention. This proves real parameter influence in that context, not learned general joint optimality.

## Target-origin mechanism and competing explanation

Four natural scenes × three real grasps, with full native reexecution for each branch. Eight contexts reach feedback; four fail or exhaust budget before it. All are included in denominators. Identical available prefixes, observation and full current queue are checked; no PhysX state restoration. Twelve repeated zero branches reproduce1200 physical steps exactly.

| First feedback intervention; then continue active queue | Native successes /12 | Rescue / regression vs zero |
|---|---:|---:|
| Zero | 5 | 0/0 |
| Refresh actual attachment only, no offset | 0 | 0/5 |
| Refresh +3mm / −3mm Z | 0 / 0 | 0/5 each |
| Active requested target +3mm Z | 2 | 0/3 |
| Active requested target −3mm Z | 5 | 0/0 |

The existing nominally3mm commands move the requested target **5.74–21.98mm** from its active origin in the reachable contexts. Remaining-target commands preserve requested orientation and translate it3mm in the goal frame. Median first-four-step commanded joint-vector differences from zero shrink from~0.0135/0.0110rad to~0.00344rad. These are joint-vector norms, not per-joint limits or TCP displacement.

The zero-offset refresh control also destroys all five nominal successes: **attachment re-estimation/replanning is a competing cause, not merely the sign of a small residual**. This experiment does not isolate target orientation versus translation versus time parameterization. Remaining-target +3mm still destroys three successes, and neither sign adds a success beyond nominal. Thus smaller requested displacement is **not** a success guarantee. The implementation anchors the *requested terminal target*, not FK of the actual queue endpoint; it remains a full replan, not a bounded motor residual. It is opt-in; the default controller and both online arms stayed unchanged.

[Same-prefix RAW/English video and synchronized curves](video_target_origin/SYNC_VIEWER.html):759501/grasp0,+3mm; rebuilding target fails, remaining-target version retains nominal success.101frames,20fps, measured-state playback,0new physics; max pose-readback error1.1921e−7. This is retention against an intervention-induced failure, not a new rescue relative to nominal.

## Integrity and exact scope of resume

- Immutable per-arm manifest pins source hashes, parent model, observation/action/controller, train/dev seeds, update budget and final-checkpoint rule. Distinct own-policy replay verified by `BUFFER_AUDIT.json`.
- Atomic model/target/all-optimizer/replay/labels/RNG/job-cursor checkpoint; durable episode transaction before updates. Startup lock prevents duplicate run writers. Raw episode counters never reset onto existing filenames.
- Actual interruption after collection but before update reproduces all network/optimizer tensors, Python/NumPy/Torch-CPU and replay RNG, transitions and200native steps of uninterrupted execution. No repeated episode. Completed resume is idempotent; changed seed rejected.
- **Mid-PhysX crash is not exact state resume.** An uncommitted START fails explicitly so attempted partial costs must be recovered before proceeding; no silent duplicate or fabricated zero cost. Filesystems fsync/power-loss recovery were not tested.
- A narrow audited V1→V2 migration changes only provenance checking/idempotent completed resume. Unrelated Git documentation commits do not invalidate identical source; all other code/config differences still reject. Frozen execution sources remain unchanged.

## Cost, evidence and scientific decision

344 complete native episodes /34400 physical steps:128online train,96checkpoint evaluations,32frozen controls,84mechanism/refresh contrasts,4resume-proof episodes.10240 main Actor/Critic iterations each plus16 resume-proof iterations; legacy auxiliary updates separately equal10256.958 effective main training decisions, including128 real predecessor decisions.2501planning queries; summed planning wall11.94s; summed episode wall1887.40s is **not elapsed time** because runs overlap. GPU peak was not instrumented.

Retain own-policy online/resume infrastructure, stronger truthful action instrumentation, opt-in remaining-target operator, all learned failures and complete source/model data. Do not relabel BC or relative target coordinates as originality. Main unresolved learning issue remains stable policy improvement: SAC loses four nominal successes at its final checkpoint, while BC retains count but exchanges one success for one failure. The target-origin result justifies a **separate future frozen-interface online comparison**, not retroactively changing this group's runtime. Continue from these real models/replays; do not rerun the failed preference loss or broaden budget heads.

See `SUMMARY.json`, `RAW_MANIFEST.json`, `LOCAL_INVENTORY_SHA.json`, `REPRODUCE.sh`, `PRIOR_ART.md`, and `REVIEW.html`. Full local root: `/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/online_integrity_20261105`. Public archives contain all344trajectories, exact launch sources, key models and final complete resume states; every intermediate transaction/resume checkpoint remains local and SHA-indexed. No restricted field data included.
