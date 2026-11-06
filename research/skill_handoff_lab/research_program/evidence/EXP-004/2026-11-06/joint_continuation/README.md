# Joint predecessor–recurrent successor learning: 2026-11-06

Open [REVIEW.html](REVIEW.html) for figures, complete videos, synchronized curves and evidence links.

**Result:** actual joint RL is implemented and trained; none of the three learning revisions establishes an independent improvement over its strong learned initialization and matched ordinary RL. This is a completed bounded research cycle, not a convergence claim.

## What changed and what was learned

The predecessor really selects one of three physical grasps (−10/0/+10 mm); the recurrent successor reopens decisions after a complete inherited Atomic Option. All methods keep native Peg100, actual grasp/SkillResult, strong shared planning and identical execution authority. Rear selector0 executes the strong rule chunk (which can issue physical10); physical0 continues the current active queue. The two proposal selectors are state-dependent frozen parameterized policies, **not free access to every member of the twelve-action library**. No handoff object reset.

1. Fresh critics and trainable front/rear residual logits replace frozen front/one-deviation learning. Ordinary mean-twin discrete SAC versus recent suffix MC/TD mixing: two independent new-network/exploration seeds, same pretrained base and scenes. Both variants lose initial successes. Mixing real returns into predecessor targets did not solve policy improvement.
2. Four executed-action aliases had different learned Q despite identical physical continuations. Four native reruns verify exact 100-step equality for two alias pairs. An execution-conditioned shared scalar critic removes this algebraic inconsistency; a flat critic receives the same identity information. Shared critic ends7/16 versus flat9/16. Correct representation is not sufficient for improvement.
3. Whole-trajectory paired-return policy gradient uses the actual sampled joint continuation, not a fixed-suffix advantage injected into a changing Actor. A root-state frozen-policy return is an action-independent baseline; clipped repeated policy updates remain a mature approximation, not a new safe-improvement theorem. Zero terminal difference causes no optimizer step. Equal-physical-budget SAC receives the same reference data. Both finish9/16; PG preserves15/16 complete initial deployment traces, SAC12/16. No new deployment successes. Training rescue/regression is3/3 for each versus its reference, not generalization gain.

## Fixed-final-checkpoint results

| Arm | Own episodes | Initial → middle → final |
|---|---:|---|
| identity_flat_32001 | 64 | 9 → 8 → 9 /16 |
| identity_semantic_32001 | 64 | 9 → 7 → 7 /16 |
| paired_pg_32101 | 32 | 9 → 9 → 9 /16 |
| paired_sac_32101 | 32 | 9 → 9 → 9 /16 |
| return_mix_31901 | 64 | 13 → 12 → 11 /16 |
| return_mix_31902 | 64 | 13 → 12 → 11 /16 |
| sac_31901 | 64 | 13 → 11 → 11 /16 |
| sac_31902 | 64 | 13 → 12 → 10 /16 |

The first four arms use development764300–315; execution-identity and paired-return arms use764600–615. These cohorts must not be pooled or used for between-iteration causal claims. All checkpoints shown; no best-checkpoint selection. Initializations for the first comparison share the pretrained base and data; later revisions have one training seed each, not a multi-seed confirmation.

## Physical opportunity and competing explanations

On sixteen new training-design scenes764000–015, fixed grasps under rule score7/8/8 and measured union10; under warm successor8/9/8 and union10. Scene764008 requires measured grasp2, whereas764000/013 fail with grasp2. This demonstrates real conditional predecessor effects. Improved successor rescues other grasps in764004, reducing the remaining predecessor opportunity. It does not establish strict successor-induced ranking reversal.

Selected training rescue764712 was followed by three actual full-task interventions: grasp1+warm succeeds, grasp0+first correction2 succeeds, and both succeed (grasp0+warm fails). Thus either component alone suffices; the visually striking joint training rescue is **not evidence of joint necessity or synergy**. Changing grasp also changes handoff geometry and remaining budget, so budget alone is not isolated as cause.

Four parameterization errors at training scene764157 are retained failures of the common planner, not silently dropped. No geometric-impossibility conclusion is drawn from them.

## Frozen historical policies: new natural confirmation

On764400–431 (32 new natural scenes), strong rule19/32, paired initializer20/32, prior final RL21/32, supervised replay20/32. Prior final RL rescues764400/764430 versus rule with no regression. Scene-cluster bootstrap95% difference interval for prior RL versus rule is[0,0.15625]; initializer versus rule[0,0.09375]. One extra case over initialization is preliminary and cannot establish a robust independent method advantage. The previous24-scene historical results are unchanged. These32 scenes are now seen exploratory confirmation, never sealed tests.

## Cost, reproducibility and data isolation

1163 complete native episodes;116300 physical control steps;9857 planning calls. 448 are own-policy training episodes across8 arms. 16880 actor and16640 critic optimizer steps (paired PG240 actor-only; paired SAC1280 actor+critic). Summed episode wall time7346.062s and planner time50.113s are cumulative timings, not elapsed runtime. GPU snapshot is not peak memory. Inherited training cost remains in the previous ledger, not attributed to this cycle.

Training scenes:764100–163,764200–263,764700–731; design764000–015; development764300–315 and764600–615; frozen confirmation764400–431. No sealed data. Shared96 design episodes were reconstructed from actual executor origins and checked by four complete native reexecutions, not incompatible transition relabeling. Third iteration adds32 shared real reference episodes. Charge shared data to each consuming method; unique cost counts it once. Protocols include the prelaunch shared-data amendment. Environment, code snapshots, seeds, pending/current transaction files, raw failures, optimizers and checkpoints are retained.

[SUMMARY](SUMMARY.json), [cost](COST.json), [actual support](SUPPORT.json), [deployment changes](DEPLOYMENT_CHANGE.json), [prior art and theory](PRIOR_ART.md), [archive SHA/member manifest](RAW_MANIFEST.json). Raw archives contain run scripts, histories, complete compressed native trajectories, manifests, model/optimizer/RNG checkpoints and source snapshots. Extract all RAW_*.tar.gz into one empty directory; absolute local paths in run scripts need relocation. Python environment is pinned in ENVIRONMENT.json. Do not run robot entry points.

Set `PYTHONPATH` to the pinned ManiSkill source directory from ENVIRONMENT.json (the isolation wrapper already does this). Recompute with repository tools `analyze_joint_continuation.py`, `audit_joint_support.py`, `audit_joint_alias_values.py`, `plot_joint_continuation.py` using `--root <extracted-run-root>`. Native reproduction uses archived run scripts through `handoff_offline.sh`; no internet/device networking is required by physics.

## Videos and limits

Both videos contain101 measured states at20fps, no new physical steps. Pose readback maximum1.79e-7. They are labeled MEASURED STATE REPLAY and not independent physics trials; success is from their archived native trajectories. RAW, English annotation, source SHA and clickable per-frame curves are supplied separately.

- [Training764712 synchronized comparison](videos/training_764712/SYNC_VIEWER.html): sampled training rescue, not greedy final improvement; factorial qualification above.
- [Regression764606 synchronized comparison](videos/regression_764606/SYNC_VIEWER.html): shared-critic final regression; grasp1→2 and subsequent decisions both change, not isolated grasp causality.

No second-task RL transfer was run. Existing Tool/Charger review is retained; saturated or unrecoverable small panels are not an excuse to claim transfer. Full-catalogue action coverage, stable independent improvement, three-or-more seeds, second-task transfer and paper-level generalization remain open.

## One next scientific decision

Preserve the warm learned baseline, genuine joint runtime and all failed models. Test whether legal physical Options omitted by the two frozen proposal policies create repeatable residual recovery opportunities at real handoffs, under the same strong controller and shared authority. This decides whether to train a fuller executable-action interface; do not expand MC mixing, alias loss or paired PG merely because their algebra is plausible. Independent algorithm gain remains UNVERIFIED; methods and negative experiments are usable paper material now.


### 2026-11-07 initialization erratum (historical scores unchanged)

Direct checkpoint comparison shows joint_continuation sac_31901 and sac_31902 INITIAL Actor, Critic and target tensors are identical: the shared Gate constructor resets Torch RNG to31801 after the requested seed. The two runs have distinct exploration RNG, **not independent neural initialization**. Earlier independent-new-network wording is superseded. Their actual outcomes and negative evidence remain unchanged. The next full-catalogue learner reseeds after loading shared assets and verifies different hidden/Critic weights while keeping cross-arm initialization matched.
