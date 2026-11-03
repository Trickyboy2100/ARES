# Discrete physical Options and task-conditioned handoff acquisition

**Independent RL method advantage: UNVERIFIED.** This release preserves an action-correct learner, genuine online physical updates, and three successive acquisition experiments. Prediction improvements are reported separately from policy success. No sealed test, hardware or restricted field data was used.

## Main result: frozen final policies

| Final policy | Seed31301 | Seed31302 |
|---|---:|---:|
| dsac | 4/8 | 5/8 |
| projected | 5/8 | 5/8 |
| frontier_policy | 4/8 | 4/8 |
| frontier_uniform | 4/8 | 4/8 |
| frontier_ees | 4/8 | 4/8 |
| frontier_handoff | 4/8 | 4/8 |
| frontier_local | 4/8 | 4/8 |
| frontier_independent_kernel | 4/8 | 3/8 |
| frontier_joint | 4/8 | 4/8 |
| matched_joint_supervised | 5/8 | 3/8 |
| matched_joint_fqi | 5/8 | 3/8 |
| matched_joint_iql | 3/8 | 4/8 |
| matched_joint_cql | 3/8 | 4/8 |
| matched_joint_stap | 4/8 | 5/8 |
| frontier_current_joint | 2/8 | 4/8 |
| frontier_current_independent_kernel | 3/8 | 4/8 |
| matched_current_joint_supervised | 5/8 | 3/8 |
| matched_current_joint_iql | 4/8 | 4/8 |
| mature_policy128 | 5/8 | 3/8 |

The V0 action-support defect is confirmed, but V1 correctness does not imply improved success. Original joint acquisition scores 4/8 / 4/8; EES-style control scores 4/8 / 4/8. The current-anchor revision scores 2/8 / 4/8. Larger ordinary SAC scores 5/8 / 3/8, while same-history supervised scoring scores 5/8 / 5/8. These comparisons do not establish an independent RL increment.

Eight predeclared development scenes751220–227; two optimization seeds share these same eight scene clusters. They are not16 independent test scenes. The original54 models were frozen before the first expansion rollout. A separately timestamped14-model current-anchor revision was declared from training-distribution evidence; those models were frozen before loading any expansion outcomes. It is an additional development experiment, not part of the original54-model protocol. No best-checkpoint selection. Fixed alternatives: fixed00=4/8; fixed01=5/8; fixed10=4/8; fixed11=4/8; fixed20=4/8; fixed21=6/8. Old-training selection scores are retained separately in `PRIOR_FIXED_SELECTION.json`; the first-tie training choice is fixed00, not the fresh-panel winner fixed21. The best fixed action on this panel is a hindsight benchmark, not an online oracle. Its6/8 equals the measured six-action union on this panel. Across the six monitor plus eight expansion scenes, the measured union is12/14 while the best single fixed action is10/14: conditional adaptation remains possible, but the tested learner/acquisition mechanism has not demonstrated that gain.

The original primary18 model/seed conditions and four additional current-anchor conditions each execute eight full native episodes. Their entire100-step traces must exactly match the corresponding finite physical branch. Remaining comparator policies use the measured48-branch panel: this is explicitly logged-policy evaluation, **not additional physical execution**. `expansion_eval/SCORES.json` labels each row. Matched-data supervised/FQI/IQL/CQL/STAP controls receive exactly the same acquired histories and update schedule. The128-episode ordinary learner uses a larger64-scene training distribution; it is a separate maturity diagnostic, not an equal-budget candidate comparison.

## What changed in the learning code

V0 used a continuous two-output actor and continuous critic while actual options are three grasp offsets(-10,0,+10mm), then two insertion targets(original/actual attachment). Sampled continuous actions fall off the executed support; projection changes Q and soft Bellman targets. Continuous differential entropy is not categorical option entropy. This is a semantic defect, not proof that it caused every failure.

V1 explicitly enumerates legal option IDs and masks action2 at the successor boundary. It retains actual SkillResult, option duration, native termination/failure and remaining budget. Gamma1 matches terminal native success under100 steps, with remaining budget observed; no hidden discount per option. DSAC, FQI, IQL, conservative-Q, conditional outcome scoring and functional STAP share54-dimensional privileged simulated state and the physical interface. The supervised label is the observed behavior continuation outcome, not an optimal successor oracle. STAP combines reach probability, a learned successor-state model and successor success in log space; it is a functional finite-option comparison, not the full original STAP training system.

The six analytic learner fixtures test code semantics only. Physical fixture replay reproduces old complete100-step traces exactly. `TRAINING_REPLAY_AUDIT.json` reconstructs real optimizer histories and verifies checkpoint parameters. `SHARED_CONTROL_AUDIT.json` checks identical warmup traces and MODEL_8 across acquisition methods. Discretization, normalization, LayerNorm,50:50 offline/online mixing and mature algorithms are retained infrastructure, not claimed contributions. The current tuning budget does not establish asymptotic SAC convergence.

## One acquisition hypothesis, three actual revisions

1. Initial handoff advantage/uncertainty sampling did not produce stable final gains. With only two successor actions its ranking can collapse to the non-incumbent action; a shared budget multiplier cannot resolve ranking.
2. Expected-update acquisition enumerates four possible paired outcomes, performs a virtual suffix-Q update and a predecessor Bellman update, and scores the expected change in the initial option decision. Real training remains ordinary DSAC. Fixed-prefix repeated sampling wasted budget on deterministic labels; the next revision gives all allocators the same no-repeat coverage constraint and permits new predecessor options.
3. Pair outcomes are correlated. On29 old training handoffs from10 scenes, leave-one-scene-out prediction gives observed discordance0.138, independent estimate0.358, joint estimate0.152; counterpart Brier0.313→0.182. This motivated a frozen five-neighbor joint model and an **exact same-marginal independence ablation**. These prediction numbers are not RL gains. The only changed acquisition component is outcome weighting; no neural auxiliary head, reward, observation or controller change.

Virtual acquisition uses one critic, two SGD steps(.01), greedy-Q decisions and approximate outcome probabilities; actual training uses twin critics, Adam and an actor. This surrogate mismatch is a concrete limitation, tested against actual actor changes and old-panel return in the optimizer replay audit. Expected error reduction, value of information, competence practice, uncertainty sampling and nominal fallback are known ideas. No novelty is asserted from combining their names.

4. An acquisition-objective defect was identified before inspecting the new expansion outcomes:60/68 scoring anchors are duplicated old design initials. A single-factor revision scores only the eight already observed current training initials, while retaining identical prior replay, optimizer, outcome kernel, controller and24-episode budget. Joint and same-marginal independent kernels are both physically retrained for both seeds. This tests task-distribution relevance rather than adding another loss/head. Protocol and freeze timing are preserved in `DISTRIBUTION_REVISION_PROTOCOL.json` and `anchor_eval/FROZEN_MODELS.json`.

## A further mechanism check: deployed policy versus critic decisions

This was declared **after observing development results**, with no new training: it is a posthoc baseline mechanism confirmation, not an unseen test. On current-anchor joint seed31301, Actor/min-Q/mean-Q yield2/8,4/8,5/8. On128-episode ordinary seed31302 they yield3/8,2/8,5/8. Six extraction conditions were physically executed for48 native episodes, and all100 steps agree with their real finite branches. Thus Actor approximation and clipped-critic aggregation have different effects; a single explanation such as insufficient Actor updates is not supported.

The independent-kernel current-anchor model also reaches6/8 under mean-Q in logged finite-branch evaluation (not another direct rollout). Changing extraction is a standard baseline control, **not a new RL algorithm**. Acquisition was scored using one critic's greedy decisions while deployment used a learned Actor: that mismatch is now a concrete mechanism to resolve before claiming a data-acquisition contribution. The value of the collected data and the way it is used by the deployed learner must be separated. See `POLICY_EXTRACTION_AUDIT.json` and `EXTRACTION_COMPARISON.json` for true rescues and regressions, not just net success.

## Physical evidence and fairness

Official PegInsertionSide-v1, pinned ManiSkill62ff3a5, native100 and native success. Shared official geometry with inherited GPU-OBB correction, speed2, all three refinements. No task weakening or scene reset at handoff. Every paired prefix is fully reexecuted and charged200 physical steps per pair. Early planner exits are completed with terminal holds to100; their failures remain in the dataset.

Compatible prior:60Peg episodes/118 option transitions, same controller and reward semantics. Tool, old stale-OBB runs, measured-insert variants and incompatible controllers are excluded. Training751000–015; matched acquisition751000–007; larger ordinary baseline751000–063. Monitor dev751200–205. Previous10 design and4 seen dev scenes are not described as fresh tests.

All models see the same privileged simulator state; this is not a sensor-only deployment claim. The fixed action(0mm,original target) already solves6/6 monitor scenes. No algorithm gain can be established solely from that saturated monitor. Actual strict-prefix examples show that refreshed attachment can rescue751000 and degrade751002. They establish conditional physical effects, not a method advantage.

## Real cost

Audited completed native episodes: 2458; physics control steps: 245800. Primary critic optimizer iterations across real experiment models: 250400 (audit reconstruction and analytic fixtures excluded). This includes training, fixtures and evaluation; the table separates training. Each full physical rollout is100 steps. Planner counts are actual plan_screw queries; internal IK iterations are not separately measured. Concurrent GPU wall times are not isolated speed benchmarks. Virtual acquisition compute, training time and optimizer steps are retained in ACQUIRE/HISTORY/checkpoints. Optimizer reconstruction is audit work and is excluded from training budgets.

| Run | Train episodes / decisions | Train physics steps | All planner calls | Planner seconds |
|---|---:|---:|---:|---:|
| frontier_current_independent_kernel_31301 | 24 / 47 | 2400 | 333 | 1.7 |
| frontier_current_independent_kernel_31302 | 24 / 47 | 2400 | 333 | 1.7 |
| frontier_current_joint_31301 | 24 / 47 | 2400 | 333 | 1.6 |
| frontier_current_joint_31302 | 24 / 47 | 2400 | 333 | 1.6 |
| frontier_ees_31301 | 24 / 43 | 2400 | 321 | 1.7 |
| frontier_ees_31302 | 24 / 43 | 2400 | 321 | 1.6 |
| frontier_handoff_31301 | 24 / 47 | 2400 | 333 | 1.8 |
| frontier_handoff_31302 | 24 / 47 | 2400 | 333 | 1.7 |
| frontier_independent_kernel_31301 | 24 / 47 | 2400 | 333 | 1.6 |
| frontier_independent_kernel_31302 | 24 / 47 | 2400 | 333 | 1.6 |
| frontier_joint_31301 | 24 / 47 | 2400 | 333 | 1.7 |
| frontier_joint_31302 | 24 / 47 | 2400 | 333 | 1.6 |
| frontier_local_31301 | 24 / 45 | 2400 | 327 | 1.7 |
| frontier_local_31302 | 24 / 45 | 2400 | 327 | 1.7 |
| frontier_policy_31301 | 24 / 45 | 2400 | 327 | 1.6 |
| frontier_policy_31302 | 24 / 45 | 2400 | 327 | 1.7 |
| frontier_uniform_31301 | 24 / 47 | 2400 | 333 | 1.7 |
| frontier_uniform_31302 | 24 / 45 | 2400 | 327 | 1.7 |
| mature_policy_31301 | 128 / 240 | 12800 | 1065 | 5.2 |
| mature_policy_31302 | 128 / 245 | 12800 | 1082 | 5.2 |

## Review and reproduction

Open `REVIEW.html`. Every scientific panel is also available separately in `figures/`. Videos are full101-frame measured-state visualizations of saved initial+100 physical states,20fps; no new physics is generated by rendering. RAW is unannotated pose playback, not a live RGB sensor recording. English versions and clickable frame-synchronized axial curves are linked individually.

Restore this package and the preceding `peg_value` package using `restore_post_contact_evidence.py`; verify archive/member SHA in `RAW_MANIFEST.json`. Source snapshots accompany each run; the final source is in the repository. Explicitly pass restored prior/root paths when reproducing. Offline runtime is invoked only through `handoff_offline.sh`, using the pinned local method-evolution Python environment. Never run production ARES-R entry points.

```bash
# Read-only evidence verification / deterministic optimizer reconstruction:
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/analyze_discrete_handoff.py --root RESTORED_RUN
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/audit_handoff_controls.py --root RESTORED_RUN
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/audit_option_training_replay.py --root RESTORED_RUN --prior RESTORED_PEG_VALUE
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/summarize_handoff_results.py --root RESTORED_RUN
```

To collect a new replication (not reuse stored labels), run the pinned Python through the isolation wrapper:

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/run_frontier_acquisition.py --prior RESTORED_PEG_VALUE --out NEW_RUN --mode joint --seed 31301
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/run_frontier_acquisition.py --prior RESTORED_PEG_VALUE --out NEW_CONTROL --mode independent_kernel --seed 31301
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/run_discrete_handoff.py --mode online --prior RESTORED_PEG_VALUE --out NEW_ORDINARY --seed 31301 --allocation policy --episodes 128 --train-start 751000 --train-count 64 --eval-every 32
```

The raw archives retain unsuccessful episodes, failed audit attempts and corrected audit code, all checkpoints, acquired buffers, source snapshots and frozen protocol. See `PRIOR_ART.md` for pinned source review and limits: EES comparison is a finite-option competence/extrapolation analogue, not a full reproduction of symbolic EES; RLPD-inspired mixing is not full RLPD; RecoveryChaining source was not located. These limits prevent a publication-level superiority claim over those original systems.
