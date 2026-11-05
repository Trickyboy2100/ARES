# Rule-relative complete Option learning — 2026-11-05

**Actual trained methods and complete native physics; independent online-RL gain is not established.** Base `9a30507`; initial implementation `680a88a`; exact executed source/model hashes and command scripts are archived. No ARES-R writes, hardware commands or sealed tests.

Open [REVIEW.html](REVIEW.html) for the results, independent figures and complete RAW/English-annotated synchronized videos. [SUMMARY.json](SUMMARY.json) is recomputed from terminal simulator records; [COST.json](COST.json) separates physical and optimizer costs. The [LaTeX table](RESULTS_TABLE.tex) is ready to include in the manuscript.

## Main result

| Method | Dev /16 | Frozen confirmation /24 | Rescue / regression vs shared rule |
|---|---:|---:|---|
| Frozen rule, shared predecessor | 9 | 16 | 0 / 0 |
| Frozen rule, fixed grasp1 | 9 | 16 | 1 / 1 |
| Frozen complete macro 31701 | 7 | — | — |
| Frozen complete macro 31702 | 10 | — | — |
| Historical Q gate | 8 | — | — |
| Fixed first proposal1 | 11 | 16 | 4 / 4 |
| Shared supervised initialization | 11 / 11 | 19 / 18 | 3/0 ; 3/1 |
| Online supervision | 10 / 10 | 18 / 19 | 2/0 ; 3/0 |
| Supervision + shared-data rehearsal | 11 | 19 | 3 / 0 |
| Online sequence RL | 12 / 12 | 19 / 18 | 3/0 ; 3/1 |
| Entry-only RL ablation | 12 | 19 | 3 / 0 |

Two numbers are neural initializations31801/31802, **not independent scene cohorts**. Confirmation has24 natural scene clusters (763400–423), fixed final64, no best-checkpoint selection. All primary comparisons share the inherited learned three-grasp predecessor; fixed-grasp1 is an additional stronger-reference check. Macro controls use the common predecessor and their original frozen rear policies. The historical-Q gate and full macros were compared on the16-scene development cohort, not rerun on confirmation. All old final64 scores remain unchanged.

The important attribution is negative: the two shared initializers already score19/24 and18/24, with exactly the same per-scene success outcomes as their respective final RL models. Online supervision scores18/24 and19/24; ordinary shared-data rehearsal reaches19/24. Thus the apparent12/16 versus10/16 development advantage does not establish an independent RL improvement. The learned conditional interface does avoid three of the four regressions of fixed first-proposal adoption for seed31802, and all four for31801, while retaining three of its four rescues. This is preliminary state-conditioned selection evidence, already explained by supervised paired-data initialization.

## What was implemented and actually trained

- A real single-deviation executor: choose a complete inherited Option, preserve its active queue/reference and release rule, then return to the frozen strong rule. The gate's wait action means **follow the rule**, not physical action0.
- 107-input,32-hidden-unit shared proposal scorer; two existing proposals, legal-action masks and a default rule logit. The83 execution observations, controller, native100 horizon and original success predicate are shared.
- 64 new paired training conditions763000–063:128 original-reset executions,3 rescues/7 regressions/54 unchanged outcomes. All seven regressions have positive historical mean-Q gaps. Old behavior Q is not a calibrated value for this different continuation contract.
- Actual online own-policy training on763100–163. The candidate uses measured complete return relative to a separately executed same-initial rule baseline, with twofold regression cost and clipped sequence likelihood-ratio updates. The supervision control uses observed Option effects; the rehearsal control retains the common paired dataset. Both have the same allowed information and physical collection budget.
- First iteration: common warm500 updates, then64 episodes each for RL and supervision. The initializer underfit all three positive training examples. A training-only convergence check justified warm2000 for both arms in iteration2, then two independent network initializations. One-factor entry-only credit and supervised rehearsal were run, not merely proposed.

**512 own-policy training episodes**,13000 shared-initialization optimizer steps across arms,2336 online optimizer steps,40 named weight checkpoints. Each online comparison arm is charged128 pair-collection episodes +64 counterfactual baseline episodes +64 own episodes before development/confirmation. Shared physical data are not treated as free. Total unique execution: **1528 complete native episodes,152800 physical steps,14644 planning queries**. Summed episode wall time9228.75s is not elapsed wall time because jobs ran concurrently; optimizer wall time was not separately instrumented. There were no partial physics attempts in this cycle's raw inventory. See COST.json for exact timing.

## Mechanism findings and limits

1. **Waiting-credit hypothesis did not engage.** Every nonzero update of seed31801 selected proposal1 at the first disagreement; there were no preceding wait decisions in those updates. Full-sequence and entry-only training therefore produce identical64 training +48 development physical trajectories. This is not evidence for a new stopping-time RL algorithm. [CREDIT_ABLATION.json](CREDIT_ABLATION.json), [NONZERO_UPDATE_SUPPORT.json](NONZERO_UPDATE_SUPPORT.json).
2. **Queue continuation is not rule preservation.** At training763039, identical79-step physical prefixes diverge when rule action10 is replaced by four steps of physical action0. Insertion begins at84 versus88 and terminal axial positions are+35.3mm versus−98.3mm. This isolates a real harmful deviation, but does not isolate geometric change from timing as its sole cause.
3. **The physical prefix includes executor state.**310 adoption contrasts passed equality of the full saved joint queue, cursor, last action, hold/failure state, phase, active target and actual observation, in addition to collection-time physical-prefix checks. No PhysX state reset was used. [EXECUTION_PREFIX_CHECKS.json](EXECUTION_PREFIX_CHECKS.json).
4. **Predecessor effects remain real but are not newly learned here.** The shared frontend chooses grasps0/1 in the collected data (all three remain legal). On confirmation, fixed grasp1 rescues763404 and breaks763412 relative to the shared frontend, leaving the same16/24 total. The separate two-condition front probe changes no terminal result. New joint predecessor/successor RL superiority remains untested.
5. **Second-task repair is control, not RL.** Tool post-hook comparisons give4/8 for none/feedback/rehook. The early-contact probe exposed an inherited bug: declined feedback discarded the nominal caller/queue, and corrections could run before the event. The new `program_tool_recovery_v2.py` preserves the old nominal100-step trajectories in all8 contexts, preserves all100 steps for three actual declined corrections, and intervenes only after the measured contact prefix. Corrected early results: none3/8, feedback3/8, press2/8 (0 rescues,1 regression). The old early feedback0/8 and press0/8 are retained as invalid-continuation diagnostics, not a fair algorithm comparison. No second-task RL was trained on an interface with no measured recovery gain. [TOOL_CONTINUATION_CHECKS.json](TOOL_CONTINUATION_CHECKS.json).

## Reproduction and artifacts

The complete public native-simulation data, models, manifests, pending transaction records, exact source variants and executed shell commands are in the SHA-verified [RAW_MANIFEST.json](RAW_MANIFEST.json) chunks. Local original run root:
`/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/rule_option_20261105`.

Extract every RAW archive into that run root (or remap its recorded absolute paths). Frozen parent macro weights and parent SAC are dependencies from the previous atomic_handoff/success_retention evidence; exact paths and hashes are in PROVENANCE.json. The pinned ManiSkill revision remains 62ff3a5896b4d5b4cf0ac4c8d79afe600c9404a3 and the existing method-evolution environment is reused. Do not install a newer simulator to reproduce these scores.

Run `tools/analyze_rule_option.py --root RUN` and `tools/audit_tool_rule_continuation.py --root RUN` to recompute terminal outcomes and contracts; `tools/plot_rule_option.py --root RUN --out FIGURES` regenerates the figures. Actual training/evaluation commands are the archived `run_v2_*.sh`, `run_followup_*.sh` and `run_fixed.sh`; use a new output directory for fresh collection. All physics commands use `tools/handoff_offline.sh` to unshare networking and protect ARES-R. Transaction resume supports completed durable episode records; a partially interrupted PhysX episode is explicitly inspected, never silently reused as a causal prefix.

Videos replay101 measured frames (initial +100 controls); they add **zero physics episodes**. Both RAW and annotated MP4s, exact frame data, sourceSHA and clickable synchronization viewers are independently accessible:

- [763410 learned adoption](videos/adoption_763410/SYNC_VIEWER.html): rule failure versus trained adoption success; the shared initializer also succeeds, so this is not an online-RL gain.
- [763039 harmful queue continuation](videos/continue_763039/SYNC_VIEWER.html): rule success versus a four-step deviation. This is a training mechanism case, not independent testing.

All figures have separate PNG and PDF files in `figures/`. Literature/code comparisons and theory boundaries: [PRIOR_ART.md](PRIOR_ART.md). The replay scripts use exact saved poses; they never substitute rendering for physical evidence.

## Retained assets and next discriminating experiment

Retain the trained conditional Option-adoption interface, shared paired-data initialization, same-prefix executor verification, own-policy training/checkpoints, frozen first-proposal baseline and Tool's corrected continuation. Do not expand the waiting-credit mechanism: its distinguishing gradient was absent in the useful training cases. Promote the initializer and supervised rehearsal to strong learned references rather than repeatedly counting their gains as new RL.

Next, use new training-only natural initial conditions to test whether changing the real predecessor while holding this learned successor fixed creates reproducible task-conditioned ordering changes beyond the best fixed grasp. Only then train a matched ordinary two-stage learner and a concrete successor-conditioned increment. The present24 confirmation cases are now seen development evidence; they must not be reused as fresh confirmation. No SCI-level generalization, new safety theorem, or independent algorithm advantage is claimed from this cycle.
