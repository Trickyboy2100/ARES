# EXP-003 — post-contact action effects and effect-direction SMDP SAC

**Decision: independent method gain remains UNVERIFIED.**
Open [REVIEW.html](REVIEW.html) for the result table, physical mechanism figures, complete RAW/English videos and clickable frame-linked curves. This is actual public ManiSkill physics, not hardware or a synthetic success table. The original paper question and all previous assets remain intact.

## What the experiments established

1. The old height audit confused selected with executed parameters: **11 selected grasp contexts**, not9; only8 reached the actual HOOK/PULL executions. Height did not change complete success in this panel; acceleration changed one context. The supplied audit was corrected and actually run. Historical statements are superseded, not erased.
2. Native100 contains a post-contact decision window. The initial PULL boundary is **not necessarily the first physical contact**. In22 audited nominal contexts,14 had earlier moving contacts; median PULL-boundary delay19.5steps, maximum35. `NO_EFFECTIVE_CONTACT_BOUNDARY` in old files means no qualified PULL decision boundary, not proof of no physical contact.
3. Four bounded actuator experiments (endpoint lateral/Z, absolute local yaw, measured-relative yaw, axial/pitch) initially showed continuous effects but no binary rescue in their extreme cases. A failed case can have the cube about50mm below a tilted tool, while successful hooking places it about5mm below; a force/motion trigger is not a successful-hook contract.
4. Ordinary SAC failures identified a genuinely actionable region. Strict matched-prefix interventions rescued748028; interventions also destroyed successful nominal continuations. In the subsequent120-episode panel (12 new design scenes ×2 grasps ×5 selected actions),14/24grasp contexts reached the PULL correction boundary. **3 fixed-prefix contexts were rescued;4 nominally successful contexts could be harmed.** However, all rescues were already covered by70mm nominal grasping: both the best nominal-grasp coverage and joint grasp/contact oracle are8/12. The oracle is retrospective coverage, not a deployed policy.
5. The earlier-contact experiment was executed, not deferred. Its8 zero-action episodes exactly equal the original complete100-step traces;16 nonzero comparisons share identical physical prefixes. A second24-episode run used the **same transient correction/rejoin primitive at PULL**, removing the original continuation-semantics confound. For748058/40mm, first-contact negative yaw ends at598mm while positive yaw ends632mm; at PULL the same primitive ends828/606mm respectively. Different actual contact states/phases produce different direction rankings. The early rescue has only≈2mm native margin and is not a robustness claim; task coverage beyond the stronger grasp is still not increased.

## Real learning and controls

The candidate learns a frozen finite-difference **terminal cube-effect map** from measured paired trajectories, then uses it to orient the existing2D parameter box for ordinary SMDP SAC. It does not reduce or enlarge the legal action box, change native reward, add future information, or replace the strong controller. It is a representation hypothesis, not established novelty. The map predicts full-horizon effects including failures/budget, not an instantaneous contact Jacobian.

- Official `PullCubeTool-v1`, original100-step horizon and native cube/base distance criterion; GPU physics0 only.
- Stage0 legal grasp35–105mm; stage1 post-contact lateral±15mm and measured-relative yaw±0.15rad. A shared infinity-norm≤0.1 no-op region continues the original native trajectory.
- Same54D actual state/effect/budget information. The unused stage0 coordinate is masked in actor, entropy, critic and stored replay. `gamma=1` with observed finite-horizon budget implements undiscounted SMDP return; physical durations are recorded, not treated as equal physical time.
- Common physical data:190 full episodes/19000control steps,320 replay transitions. Its matched prefixes include repeated predecessor transitions; these are not320 independent scenes. One training seed31171; identical actor/critic128×128 sizes, critic LayerNorm, alpha.02, LR3e-4, batch64.
- All four SAC representations receive the same500-update offline warm start. Ordinary and conditional then each execute256 real RL decisions with2gradient iterations per decision; save OFFLINE/128/256 weights and buffers. Ordinary uses141 full episodes+70partial steps; conditional142+75. **Decision budgets match; physical cost differs by105steps (about0.74%).** Global-direction and generic-effect-auxiliary controls are compared only at the common offline budget, not misrepresented as fully trained online ablations.
- The first exploratory ordinary run was intentionally stopped at145decisions/162gradient iterations. Its93 complete episodes,28partial steps, checkpoint128, interrupted weights/buffer and failure traces are retained. It is not a completed256-decision run.
- STAP uses the retained CEM/Q-product kernel with newly fitted actual-boundary Q and transition models; direct supervised scoring shares the same physical data. Their1000joint optimization iterations and model-query costs are separate. This is a bounded functional comparator, not official STAP benchmark/checkpoint reproduction or proof of convergence.

### Whole native task, same8 new development seeds748100–748107

| Method | Shared500-update stage | +128 online decisions | +256 online decisions |
|---|---:|---:|---:|
| Ordinary same-interface SAC |6/8|6/8|6/8|
| Conditional effect-direction SAC |5/8|5/8|6/8|
| Global effect direction |5/8|not run|not run|
| Generic effect auxiliary |5/8|not run|not run|
| STAP functional core |6/8 (1000joint updates)|—|—|
| Direct supervised score |4/8 (shared comparator fitting)|—|—|

Fixed70mm, fixed105mm and the tested conservative feedback rule each score4/8. A subsequent fixed87.5mm midpoint control scores**6/8**, matching both trained policies; it diagnoses nearly constant grasp selection and is not an untouched final-test result. The inherited five-candidate geometric pregrasp lookahead scores1/8 and is explicitly **not** selected as the strongest baseline.

The256-checkpoint **same87.5mm physical-prefix** comparison is nominal6/8, ordinary6/8, conditional6/8, with7 exact post-contact prefixes and one common earlier failure. Candidate vs ordinary: **zero rescues, zero regressions**, both in the full-task comparison and matched-prefix comparison. The remaining common failures are a planning failure and a tilted-tool/unproductive-contact case; they are not evidence for a new RL mechanism.

This is a single training seed and8 development scene clusters, not sealed testing, cross-seed convergence, or paper-level generalization. The matrix's12 scenes and selected timing cases are different training/design evidence, not extra independent evaluation samples.

## Prediction is not control gain

The first18D standardized-state ridge was worse than a constant effect predictor: selected leave-scene-out squared error0.02530 vs0.00374. Replacing it with physically scaled actual tool-frame cube position, attachment, budget and remaining tool goal reduced the selected training-scene CV error to0.001301. All model versions and coefficients remain archived.

However, among22 contexts with nontrivial effect, best-direction-axis prediction changes only18/22→19/22; much of the improvement is effect magnitude. The256-checkpoint terminal-value MSE on7 actual-contact development cases is ordinary0.00442 vs candidate0.09200. These networks are unbounded value estimates, **not calibrated probabilities**. See [predictions vs physical outcomes](analysis/PREDICTIONS_VS_PHYSICS.json).

The representation Jacobian test passes, but it applies **before** the common no-op execution projection. It does not establish differential entropy for executed actions including a no-op atom. Earlier protocol wording saying exact physical-action density is corrected by [the retained addendum](analysis/ENTROPY_SCOPE_ADDENDUM.json); no training outcome was edited.

[Prior-art scope](analysis/PRIOR_ART.md): LASER already learns action representations; RAPS parameterized primitives, ILA contact/action latents, and PA-RL potential-field parameters prevent claiming coordinate learning itself as originality. No extra novelty name is introduced.

## Costs, evidence and reusable assets

[SUMMARY.json](analysis/SUMMARY.json) / [RESULTS.csv](analysis/RESULTS.csv) are recomputed from raw physical files and include planning calls/time, per-method full-task scores, paired outcomes, model SHA and costs. Overall: **858 full episodes/85800 control steps +173 saved partial control steps**. SAC3186gradient iterations mean3186actor and3186critic optimizer steps; comparator fitting adds1000joint optimizer steps. Shared offline trajectories are counted once globally. GPU rendering adds0physics steps. These totals span multiple research loops, not858 independent test scenes.

Both256 SAC replay buffers (576transitions each, including shared data) and the initial128buffer were reconstructed from actual initial/contact/terminal states and actions; errors below1e-5. The firstContact and PULL-rejoin zero actions preserve all original100-step trajectories. Every archive entry is SHA-verified after packaging.

The746204 issue received exactly3 bounded CPU planner/IK queries from saved state,0new physics. All failed; this is not a global infeasibility proof and did not dominate the research.

Permanent assets: masked two-boundary SAC, measured contact/time accounting, exact-prefix audits, zero-preserving event intervention callback, common geometric controller, STAP kernel, all historical skill results and the new complete video workflow. The effect-coordinate candidate remains a **negative/untested-generalization hypothesis**, not a promoted method.

## Reproduction

Use the existing isolated environment and `tools/handoff_offline.sh`. Native physics is network/serial isolated, GPU0 only; ARES-R checkouts are read-only. No sealed data, site files, OS/driver modifications or live-device commands were used.

All raw traces, models, replay buffers, source snapshots, logs and failed attempts are in `RAW_*.tar.gz`; verify/extract with `tools/restore_post_contact_evidence.py`. The manifest records the original run root:
`/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003/post_contact_20261102`.
The restored analysis JSON uses original absolute provenance paths; the restore helper can create a separately marked portable metadata copy without changing gzipped physical traces or weights.

Principal commands (paths supplied explicitly; new run outputs must not already exist):

```bash
python research/skill_handoff_lab/research_program/scripts/audit_smdp_action_authority.py
python research/skill_handoff_lab/tools/audit_post_contact_window.py --help
python research/skill_handoff_lab/tools/program_post_contact.py --help
python research/skill_handoff_lab/tools/program_contact_direction.py --help
python research/skill_handoff_lab/tools/program_first_contact.py --help
python research/skill_handoff_lab/tools/evaluate_contact_methods.py --help
python research/skill_handoff_lab/tools/summarize_contact_research.py --root "$ARES_REPLAY_ROOT"
python research/skill_handoff_lab/tools/report_post_contact.py --root "$ARES_REPLAY_ROOT"
python research/skill_handoff_lab/tools/build_post_contact_review.py --root "$ARES_REPLAY_ROOT"
```

Training commands and exact seeds are in each run's PROTOCOL; physics commands must use the existing pinned interpreter via `handoff_offline.sh`, not the system Python. Baseline repository commit before this work: `c96d2831fff938281420d58a017622e1d2f081d3`. Existing ManiSkill/SB3/STAP sources and licenses are inherited, not upgraded. Per-run source snapshots take precedence over later convenience flags. The `near` wrapper/dependency copy was preserved after launch; it is not claimed to be a start-time source-hash attestation. New timing/learning/evaluation snapshots were copied before their loops.

## Research decision

Do not expand unchanged yaw-network training or present predictor accuracy as method progress. Native100 **does** contain executable post-contact decisions; there is no basis to claim otherwise to justify a task switch. The measured limitation is that the best predecessor parameter already covers the discovered rescues, while the two remaining failures need different executable recovery, not a reparameterization of the same lateral/yaw box. A next productive loop must establish residual benefit after this stronger fixed/ordinary/STAP baseline—especially the onset of tilted-tool contact versus true hooking, and the actual remaining recovery budget. If that residual disappears under a shared reasonable controller, retain the controller and use the already mature PlugCharger200 line to test genuine grasp-result/contact-condition dependence; do not revive saturated bias-only or weak-controller experiments. This is a concrete unresolved research requirement, not a claimed independent method result.


Browser interaction QA: the browser automation URL policy rejected `file://`; no alternate surface or local-server workaround was attempted. Frame encoding/counts, signal alignment, pose readback and static PNG inspection passed. Clicking/playing inside the browser is **not automatically verified**; open REVIEW.html locally for review. Cross-directory restore and full raw-derived results were actually verified; see analysis/RESTORE_QA.json.
