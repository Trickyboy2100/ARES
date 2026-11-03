"""Evidence-driven review. Requires final audited results, never fills missing scores."""
import argparse,json,html
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;z=json.loads((r/'FINAL_RESULTS.json').read_text());s=z['scores'];audit=z['physical_audit'];runs=audit['runs']
def score(name,seed):
 q=[x for x in s if x['method']==name and x['seed']==seed];assert len(q)==1;return f"{q[0]['success']}/{q[0]['n']}"
names=['dsac','projected','frontier_policy','frontier_uniform','frontier_ees','frontier_handoff','frontier_local','frontier_independent_kernel','frontier_joint','matched_joint_supervised','matched_joint_fqi','matched_joint_iql','matched_joint_cql','matched_joint_stap','mature_policy128']
table='| Final policy | Seed31301 | Seed31302 |\n|---|---:|---:|\n'+'\n'.join(f'| {n} | {score(n,31301)} | {score(n,31302)} |'for n in names)
fixed='; '.join(f"{x['method']}={x['success']}/{x['n']}"for x in s if x['seed']is None)
budget='| Run | Train episodes / decisions | Train physics steps | All planner calls | Planner seconds |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {n} | {d['training_episodes']} / {d['training_decisions']} | {d['training_steps']} | {d['planner_queries']} | {d['planning_s']:.1f} |"for n,d in runs.items()if n.startswith(('frontier_','mature_')))
text=f'''# Discrete physical Options and task-conditioned handoff acquisition

**Independent RL method advantage: UNVERIFIED.** This release preserves an action-correct learner, genuine online physical updates, and three successive acquisition experiments. Prediction improvements are reported separately from policy success. No sealed test, hardware or restricted field data was used.

## Main result: frozen final policies

{table}

Eight predeclared development scenes751220–227; two optimization seeds share these same eight scene clusters. They are not16 independent test scenes. All model hashes were frozen before the first expansion rollout. No best-checkpoint selection. Fixed alternatives: {fixed}. The best fixed action on this panel is a hindsight benchmark, not an online oracle.

The primary18 model/seed conditions each execute eight full native episodes. Their entire100-step traces must exactly match the corresponding finite physical branch. Remaining comparator policies use the measured48-branch panel: this is explicitly logged-policy evaluation, **not additional physical execution**. `expansion_eval/SCORES.json` labels each row. Matched-data supervised/FQI/IQL/CQL/STAP controls receive exactly the same acquired histories and update schedule. The128-episode ordinary learner uses a larger64-scene training distribution; it is a separate maturity diagnostic, not an equal-budget candidate comparison.

## What changed in the learning code

V0 used a continuous two-output actor and continuous critic while actual options are three grasp offsets(-10,0,+10mm), then two insertion targets(original/actual attachment). Sampled continuous actions fall off the executed support; projection changes Q and soft Bellman targets. Continuous differential entropy is not categorical option entropy. This is a semantic defect, not proof that it caused every failure.

V1 explicitly enumerates legal option IDs and masks action2 at the successor boundary. It retains actual SkillResult, option duration, native termination/failure and remaining budget. Gamma1 matches terminal native success under100 steps, with remaining budget observed; no hidden discount per option. DSAC, FQI, IQL, conservative-Q, conditional outcome scoring and functional STAP share54-dimensional privileged simulated state and the physical interface. The supervised label is the observed behavior continuation outcome, not an optimal successor oracle. STAP combines reach probability, a learned successor-state model and successor success in log space; it is a functional finite-option comparison, not the full original STAP training system.

The six analytic learner fixtures test code semantics only. Physical fixture replay reproduces old complete100-step traces exactly. `TRAINING_REPLAY_AUDIT.json` reconstructs real optimizer histories and verifies checkpoint parameters. `SHARED_CONTROL_AUDIT.json` checks identical warmup traces and MODEL_8 across acquisition methods. Discretization, normalization, LayerNorm,50:50 offline/online mixing and mature algorithms are retained infrastructure, not claimed contributions. The current tuning budget does not establish asymptotic SAC convergence.

## One acquisition hypothesis, three actual revisions

1. Initial handoff advantage/uncertainty sampling did not produce stable final gains. With only two successor actions its ranking can collapse to the non-incumbent action; a shared budget multiplier cannot resolve ranking.
2. Expected-update acquisition enumerates four possible paired outcomes, performs a virtual suffix-Q update and a predecessor Bellman update, and scores the expected change in the initial option decision. Real training remains ordinary DSAC. Fixed-prefix repeated sampling wasted budget on deterministic labels; the next revision gives all allocators the same no-repeat coverage constraint and permits new predecessor options.
3. Pair outcomes are correlated. On29 old training handoffs from10 scenes, leave-one-scene-out prediction gives observed discordance0.138, independent estimate0.358, joint estimate0.152; counterpart Brier0.313→0.182. This motivated a frozen five-neighbor joint model and an **exact same-marginal independence ablation**. These prediction numbers are not RL gains. The only changed acquisition component is outcome weighting; no neural auxiliary head, reward, observation or controller change.

Virtual acquisition uses one critic, two SGD steps(.01), greedy-Q decisions and approximate outcome probabilities; actual training uses twin critics, Adam and an actor. This surrogate mismatch is a concrete limitation, tested against actual actor changes and old-panel return in the optimizer replay audit. Expected error reduction, value of information, competence practice, uncertainty sampling and nominal fallback are known ideas. No novelty is asserted from combining their names.

## Physical evidence and fairness

Official PegInsertionSide-v1, pinned ManiSkill62ff3a5, native100 and native success. Shared official geometry with inherited GPU-OBB correction, speed2, all three refinements. No task weakening or scene reset at handoff. Every paired prefix is fully reexecuted and charged200 physical steps per pair. Early planner exits are completed with terminal holds to100; their failures remain in the dataset.

Compatible prior:60Peg episodes/118 option transitions, same controller and reward semantics. Tool, old stale-OBB runs, measured-insert variants and incompatible controllers are excluded. Training751000–015; matched acquisition751000–007; larger ordinary baseline751000–063. Monitor dev751200–205. Previous10 design and4 seen dev scenes are not described as fresh tests.

All models see the same privileged simulator state; this is not a sensor-only deployment claim. The fixed action(0mm,original target) already solves6/6 monitor scenes. No algorithm gain can be established solely from that saturated monitor. Actual strict-prefix examples show that refreshed attachment can rescue751000 and degrade751002. They establish conditional physical effects, not a method advantage.

## Real cost

Audited completed native episodes: {audit['total_complete_episodes']}; physics control steps: {audit['total_physics_steps']}. Primary critic optimizer iterations across real experiment models: {z['training_cost']['total_primary_critic_optimizer_iterations']} (audit reconstruction and analytic fixtures excluded). This includes training, fixtures and evaluation; the table separates training. Each full physical rollout is100 steps. Planner counts are actual plan_screw queries; internal IK iterations are not separately measured. Concurrent GPU wall times are not isolated speed benchmarks. Virtual acquisition compute, training time and optimizer steps are retained in ACQUIRE/HISTORY/checkpoints. Optimizer reconstruction is audit work and is excluded from training budgets.

{budget}

## Review and reproduction

Open `REVIEW.html`. Every scientific panel is also available separately in `figures/`. Videos are full101-frame measured-state visualizations of saved initial+100 physical states,20fps; no new physics is generated by rendering. RAW is unannotated pose playback, not a live RGB sensor recording. English versions and clickable frame-synchronized axial curves are linked individually.

Restore this package and the preceding `peg_value` package using `restore_post_contact_evidence.py`; verify archive/member SHA in `RAW_MANIFEST.json`. Source snapshots accompany each run; the final source is in the repository. Explicitly pass restored prior/root paths when reproducing. Offline runtime is invoked only through `handoff_offline.sh`, using the pinned local method-evolution Python environment. Never run production ARES-R entry points.

```bash
# Read-only evidence verification / deterministic optimizer reconstruction:
python research/skill_handoff_lab/tools/analyze_discrete_handoff.py --root RESTORED_RUN
python research/skill_handoff_lab/tools/audit_handoff_controls.py --root RESTORED_RUN
python research/skill_handoff_lab/tools/audit_option_training_replay.py --root RESTORED_RUN --prior RESTORED_PEG_VALUE
python research/skill_handoff_lab/tools/summarize_handoff_results.py --root RESTORED_RUN
```

To collect a new replication (not reuse stored labels), run the pinned Python through the isolation wrapper:

```bash
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/run_frontier_acquisition.py --prior RESTORED_PEG_VALUE --out NEW_RUN --mode joint --seed 31301
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/run_frontier_acquisition.py --prior RESTORED_PEG_VALUE --out NEW_CONTROL --mode independent_kernel --seed 31301
bash research/skill_handoff_lab/tools/handoff_offline.sh PINNED_PYTHON research/skill_handoff_lab/tools/run_discrete_handoff.py --mode online --prior RESTORED_PEG_VALUE --out NEW_ORDINARY --seed 31301 --allocation policy --episodes 128 --train-start 751000 --train-count 64 --eval-every 32
```

The raw archives retain unsuccessful episodes, failed audit attempts and corrected audit code, all checkpoints, acquired buffers, source snapshots and frozen protocol. See `PRIOR_ART.md` for pinned source review and limits: EES comparison is a finite-option competence/extrapolation analogue, not a full reproduction of symbolic EES; RLPD-inspired mixing is not full RLPD; RecoveryChaining source was not located. These limits prevent a publication-level superiority claim over those original systems.
'''
(r/'README.md').write_text(text)
rows=''.join('<tr><td>'+html.escape(n)+'</td><td>'+score(n,31301)+'</td><td>'+score(n,31302)+'</td></tr>'for n in names)
vid=''
for folder,title in [('video_handoff','Same-prefix rescue'),('video_learning','Same-prefix degradation')]:
 if(r/folder/'ANNOTATED.mp4').exists():vid+=f'<section><h2>{title}</h2><video controls preload="metadata" src="{folder}/ANNOTATED.mp4"></video><p><a href="{folder}/RAW.mp4">Full RAW measured-state replay</a> · <a href="{folder}/SYNC_VIEWER.html">Clickable synchronized physical curves</a> · <a href="{folder}/MANIFEST.json">Frame and source SHA</a></p></section>'
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ARES: physical Option learning</title><style>body{{font:17px/1.6 system-ui;background:#edf2f7;color:#14283d;margin:0}}main{{max-width:1180px;margin:auto;padding:40px 24px}}h1{{font-size:38px;line-height:1.15}}section{{background:white;padding:26px;margin:24px 0;border-radius:12px}}img,video{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ccd5df;text-align:left}}a{{color:#145c9d}}.status{{padding:10px;background:#fff1cd;font-weight:bold}}details pre{{white-space:pre-wrap}}</style><main><h1>Physical Options → real handoffs → learned decisions</h1><p>ARES single-paper research · native PegInsertionSide100 · two training seeds · no physical robot</p><p class="status">Independent method advantage: UNVERIFIED. Prediction accuracy and correct action semantics are not algorithm gains.</p><section><h2>Frozen final results</h2><p>Eight predeclared fresh development scenes. All policies frozen before evaluation; no sealed tests. Some comparator rows use an explicitly labeled real finite-branch panel rather than new rollouts.</p><table><tr><th>Policy</th><th>Seed31301</th><th>Seed31302</th></tr>{rows}</table><p>{html.escape(fixed)}</p><img src="figures/02_frozen_expansion.png"><a href="FINAL_RESULTS.json">Complete results, costs and rescue/regression pairs</a><img src="figures/06_decision_attribution.png"><a href="DECISION_ATTRIBUTION.json">Failure and critic-greedy diagnostics</a></section><section><h2>Actual online learning</h2><img src="figures/01_matched_learning.png"><img src="figures/04_ordinary_extended.png"><p>All matched24-episode methods share identical warmup physics and checkpoint. Every full prefix is charged. Larger128-episode ordinary SAC is reported separately.</p></section><section><h2>A concrete acquisition mechanism, not another auxiliary head</h2><p>Enumerate possible paired suffix outcomes → virtual suffix/predecessor update → score initial-action decision change → acquire full real trajectories → ordinary discrete SAC update. Revisions test coverage and pair-outcome dependence while keeping physical control unchanged.</p><img src="figures/03_outcome_dependence.png"><p>Same marginal probabilities; covariance is the ablation. Better outcome prediction alone does not establish improved deployment.</p></section>{vid}<section><h2>Audit and reproducibility</h2><img src="figures/05_action_support.png"><p><a href="README.md">Detailed scientific conclusions and limitations</a> · <a href="PRIOR_ART.md">Pinned prior-art comparison</a> · <a href="RAW_MANIFEST.json">All archives and SHA</a> · <a href="SEMANTICS_AUDIT.json">V0 action-support audit</a></p><p>Official geometry, GPU-OBB correction, native success,100-step horizon, full grasp execution, actual SkillResult, shared observations/actions. All videos are labeled measured-state replay, not new physics.</p></section></main></html>'''
(r/'REVIEW.html').write_text(page)
