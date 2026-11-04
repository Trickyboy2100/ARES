"""Compact review generated from actual artifacts, never placeholder scores."""
import argparse,json,html
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;s=json.loads((r/'RESULTS_SUMMARY.json').read_text())
def count(path):
 if not path.exists():return 'INCOMPLETE'
 x=json.loads(path.read_text());return f"{sum(z['success']for z in x)}/{len(x)}"
rows=[]
for mode in ['ordinary','suffix','budget']:
 v=s['methods'][mode];final=v['local_training_native_evaluations'].get('FINAL');dep=r/f'deployment_labels_{mode}'
 if mode=='ordinary'and(r/'deployment_labels_ordinary_retry').exists():dep=r/'deployment_labels_ordinary_retry'
 rows.append([mode,f"{v['full_native_training_evaluations']['96']['success']}/16",f"{final['success']}/16"if final else'INCOMPLETE',count(dep/'EVAL_FINAL.json')])
mdtable='| Method | After 96 native training episodes | After ~10k local training steps | Deployment-label ablation (~2840 steps, independent branch) |\n|---|---:|---:|---:|\n'+'\n'.join('| '+' | '.join(row)+' |'for row in rows)
legacy=json.loads((r/'retained_baselines/physics/RESULTS.json').read_text());legacytext=', '.join(f"{tag}: {sum(z['success']for z in legacy if z['tag']==tag)}/16"for tag in ['fixed11','retained_sac'])
text=f'''# Closed-loop successor skill and budget-conditioned handoff — 2026-11-04

**UNVERIFIED independent algorithm gain.** This is actual public ManiSkill GPU physics, not robot deployment. Start from `616099e`; retain all previous controllers, Options and negative results.

## What was actually built and trained

The predecessor still chooses three legal grasp offsets. After ALIGN, the successor can update every four actual control steps: continue the active plan, measured attachment target, goal-frame y/z ±3mm, withdraw6mm, or hold. Decisions are phase-bounded: hold ends a refinement after one actual step, while insertion hold lasts up to four; all durations are measured. Repeating an active target continues its queue rather than restarting from rest. Original refinement paths, joint limits, native success and100-step deadline remain. Zero-feedback paths were physically checked against original trajectories (exact equality on four pilot cases). The arm has a real simulated finger grasp: attachment can slip; no fixed-object joint is used.

Common finite SAC uses64 observable features, masked3/8 actions, actual Option duration, native terminal success plus telescoping potential shaping, shared CQL/BC warmstart and mean-Q. Budget is observed by every method. The candidate changes predecessor Bellman backups using an actual-success completion-time CDF; the scalar suffix comparator uses ordinary downstream success value. This is a testable hypothesis, **not a novelty claim**. Read [prior comparison](PRIOR_ART.md).

## Full native-start results

All cells below are16 seen development initial states753200–753215, continuous native100 execution with no handoff reset, one training seed31501. They are exploratory and not sealed or publication-level tests. Retained controls: **{legacytext}**.

{mdtable}

The ~2840-step branch starts again from the96-episode checkpoint, not from the10k-local-step final model. It reserves two of eight suffix rollouts per captured prefix for deterministic deployment-policy labels; all methods share this sampling. It tests exploration/deployment label mismatch and does not establish an original algorithm. Optimization counts and actual physical costs are separately retained.

## What the physical intervention actually established

Six hindsight-selected TRAINING recovery candidates were reexecuted from native starts with identical real prefixes (48–67 steps; maximum recorded action/q/qvel/peg/TCP difference0). Zero intervention6/6, strong rule6/6, mined correction schedule4/6. Thus there are **zero demonstrated native recoveries and two regressions** in this selected set. Restored-local success is not a full-task recovery. Closed-loop software and trained models are reusable assets; a new beneficial manipulation capability has not yet been established.

Seed755011 shows the issue clearly: the correction run remains `grasped=True` at all52 post-handoff measurements yet accumulates substantial TCP-relative peg drift and fails; nominal continuation inserts successfully. [Actual mechanism curves](figures/MECHANISM.png), [complete RAW/English video and frame-linked curves](video_regression/SYNC_VIEWER.html). Video is101-frame measured-state replay, with no new physical execution or fabricated frames.

A second targeted ablation keeps each final learner's exact predecessor and replaces only its suffix by zero intervention. On five seen regression cases, ordinary SAC changes1/5→5/5 and the budget candidate2/5→5/5, with exact measured shared prefixes. This attributes those selected failures to suffix intervention, not to grasp choice; it is not an overall success estimate. [Ablation data](ZERO_SUFFIX_ABLATION.json).

## Learning interpretation and costs

Initial nonconservative pretraining selected rare unsupported corrections and degraded. Shared conservative warmstart restored9/16; this is mature infrastructure, not a method contribution. A frozen-policy change to persistent execution alone gave5/16 before and after (zero rescue/regression). Full96-episode training then gave the results above; the scalar comparator's earlier extra success was a changed grasp with an unchanged nominal suffix. Local training eventually activates many corrections but also causes regressions, so more updates are not automatically improvement.

Completion heads were trained on exploratory suffix behavior, while deployment is deterministic. Initial budget-head mean prediction0.215 versus actual8/16 and Brier0.273 motivate the deployment-label experiment; this observation alone does not prove the causal explanation. The deployment-label branch ends at9/16 ordinary,7/16 scalar and7/16 budget; budget hazard Brier is0.063 on its own new-policy rollouts, versus0.273 earlier. Those are different deployed-state distributions, so this is not a controlled calibration-only improvement claim. In particular, a lower prediction error still did not produce policy superiority. The hazard represents recent-policy outcomes, not an oracle for future policies or unexecuted deadlines. Censored hazards and downstream values already have substantial prior art.

[Exact recomputed metrics and scope-separated cost](RESULTS_SUMMARY.json), [source/model/trajectory archive manifest](RAW_MANIFEST.json). Local training executes and charges each prefix once and every suffix step; restoration and one charged contact-refresh step are explicitly logged. Restored contacts are not bit-exact counterfactuals. Only native-start evaluations count toward the table. The suffix curriculum was resumed from a saved checkpoint after a CUDA700 failure; optimizer/model state was restored but exploration RNG restarted, disclosed in the raw recovery record. A separate SIGTERM interrupted one label-ablation attempt. Archived costs are exact; unarchived partial failed attempts make total attempted cost a lower bound. No failure is hidden as a normal successful run.

## Retain / do not claim

Retain the feedback runtime, exact-zero regression, persistent parameter execution, finite SMDP learner, checkpoints, reachable-handoff training, physical-cost audit and all negative evidence. Do not return to fixed-branch scoring as the research objective. Neither new closed-loop success capability nor independent budget-method superiority is currently established. Current observations contain pose, attachment translation and robot velocity, but not full object twist/contact forces; attribution to observation aliasing needs a controlled experiment, not a claim. Preserve the strongest nominal and learned checkpoints while developing stable feedback learning.

## Reproduction and boundaries

[REPRODUCE.sh](REPRODUCE.sh) uses the pinned environment and archived compatible warm replay; sequential GPU execution is recommended. The warm set contains61/64 previous trajectories whose executed branches are exactly compatible with persistent semantics, with two actual reexecution checks. Three incompatible trajectories are excluded. Original legacy observations and assets remain untouched. Only public ManiSkill simulation assets are packaged. ARES-R stayed read-only; no hardware/network execution or sealed tests. Source snapshots, raw trajectories, models, optimizer states, budgets and failures are in the archives; [individual scientific panels](figures/) are directly reusable.
'''
(r/'README.md').write_text(text)
table='<table><tr><th>Method</th><th>96 native episodes</th><th>~10k local steps</th><th>Deployment-label ablation</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>'for x in row)+'</tr>'for row in rows)+'</table>'
page='''<!doctype html><meta charset="utf-8"><title>ARES closed-loop skill research</title><style>body{max-width:1150px;margin:40px auto;padding:0 24px;background:#f4f6f8;color:#192d3d;font:18px/1.6 system-ui}h1{line-height:1.2}a{color:#125c9b}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;padding:14px;border-bottom:1px solid #ccd6df}img,video{width:100%;background:white;border-radius:8px}.flag{padding:18px;background:#fff0d8;border-left:5px solid #c8791a}.links{display:flex;gap:25px;flex-wrap:wrap}</style><h1>Closed-loop successor skills<br>and budget-conditioned handoff</h1><p>ARES SCI research · public ManiSkill GPU physics · native100 · 2026-11-04</p><div class="flag"><b>Independent method gain: UNVERIFIED.</b> The repeated-feedback skill and training pipeline exist. Beneficial new operation capability has not yet been established.</div><h2>Actual full-task results</h2><p>16 seen development scenes; one seed. Full physical prefix, no handoff reset. '''+html.escape(legacytext)+'.</p>'+table+'''<p>The deployment-label ablation starts from the96-episode parent, not the10k-step branch. Local training outcomes are excluded from full-task scores.</p><img src="figures/LEARNING.png" alt="Actual learning curves"><h2>A local success is not necessarily a recovery</h2><p>Six training-selected candidates: nominal6/6; rule6/6; mined corrections4/6. Identical native prefixes were physically verified. No rescue; two regressions.</p><img src="figures/MECHANISM.png" alt="Actual paired physical mechanism"><h2>Complete paired trajectory</h2><video controls preload="metadata" src="video_regression/ANNOTATED.mp4"></video><p class="links"><a href="video_regression/SYNC_VIEWER.html">Frame-linked interactive curves</a><a href="video_regression/RAW.mp4">Complete RAW</a><a href="video_regression/MANIFEST.json">Video source SHA</a></p><h2>Read and reproduce</h2><p>Keep the closed-loop skill, persistent planning, conservative warmstart and measured SkillResult. The hazard candidate and scalar-value comparator are disclosed hypotheses/adaptations, not renamed original methods. Restored local states are training-only and not exact physical counterfactuals. CUDA/termination failures and all negative models remain archived.</p><p class="links"><a href="README.md">Research conclusions and limitations</a><a href="RESULTS_SUMMARY.json">Actual metrics and costs</a><a href="PRIOR_ART.md">Fabrica / SeqDex / SPIRE and competing prior</a><a href="REPRODUCE.sh">Reproduce</a><a href="RAW_MANIFEST.json">All archive members and SHA</a></p>'''
(r/'REVIEW.html').write_text(page)
