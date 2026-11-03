"""Final fixed-checkpoint statistics and figures; no policy selection or fitting."""
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'figures';out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
scores=read(r/'expansion_eval/SCORES.json');models=read(r/'expansion_eval/FROZEN_MODELS.json')['models'];assert len(scores)==len(models)+6
pred=read(r/'expansion_eval/BRANCH_PREDICTIONS.json');assert len(pred)==len(models)*8
extra=read(r/'anchor_eval/SCORES.json');assert len(extra)==14;scores+=extra;models+=read(r/'anchor_eval/FROZEN_MODELS.json')['models'];pred+=read(r/'anchor_eval/BRANCH_PREDICTIONS.json')
for m in models:assert hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']
by=collections.defaultdict(dict)
for z in pred:by[(z['method'],z['seed'])][z['scene']]=z
comparisons=[]
for seed in [31301,31302]:
 for candidate in ['frontier_current_joint','frontier_joint','frontier_handoff','mature_policy128']:
  for control in ['frontier_policy','frontier_ees','frontier_independent_kernel','frontier_local','dsac','projected','matched_joint_supervised','matched_joint_iql']:
   aa=by[candidate,seed];bb=by[control,seed];comparisons.append(dict(candidate=candidate,control=control,seed=seed,n=len(aa),rescues=sum(aa[s]['success']and not bb[s]['success']for s in aa),regressions=sum(not aa[s]['success']and bb[s]['success']for s in aa),first_action_differences=sum(aa[s]['actions'][0]!=bb[s]['actions'][0]for s in aa),scope='same natural initial scene; not necessarily same executed prefix'))
curves={x.parent.name:read(x)for x in sorted(r.glob('*/CURVE.json'))}
fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
colors=dict(policy='#34495e',ees='#27ae60',handoff='#a569bd',joint='#c0392b',independent_kernel='#2980b9',uniform='#888888')
for ax,seed in zip(axs,[31301,31302]):
 for mode,color in colors.items():
  cs=curves[f'frontier_{mode}_{seed}'];ax.plot([x['episodes']*100 for x in cs],[x['success']/6 for x in cs],'-o',label=mode,color=color,alpha=.85)
 ax.set(title=f'Training seed {seed}',xlabel='New training physics steps (prefixes charged)',ylim=(0,1.05),xticks=[0,800,1600,2400]);ax.grid(alpha=.2)
axs[0].set_ylabel('Native success / 6 monitor scenes');axs[1].legend(fontsize=8);fig.suptitle('Matched 24-episode acquisition | final checkpoint, no best-checkpoint selection');fig.tight_layout();fig.savefig(out/'01_matched_learning.png',dpi=180);fig.savefig(out/'01_matched_learning.svg');plt.close(fig)
selected=['fixed00','fixed10','fixed21','dsac','projected','frontier_policy','frontier_ees','frontier_handoff','frontier_local','frontier_independent_kernel','frontier_joint','matched_joint_supervised','matched_joint_iql','frontier_current_joint','frontier_current_independent_kernel','matched_current_joint_supervised','matched_current_joint_iql','mature_policy128']
fig,ax=plt.subplots(figsize=(12,8))
for i,name in enumerate(selected):
 zz=[z for z in scores if z['method']==name]
 for k,z in enumerate(zz):ax.scatter(z['success']/8,i+(k-.5)*.16 if len(zz)>1 else i,s=65,c=['#2980b9','#d35400'][k]if z['seed']else'#444444',marker='o'if z['seed']else'D')
labels={'fixed00':'Fixed -10mm / original (training tie)','fixed10':'Fixed 0mm / original','fixed21':'Fixed +10mm / refreshed (hindsight)','dsac':'Discrete SAC (offline)','projected':'Continuous projected SAC','frontier_policy':'Ordinary SAC, 24 episodes','frontier_ees':'EES-style practice, 24 episodes','frontier_handoff':'Expected-update acquisition','frontier_local':'Without predecessor propagation','frontier_independent_kernel':'Paired kernel: independent','frontier_joint':'Paired kernel: joint','matched_joint_supervised':'Same-history supervised','matched_joint_iql':'Same-history IQL','frontier_current_joint':'Current anchors: joint','frontier_current_independent_kernel':'Current anchors: independent','matched_current_joint_supervised':'Current-history supervised','matched_current_joint_iql':'Current-history IQL','mature_policy128':'Ordinary SAC, 128 episodes'}
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color=c,marker='o',linestyle='',label=str(seed))for seed,c in [(31301,'#2980b9'),(31302,'#d35400')]],title='Training seed',loc='lower right')
ax.set(yticks=range(len(selected)),yticklabels=[labels[x]for x in selected],xlim=(-.03,1.03),xlabel='Success rate (8 shared development scenes)',xticks=[0,.25,.5,.75,1],title='Frozen final policies | dots = training seeds 31301 / 31302');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'02_frozen_expansion.png',dpi=180);fig.savefig(out/'02_frozen_expansion.svg');plt.close(fig)
j=read(r/'JOINT_OUTCOME_AUDIT.json')['summary'];fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Observed','Independent','Joint'],[j['actual_discordance_rate'],j['independent']['discordance_probability'],j['joint']['discordance_probability']],color=['#34495e','#e67e22','#2980b9']);ax.set(ylabel='P(two suffix outcomes disagree)',ylim=(0,.5),title='29 handoffs / 10 old training scenes\nLeave-one-scene-out; prediction diagnostic, NOT RL gain');fig.tight_layout();fig.savefig(out/'03_outcome_dependence.png',dpi=180);fig.savefig(out/'03_outcome_dependence.svg');plt.close(fig)
# Scope labels preserve logged finite-branch evaluation vs directly executed models.
main_iterations=sum(x.get('optimizer_iterations',0)for x in read(r/'ANALYSIS.json')['runs'].values())
offline=read(r/'offline/TRAINING.json');offmax={}
for x in offline:offmax[x['method'],x['seed']]=max(offmax.get((x['method'],x['seed']),0),x['updates'])
matched=read(r/'expansion_eval/MATCHED_TRAINING.json')+read(r/'anchor_eval/MATCHED_TRAINING.json');projected=read(r/'projected_control/TRAINING.json');projmax={}
for x in projected:projmax[x['seed']]=max(projmax.get(x['seed'],0),x['updates'])
cost=dict(main_online_iterations=main_iterations,offline_comparator_iterations=sum(offmax.values()),matched_data_comparator_iterations=sum(x['updates']for x in matched),projected_control_iterations=sum(projmax.values()),audit_optimizer_reconstruction_excluded=True,fixture_updates_excluded=True)
cost['total_primary_critic_optimizer_iterations']=sum(v for k,v in cost.items()if k.endswith('_iterations'))
result=dict(claim='UNVERIFIED',training_cost=cost,scores=scores,pairwise=comparisons,curves=curves,scene_unit='Eight shared scene clusters; two optimization seeds are not sixteen independent scenes',monitor_scope='six development scenes observed during training; fixed10 succeeds6/6',expansion_scope='eight predeclared development scenes, no training, never sealed; all final policies frozen before physics',physical_audit=read(r/'ANALYSIS.json'),training_replay_audit=read(r/'TRAINING_REPLAY_AUDIT.json'))
(r/'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2));print('Final model hashes verified:',len(models),'scores',len(scores))
fig,ax=plt.subplots(figsize=(8,4))
for seed,color in [(31301,'#2980b9'),(31302,'#d35400')]:
 cs=curves[f'mature_policy_{seed}'];ax.plot([x['episodes']*100 for x in cs],[x['success']/6 for x in cs],'-o',color=color,label=f'seed {seed}')
ax.axhline(1,color='#34495e',ls='--',label='fixed 0mm/original');ax.set(xlabel='New training physics steps',ylabel='Success / 6 monitor scenes',ylim=(0,1.05),title='Ordinary discrete SAC | 128 episodes / 64 train scenes');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'04_ordinary_extended.png',dpi=180);fig.savefig(out/'04_ordinary_extended.svg');plt.close(fig)
c=read(r/'ACTION_SUPPORT_CURVE.json');fig,ax=plt.subplots(figsize=(8,4));ax.plot(c['continuous_requested'],c['critic'],label='Q at requested continuous action',color='#c0392b');ax.plot(c['continuous_requested'],c['critic_at_executed_option'],label='Q at executed finite option',color='#2980b9');ax.axvline(c['actor_request'][0],ls=':',color='#777777',label='Actor request');ax.set(xlabel='Continuous requested action',ylabel='Saved critic output (not success probability)',title='V0 action-support mismatch | actual saved handoff750103');ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'05_action_support.png',dpi=180);fig.savefig(out/'05_action_support.svg');plt.close(fig)
a=read(r/'DECISION_ATTRIBUTION.json')['models'];pick=['dsac','frontier_policy','frontier_ees','frontier_joint','matched_joint_supervised','mature_policy128'];fig,axs=plt.subplots(1,2,figsize=(12,5),sharey=True);cats=['SUCCESS','NO_MEASURED_LEGAL_SOLUTION','PREDECESSOR_OPTION_SELECTION','SUCCESSOR_OPTION_SELECTION'];palette=['#2a9d8f','#aaaaaa','#e9c46a','#e76f51']
for ax,seed in zip(axs,[31301,31302]):
 bottom=np.zeros(len(pick))
 for cat,col in zip(cats,palette):
  vals=np.array([next(x for x in a if x['method']==n and x['seed']==seed)['counts'][cat]for n in pick]);ax.barh(range(len(pick)),vals,left=bottom,color=col,label=cat);bottom+=vals
 ax.set(title=f'Seed {seed}',yticks=range(len(pick)),yticklabels=pick,xlim=(0,8),xlabel='Eight shared development scenes');ax.invert_yaxis()
axs[1].legend(fontsize=7,loc='lower right');fig.suptitle('Failure attribution from all six actually executed physical branches');fig.tight_layout();fig.savefig(out/'06_decision_attribution.png',dpi=180);fig.savefig(out/'06_decision_attribution.svg');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,seed in zip(axs,[31301,31302]):
 for mode,color in [('joint','#777777'),('current_joint','#c0392b'),('current_independent_kernel','#2980b9'),('ees','#27ae60')]:
  cs=curves[f'frontier_{mode}_{seed}'];ax.plot([x['episodes']*100 for x in cs],[x['success']/6 for x in cs],'-o',label=mode,color=color)
 ax.set(title=f'Seed {seed}',xlabel='Training physics steps',ylim=(0,1.05));ax.grid(alpha=.2)
axs[0].set_ylabel('Native success / 6 monitor scenes');axs[1].legend(fontsize=8);fig.suptitle('Single-factor revision: acquisition anchors from current training distribution');fig.tight_layout();fig.savefig(out/'07_anchor_distribution.png',dpi=180);fig.savefig(out/'07_anchor_distribution.svg');plt.close(fig)
ex=read(r/'POLICY_EXTRACTION_AUDIT.json')['models'];direct=read(r/'extraction_eval/SCORES.json');assert len(direct)==6;comparison=[]
for d in direct:
 old=next(x for x in ex if x['run']==d['run'])['policies']['actor']['rows']
 rr=[x for x in read(r/'extraction_eval/physics/RESULTS.json')if x['path']in d['sources']];original={x['scene']:x for x in old};comparison.append(dict(run=d['run'],extraction=d['extraction'],actor_success=sum(x['success']for x in old),direct_success=d['success'],rescues=sum(x['success']and not original[x['seed']]['success']for x in rr),regressions=sum(not x['success']and original[x['seed']]['success']for x in rr),scope='same initial conditions, posthoc direct physical confirmation; no training'))
(r/'EXTRACTION_COMPARISON.json').write_text(json.dumps(comparison,indent=2));fig,ax=plt.subplots(figsize=(11,5));xx=np.arange(len(ex))
for j,(mode,col)in enumerate([('actor','#34495e'),('min','#e9c46a'),('mean','#2a9d8f')]):ax.bar(xx+(j-1)*.24,[x['policies'][mode]['success']for x in ex],.24,label=mode,color=col)
ax.set(xticks=xx,xticklabels=[x['run'].replace('frontier_','').replace('_313','\nseed313')for x in ex],ylabel='Native success / 8 scenes',ylim=(0,8),title='Frozen weights: policy extraction, not a new learning algorithm\nLogged complete real branch panel; six conditions physically reconfirmed');ax.legend();fig.tight_layout();fig.savefig(out/'08_policy_extraction.png',dpi=180);fig.savefig(out/'08_policy_extraction.svg');plt.close(fig)
# Normalize exporter-only whitespace; scientific values and SVG geometry unchanged.
for svg in out.glob('*.svg'):
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
