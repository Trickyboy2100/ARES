"""Final fixed-checkpoint statistics and figures; no policy selection or fitting."""
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'figures';out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
scores=read(r/'expansion/SCORES.json');models=read(r/'expansion/FROZEN_MODELS.json')['models'];assert len(scores)==len(models)+6
pred=read(r/'expansion/BRANCH_PREDICTIONS.json');assert len(pred)==len(models)*8
for m in models:assert hashlib.sha256(Path(m['path']).read_bytes()).hexdigest()==m['sha256']
by=collections.defaultdict(dict)
for z in pred:by[(z['method'],z['seed'])][z['scene']]=z
comparisons=[]
for seed in [31301,31302]:
 for candidate in ['frontier_joint','frontier_handoff','mature_policy128']:
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
selected=['fixed10','dsac','projected','frontier_policy','frontier_ees','frontier_handoff','frontier_local','frontier_independent_kernel','frontier_joint','matched_joint_supervised','matched_joint_iql','mature_policy128']
fig,ax=plt.subplots(figsize=(11,6))
for i,name in enumerate(selected):
 zz=[z for z in scores if z['method']==name]
 for k,z in enumerate(zz):ax.scatter(z['success']/8,i+(k-.5)*.16 if len(zz)>1 else i,s=65,c=['#2980b9','#d35400'][k],marker='o'if z['seed']else'D')
ax.set(yticks=range(len(selected)),yticklabels=selected,xlim=(-.03,1.03),xlabel='Native task success / 8 predeclared fresh development scenes',title='Frozen final policies | dots = training seeds 31301 / 31302');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(out/'02_frozen_expansion.png',dpi=180);fig.savefig(out/'02_frozen_expansion.svg');plt.close(fig)
j=read(r/'JOINT_OUTCOME_AUDIT.json')['summary'];fig,ax=plt.subplots(figsize=(7,4));ax.bar(['Observed','Independent','Joint'],[j['actual_discordance_rate'],j['independent']['discordance_probability'],j['joint']['discordance_probability']],color=['#34495e','#e67e22','#2980b9']);ax.set(ylabel='P(two suffix outcomes disagree)',ylim=(0,.5),title='29 handoffs / 10 old training scenes\nLeave-one-scene-out; prediction diagnostic, NOT RL gain');fig.tight_layout();fig.savefig(out/'03_outcome_dependence.png',dpi=180);fig.savefig(out/'03_outcome_dependence.svg');plt.close(fig)
# Scope labels preserve logged finite-branch evaluation vs directly executed models.
result=dict(claim='UNVERIFIED',scores=scores,pairwise=comparisons,curves=curves,scene_unit='Eight shared scene clusters; two optimization seeds are not sixteen independent scenes',monitor_scope='six development scenes observed during training; fixed10 succeeds6/6',expansion_scope='eight predeclared development scenes, no training, never sealed; all final policies frozen before physics',physical_audit=read(r/'ANALYSIS.json'),training_replay_audit=read(r/'TRAINING_REPLAY_AUDIT.json'))
(r/'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2));print('Final model hashes verified:',len(models),'scores',len(scores))
