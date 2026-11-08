"""Actual native checkpoint curves; exploratory training is labelled separately."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
colors={'sac':'#3267a8','retention':'#cd7738','ranking':'#9a5c93','witness_soft':'#66883d','witness_project':'#999999','witness_qp':'#008c8c','joint_sac':'#69717a'}
names={'sac':'SAC + success BC','retention':'Reference KL + paired ranking','ranking':'Supervised + paired ranking','witness_soft':'Soft witness support','witness_project':'Stalled projection pilot','witness_qp':'QP witness update','joint_sac':'Joint predecessor + successor SAC'}
all_runs={f.parent.name:json.loads(f.read_text())for f in sorted(a.root.glob('*/HISTORY.json'))}
runs={n:h for n,h in all_runs.items()if n.rsplit('_',1)[0]in ['sac','witness_soft','witness_qp']}
def style(name):
 arm=name.rsplit('_',1)[0];return names.get(arm,arm),colors.get(arm,'#444'),'--'if name.endswith('32302')else '-'
def checkpoint(ax):
 seen=set()
 for name,h in runs.items():
  ev=[x for x in h if x['kind']=='eval'];ks=sorted(set(x['checkpoint']for x in ev));xs=[];ys=[]
  for k in ks:
   rows=[x for x in ev if x['checkpoint']==k]
   if len(rows)==16:xs.append(k);ys.append(sum(x['row']['success']for x in rows))
  label,color,ls=style(name);ax.plot(xs,ys,'o',linestyle=ls,color=color,label=label if label not in seen else None);seen.add(label)
 ax.axhline(10,color='#777',lw=1,ls=':',label='Warm initialization 10/16');ax.set(xlabel='Own-policy training episodes',ylabel='Native full-task successes / 16',title='Seen development | fixed checkpoint evaluation',xticks=[0,64,128],ylim=(0,16));ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
def training(ax):
 seen=set()
 for name,h in runs.items():
  y=[int(x['row']['success'])for x in h if x['kind']=='train'];w=16
  if len(y)<w:continue
  label,color,ls=style(name);ax.plot(range(w,len(y)+1),np.convolve(y,np.ones(w)/w,'valid'),ls,color=color,label=label if label not in seen else None);seen.add(label)
 ax.set(xlabel='Own-policy training episodes',ylabel='16-episode rolling success fraction',title='Actual exploratory collection | not greedy deployment',ylim=(0,1));ax.grid(alpha=.2)
fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');checkpoint(axs[0]);training(axs[1]);fig.suptitle('Native Peg100 | seed32301 solid, seed32302 dashed | shared executable Options')
for ext in ['png','pdf']:fig.savefig(a.out/f'LEARNING_CURVES.{ext}',dpi=200)
plt.close(fig)
for name,draw in [('CHECKPOINT_CURVE',checkpoint),('ONLINE_COLLECTION',training)]:
 fig,ax=plt.subplots(figsize=(6.3,4.5),layout='constrained');draw(ax)
 for ext in ['png','pdf']:fig.savefig(a.out/f'{name}.{ext}',dpi=200)
 plt.close(fig)
(a.out/'LEARNING_SOURCES.json').write_text(json.dumps(dict(histories=[str(a.root/n/'HISTORY.json')for n in runs],scope='Actual completed checkpoints only; seed32302 dashed; shared scenes are not independent repetitions'),indent=2))

runs={n:h for n,h in all_runs.items()if n.rsplit('_',1)[0]in ['sac','retention','ranking','joint_sac']}
fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained');checkpoint(ax)
for ext in ['png','pdf']:fig.savefig(a.out/f'FIRST_ITERATION.{ext}',dpi=200)
