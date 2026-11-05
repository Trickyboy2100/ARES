"""Data-derived independent scientific figures from verified summary."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);d=json.loads((a.root/'SUMMARY.json').read_text());plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 fig.tight_layout();fig.savefig(a.out/f'{name}.png',dpi=220);fig.savefig(a.out/f'{name}.pdf');plt.close(fig)
for family in ['joint','identity','paired']:
 fig,ax=plt.subplots(figsize=(7.5,4.5))
 for idx,(name,curve) in enumerate(d['learning_curves'].items()):
  actual='identity' if name.startswith('identity') else ('paired' if name.startswith('paired') else 'joint')
  if actual!=family:continue
  c={int(k):v for k,v in curve.items()if v['n']==16}
  if c:ax.plot(sorted(c),[c[k]['success']for k in sorted(c)],marker=['o','s','^','D'][idx%4],linestyle=['-','--',':','-.'][idx%4],fillstyle='none',markersize=7+idx%3,label=name.replace('_',' '))
 ax.set(xlabel='Own-policy training episodes',ylabel='Complete native successes /16',ylim=(0,16),title='Joint learning: '+('764300–315'if family=='joint'else'764600–615'))
 ax.legend(fontsize=8);save(fig,'learning_curves_'+family)
for mode in ['rule','warm']:
 if mode not in d['design']:continue
 z=d['design'][mode];seeds=list(z['rows']);mat=np.array([z['rows'][s]for s in seeds],int).T
 fig,ax=plt.subplots(figsize=(10,2.7));ax.imshow(mat,cmap='Blues',vmin=0,vmax=1,aspect='auto');ax.set_yticks(range(3),['grasp 0','grasp 1','grasp 2']);ax.set_xticks(range(16),[s[-3:]for s in seeds],rotation=45);ax.set(xlabel='Natural training-design scene (764xxx)',title=f'Actual full-task outcomes: frozen {mode} successor')
 for y in range(3):
  for x in range(16):ax.text(x,y,str(mat[y,x]),ha='center',va='center',color='white'if mat[y,x]else'black')
 save(fig,f'grasp_{mode}_matrix')
names=['frozen_rule','frozen_INITIAL','frozen_CHECKPOINT_64','frozen_rehearsal'];names=[n for n in names if n in d['groups']]
fig,ax=plt.subplots(figsize=(7,4));values=[d['groups'][n]['success']for n in names];ax.bar(range(len(names)),values,color=['#555','#26889c','#e19b3b','#6b73b5'][:len(names)]);ax.set_xticks(range(len(names)),['Strong rule','Paired initializer','Prior final RL','Supervised replay'][:len(names)]);ax.set(ylabel='Complete native successes /32',ylim=(0,32),title='Frozen previous policies on a new natural cohort')
for i,v in enumerate(values):ax.text(i,v+.5,str(v),ha='center')
save(fig,'frozen_confirmation')
