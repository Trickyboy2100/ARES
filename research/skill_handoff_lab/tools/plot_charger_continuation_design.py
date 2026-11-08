"""Actual second-task controls and seen crossover repetitions; no learned transfer."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);d=json.loads((a.root/'SECOND_TASK_RESULTS.json').read_text())
def panel(ax):
 x=np.arange(3);angles=[15,30,45]
 for j,(k,label,c)in enumerate([('nominal_success','Nominal pace','#3267a8'),('feedback_success','Feedback pace | matched prefix','#008c8c')]):
  y=[d['paired_groups'][str(z)][k]for z in angles];ax.bar(x+(j-.5)*.32,y,.3,color=c,label=label)
  for xx,yy in zip(x+(j-.5)*.32,y):ax.text(xx,yy+.15,str(yy),ha='center',fontsize=10)
 ax.set(xticks=x,xticklabels=['15°','30°','45°'],ylim=(0,14),ylabel='Native full-task successes / 12',xlabel='Actually executed grasp pitch',title='Native Charger200 | 12 new natural scenes');ax.legend(fontsize=8,loc='lower left');ax.grid(axis='y',alpha=.2)
def cross(ax):
 seeds=sorted({r['seed']for r in d['crossover_repeats']});mat=np.array([[sum(r['success']for r in d['crossover_repeats']if r['seed']==seed and r['angle']==angle)for angle in [15,30]]for seed in seeds]);ax.imshow(mat,vmin=0,vmax=2,cmap='Blues',aspect='auto')
 for i in range(len(seeds)):
  for j in range(2):ax.text(j,i,f'{mat[i,j]}/2',ha='center',va='center',color='white'if mat[i,j]else'#173044',fontsize=20)
 ax.set(xticks=[0,1],xticklabels=['15°','30°'],yticks=range(len(seeds)),yticklabels=seeds,xlabel='Actually executed grasp pitch',title='Seen mechanism followup | two full repeats per cell');ax.set_ylabel('Natural scene ID')
fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');panel(axs[0]);cross(axs[1]);fig.suptitle('Predecessor dependence reproduced | nine legal successor interventions: zero rescues')
for ext in ['png','pdf']:fig.savefig(a.out/f'CHARGER_MECHANISM.{ext}',dpi=200)
plt.close(fig)
for name,fn in [('CHARGER_PAIRED_CONTROL',panel),('CHARGER_GRASP_CROSSOVER',cross)]:
 fig,ax=plt.subplots(figsize=(6.5,4.3),layout='constrained');fn(ax)
 for ext in ['png','pdf']:fig.savefig(a.out/f'{name}.{ext}',dpi=200)
 plt.close(fig)
