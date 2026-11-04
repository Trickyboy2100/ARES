"""Scientific panels from full native evaluations; no selected-case aggregation."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();d=json.loads(a.summary.read_text());a.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
names=[n for n in ['removed','persistent','controlled','persistent_controlled'] if n in d]
fig,ax=plt.subplots(figsize=(7,4))
for n in names:
 v=d[n]['evaluations'];keys=sorted(v,key=lambda k:int(k.split('_')[1]));ax.plot([int(k.split('_')[1])*100 for k in keys],[v[k]['success'] for k in keys],'-o',label=n)
ax.axhline(9,color='gray',ls='--',label='Nominal same grasp: 9/16');ax.set(xlabel='New full-native training steps',ylabel='Terminal native successes /16',ylim=(0,16),title='Seen development; common real predecessor');ax.legend(fontsize=9);fig.tight_layout();fig.savefig(a.out/'LEARNING.png');plt.close(fig)
names=[n for n in d if n!='baseline' and list(d[n]['evaluations'].values())[-1]['n']==16];vals=[list(d[n]['evaluations'].values())[-1]for n in names]
fig,ax=plt.subplots(figsize=(max(7,len(names)*1.1),4));xs=range(len(names));ax.bar(xs,[len(v['rescues'])for v in vals],label='Rescued nominal failures',color='#239d89');ax.bar(xs,[-len(v['regressions'])for v in vals],label='Lost nominal successes',color='#d95f5f');ax.axhline(0,color='gray',lw=.7);ax.set_xticks(list(xs),names,rotation=25,ha='right');ax.set(ylabel='Paired cases /16',title='Final checkpoint: rescue and regression');ax.legend();fig.tight_layout();fig.savefig(a.out/'PAIRED.png');plt.close(fig)
