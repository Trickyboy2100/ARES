"""Scientific figures from verified native outcome summaries only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();s=json.loads((a.out/'SUMMARY.json').read_text());f=a.out/'figures';f.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
def save(fig,name):
 fig.tight_layout();fig.savefig(f/(name+'.png'));fig.savefig(f/(name+'.pdf'));plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.2));keys=list(s['mechanism']);labels=['Continue','Rebuild +3 mm','Rebuild -3 mm','Plan-relative +3 mm','Plan-relative -3 mm'];values=[s['mechanism'][k]['success']for k in keys]
ax.bar(range(5),values,color=['#64748b','#c75b52','#c75b52','#2e8b83','#2e8b83']);ax.set_xticks(range(5),labels,rotation=16,ha='right');ax.set_ylim(0,12);ax.set_ylabel('Native terminal successes / 12');ax.set_title('Same-prefix interventions: retention, not new rescues\n4 natural scenes × 3 actual grasps; native Peg100')
for i,v in enumerate(values):ax.text(i,v+.15,str(v),ha='center')
save(fig,'01_parameter_effect')
fig,ax=plt.subplots(figsize=(8,4));groups=[s['mechanism'][k]['target_delta_mm'] for k in keys[1:]];ax.boxplot(groups,tick_labels=labels[1:],showfliers=True);ax.axhline(3,ls='--',color='#64748b',label='3 mm parameter label');ax.set_ylabel('Change from active requested target (mm)');ax.set_title('Attachment refresh changes the action origin\n8 reachable handoffs; requested target, not actual TCP displacement');ax.tick_params(axis='x',rotation=15);ax.legend();save(fig,'02_target_displacement')
fig,ax=plt.subplots(figsize=(8,4));colors=['#3b6fb6','#cc7a36']
for (name,arm),color in zip(s['arms'].items(),colors):
 xs=[int(k)for k in arm['evaluation']];ys=[sum(v.values())for v in arm['evaluation'].values()];ax.plot(xs,ys,'o-',label=name.replace('online_',''),color=color)
for name,vals in s['controls'].items():ax.axhline(sum(vals.values()),ls='--'if name=='nominal'else':',label=name,color='#555555'if name=='nominal'else'#2e8b83')
ax.set_ylim(0,16);ax.set_xlabel('Own-policy training episodes');ax.set_ylabel('Native successes / 16 development scenes');ax.set_title('Two-stage Option learning | same inherited initialization\nFixed final checkpoint; development is not sealed testing');ax.legend();save(fig,'03_online_development')
fig,ax=plt.subplots(figsize=(8,4))
for name,color in zip(s['arms'],colors):
 h=json.loads((a.root/name/'HISTORY.json').read_text());ys=[int(v['row']['success'])for v in h if v['job']['kind']=='train'];ax.plot(np.arange(8,len(ys)+1),np.convolve(ys,np.ones(8)/8,mode='valid'),label=name.replace('online_',''),color=color)
ax.set_ylim(0,1);ax.set_xlabel('Own-policy training episodes');ax.set_ylabel('Trailing 8-episode training success');ax.set_title('Exploratory physical collection; not policy evaluation');ax.legend();save(fig,'04_collection')
