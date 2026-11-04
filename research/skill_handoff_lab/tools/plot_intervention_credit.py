"""Publication-style plots from actual native outcomes, no model-based success."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
s=json.loads((a.root/'SUMMARY.json').read_text());rs=s['results'];plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
display={'zero_suffix':'Nominal continuation','budget_refine_zplus':'Frozen feedback rule','awac':'Fixed-pair / AWAC-style','sac':'Fixed-pair / SAC','measured_gap':'Fixed-pair / measured gap','current_sac':'Current-policy pairs / SAC','current_awac':'Current-policy pairs / AWAC-style','current_measured_gap':'Current-policy pairs / measured gap','current_measured_preference':'Current-policy pairs / preference','memory_awac':'Replay consolidation / AWAC-style','memory_measured_preference':'Replay consolidation / preference'}
names=[display.get(r['method'].split('/')[0],r['method'])+(' (offline)' if r['method'].endswith('offline_eval') else '') for r in rs];y=np.arange(len(rs));fig,ax=plt.subplots(figsize=(12,max(4,len(rs)*.42)));ax.barh(y,[r['success'] for r in rs],color='#326a99');ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,24);ax.set_xlabel('Native full-task successes / 24 development initial states');ax.set_title('Native Peg100 | no reset | budgets differ BETWEEN comparison groups')
for k,r in enumerate(rs):ax.text(r['success']+.15,k,f"{r['success']}/24  (+{r['rescue']} / -{r['regression']})",va='center',fontsize=10)
fig.tight_layout();fig.savefig(a.out/'RESULTS.png',dpi=180);plt.close(fig)
# Intervention-level labels count repeated real contrasts, not independent scenes.
labels=[];pos=[];neg=[];tie=[]
for name in ['pairs','late_pairs','rule_pairs','current_sac','current_measured_gap']:
 f=a.root/name/'PAIRS.json'
 if not f.exists():continue
 ds=json.loads(f.read_text());labels.append(name);pos.append(sum(d['delta']>0 for d in ds));neg.append(sum(d['delta']<0 for d in ds));tie.append(sum(d['delta']==0 for d in ds))
fig,ax=plt.subplots(figsize=(9,4));ax.barh(labels,pos,color='#2b8a78',label='Rescue');ax.barh(labels,neg,left=pos,color='#b84f5f',label='Regression');ax.barh(labels,tie,left=np.array(pos)+neg,color='#c7cbd1',label='Same terminal outcome');ax.set_xlabel('Measured intervention contrasts (clustered within initial states)');ax.legend();fig.tight_layout();fig.savefig(a.out/'INTERVENTIONS.png',dpi=180);plt.close(fig)
# Actual causal pair, same measured prefix; not a learned generalization claim.
f=a.root/'current_sac/PAIRS.json'
if f.exists():
 ds=json.loads(f.read_text());v=next((x for x in ds if x['delta']>0),None)
 if v:
  fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
  for key,label,color in [('base_path','Continue queue','#b84f5f'),('action_path','Single measured-target intervention','#326a99')]:
   d=json.load(gzip.open(v[key],'rt'));z=[d['initial']]+d['steps'];head=np.array([s['evaluation']['peg_head_pos_at_hole'][0] for s in z])*1000
   axes[0].plot(head[:,0],label=label,color=color);axes[1].plot(np.linalg.norm(head[:,1:],axis=1),color=color)
  for ax in axes:ax.axvline(v['prefix_steps'],color='gray',ls='--');ax.grid(alpha=.2)
  axes[0].set_ylabel('Peg-head axial position (mm)');axes[1].set_ylabel('Lateral error (mm)');axes[1].set_xlabel('Actual native control step');axes[0].legend();axes[0].set_title(f"Training initial state {v['scene']} | exact prefix | physical effect, not method gain")
  fig.tight_layout();fig.savefig(a.out/'PHYSICAL_EFFECT.png',dpi=180);plt.close(fig)
