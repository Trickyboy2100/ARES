"""Measured execution progress, not target-distance amplification."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();f=a.root/'figures';f.mkdir(exist_ok=True);series=[]
for name,label,c in [('diag_measured_g1','Measured policy: continue','#246a73'),('diag_relative_g1','Active-target policy: +3mm replan','#bc4749')]:
 r=next(v for v in json.loads((a.root/name/'RESULTS.json').read_text())if v['seed']==759205);d=json.load(gzip.open(r['path'],'rt'));series.append((label,c,r,d))
dx,dy=series[0][3],series[1][3]
identical=0
for x,y in zip(dx['steps'],dy['steps']):
 if any(x[k]!=y[k]for k in ['q','qvel','tcp','peg','action']):break
 identical+=1
assert identical==94
intervention=next(v for v in dy['summary']['decisions']if v['step']==identical)
assert intervention['id']==2 and intervention['replanned']
probe=a.root/'clock_probe/RESULTS.json'
if probe.exists():
 r=next(v for v in json.loads(probe.read_text())if v['seed']==759205 and v['method']=='zero_offset_replan');d=json.load(gzip.open(r['path'],'rt'))
 assert all(x[k]==y[k]for x,y in zip(dx['steps'][:94],d['steps'][:94])for k in ['q','qvel','tcp','peg','action'])
 series.append(('Zero-offset replan (controlled probe)','#7b6d8d',r,d))
fig,axs=plt.subplots(2,1,figsize=(8,6),sharex=True)
for label,c,r,d in series:
 xyz=np.array([v['evaluation']['peg_head_pos_at_hole'][0]for v in d['steps']])*1000;steps=[v['step']for v in d['steps']];axs[0].plot(steps,xyz[:,0],label=label,color=c,ls='--' if 'Zero-offset' in label else '-');axs[1].plot(steps,xyz[:,1],color=c,ls='--' if 'Zero-offset' in label else '-')
for ax in axs:ax.axvline(94,color='gray',ls='--');ax.set_xlim(85,100);ax.grid(alpha=.2)
axs[0].set_ylim(-145,50);axs[1].set_ylim(-5,5);axs[0].set_ylabel('Peg head X in hole frame (mm)');axs[1].set_ylabel('Peg head Y in hole frame (mm)');axs[1].set_xlabel('Actual native physical step');axs[0].legend(fontsize=8);axs[0].set_title('Same grasp, identical 94-step physical prefix; different continuation')
axs[0].annotate('3 mm target update\nqueue 8/16 → 0/15\n6 physical steps left',xy=(94,-82),xytext=(85.5,35),arrowprops=dict(arrowstyle='->'),fontsize=9,va='top')
fig.tight_layout();fig.savefig(f/'schedule_restart_mechanism.png',dpi=240);fig.savefig(f/'schedule_restart_mechanism.pdf')
(a.root/'SCHEDULE_MECHANISM.json').write_text(json.dumps(dict(seed=759205,grasp=1,prefix=identical,remaining=100-identical,old_queue_cursor=intervention['queue_before']['cursor'],old_queue_length=len(intervention['queue_before']['queue']),new_queue_length=intervention['queue_length'],requested_delta_mm=1000*intervention['requested_target_delta_m'],old_success=series[0][2]['success'],new_success=series[1][2]['success'],scope='seen diagnostic; no causal attribution to target magnitude alone; complete100step reexecution',sources=[dict(path=r['path'],sha256=r['sha256'])for _,_,r,_ in series]),indent=2))
