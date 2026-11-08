"""Measured hole-frame trajectories after the shared physical prefix."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
pairs=sum([json.loads(f.read_text())for f in a.root.glob('design*/PAIRS.json')],[])
chosen=[next(p for p in pairs if p['seed']==s and p['delta']==d)for s,d in [(768024,1),(768032,1),(768056,1),(768000,-1)]]
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
def draw(ax,p):
 for key,label,color in [('baseline_path','Frozen reference','#667085'),('branch_path','One different Option','#007f83')]:
  d=json.load(gzip.open(p[key],'rt'));rows=d['steps'][max(0,p['step']-1):];xyz=np.array([r['evaluation']['peg_head_pos_at_hole'][0]for r in rows]);t=[r['step']for r in rows]
  ax.plot(t,1000*np.linalg.norm(xyz[:,1:],axis=1),label=label,color=color,lw=2)
 ax.set(title=f"Scene {p['seed']} | {'rescue'if p['delta']>0 else 'regression'} | {p['baseline']} → {p['action']}",xlabel='Actual native physical step',ylabel='Peg-head lateral error in hole frame (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.text(.02,.97,f"Option duration: {p['duration']} steps\nSame prefix: {p['step']} steps",transform=ax.transAxes,va='top',fontsize=8)
for ax,pair in zip(axes.flat,chosen):
 draw(ax,pair);sub,sa=plt.subplots(figsize=(5,3.7),layout='constrained');draw(sa,pair)
 for ext in ('png','pdf'):sub.savefig(a.out/f"PAIR_{pair['seed']}.{ext}",dpi=180)
 plt.close(sub)
fig.suptitle('Measured intervention effects under the same frozen recurrent continuation',fontsize=13)
for ext in ('png','pdf'):fig.savefig(a.out/f'PAIR_MECHANISM.{ext}',dpi=180)
(a.out/'PAIR_MECHANISM_SOURCES.json').write_text(json.dumps(chosen,indent=2))
