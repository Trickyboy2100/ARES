"""Measured training episode curves and implemented-method diagram."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
p=argparse.ArgumentParser();p.add_argument('--grid',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
fig,axs=plt.subplots(1,2,figsize=(12,4));colors={'ppo':'tab:blue','residual':'tab:orange','contract_blind':'tab:green','contract':'tab:red'}
for d in sorted(a.grid.glob('multicontract_v19_*')):
    if d.name.endswith(('_val','_test')) or not (d/'industrial_episodes.json').exists():continue
    rows=sorted(json.loads((d/'industrial_episodes.json').read_text()),key=lambda r:r['control_step'])
    if not rows:continue
    meta=json.loads((d/'manifest.json').read_text());method=meta['controller'];seed=meta['industrial_geometry']['training_seed']
    xs=[r['control_step']*16 for r in rows];window=64
    for ax,key in zip(axs,['success','episode_return']):
        values=[float(r[key]) for r in rows];ys=[np.mean(values[max(0,i-window+1):i+1]) for i in range(len(rows))]
        ax.plot(xs,ys,color=colors[method],alpha=.7,label=f'{method}, seed {seed}')
axs[0].set_ylabel('Rolling 64 measured training episodes: success fraction');axs[0].set_ylim(-.02,1.02)
axs[1].set_ylabel('Rolling 64 measured episode return (frozen v2)')
for ax in axs:ax.set_xlabel('Real control interactions (16 independent PhysX envs)');ax.grid(alpha=.2)
axs[1].legend(fontsize=7,ncol=2);fig.suptitle('T1 + yaw-aligned T2 joint training | actual terminal outcomes, not synthetic labels')
fig.tight_layout();fig.savefig(a.output/'fig_multitask_learning.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4));ax.set_xlim(0,12);ax.set_ylim(0,4);ax.axis('off')
boxes=[(.1,2.5,'ARES Skill Runtime\ncuRobo Preplace (existing)'),(3.3,2.5,'Observed q, v, target, geometry\n+ Skill Contract c (15 fields)'),(6.5,2.5,'PPO residual policy\n25 + 15 inputs, MLP 64 / 64'),
    (3.3,.5,'Nominal target feedback\n(same across all residual baselines)'),(6.5,.5,'Sum → axis mask → step clip\nworkspace → bounded DLS IK'),(9.5,1.5,'PhysX joint drives\ngeometry / pose / deadline\nindependent FCL audit')]
for x,y,label in boxes:
    width=2.8 if x<9 else 2.4
    ax.add_patch(FancyBboxPatch((x,y),width,.95,boxstyle='round,pad=.08',facecolor='#eaf1fa',edgecolor='#35618d'))
    ax.text(x+width/2,y+.475,label,ha='center',va='center',fontsize=9)
for start,end in [((2.9,3),(3.2,3)),((6.1,3),(6.4,3)),((7.9,2.45),(7.9,1.55)),((6.1,1),(6.4,1)),((9.3,1),(9.55,1.8))]:
    ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#35618d'})
ax.text(.1,.1,'Implemented structure, not a proven novelty or performance gain. Welded held object; no contact-grasp claim.',fontsize=10)
fig.tight_layout();fig.savefig(a.output/'fig_method_framework.png',dpi=160);plt.close(fig)
