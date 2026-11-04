"""Same-initial-state measured native trajectories: benefit and regression."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
for seed in [753212,753209]:
 fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True)
 for name,label,color in [('baseline','Nominal continuation','#555555'),('controlled_consolidated','Controlled-exploration SAC','#167cb5')]:
  path=next((a.root/name/'physics').glob(f'*{seed}*.gz'));d=json.loads(gzip.decompress(path.read_bytes()));states=[d['initial']]+d['steps'];head=np.array([s['evaluation']['peg_head_pos_at_hole'][0]for s in states])*1000
  axes[0].plot(range(101),head[:,0],label=f"{label}: native {'success' if d['summary']['success'] else 'failure'}",color=color)
  dec=[x for x in d['summary']['decisions'] if x['stage']==1];axes[1].step([x['step']for x in dec]+[100],[x['id']for x in dec]+[dec[-1]['id']],where='post',label=label,color=color)
 axes[0].set(ylabel='Peg head axial position (mm)',title=f'Same real predecessor | scene {seed} | native100');axes[0].legend(fontsize=9);axes[1].set(xlabel='Measured control step',ylabel='Executed parameter ID',yticks=range(8),xlim=(0,100));fig.tight_layout();fig.savefig(a.out/f'PHYSICS_{seed}.png');plt.close(fig)
