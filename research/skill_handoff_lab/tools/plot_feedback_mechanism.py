"""Measured physical curves from strict native-start feedback counterfactuals."""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;rows=json.loads((r/'native_recovery_verification/physics/RESULTS.json').read_text());selected=[next(z for z in rows if z['seed']==755011 and z['tag']==tag)for tag in ['zero','training_schedule']];fig,axs=plt.subplots(4,1,figsize=(10,10),sharex=True)
for row,color,label in zip(selected,['#2865a0','#c55d28'],['Nominal continuation: success','Training-mined corrections: failure']):
 d=json.load(gzip.open(row['path'],'rt'));ss=[d['initial']]+d['steps'];head=np.array([z['evaluation']['peg_head_pos_at_hole'][0]for z in ss])*1000;att=np.array([Rotation.from_quat(np.r_[np.array(z['tcp'])[4:],z['tcp'][3]]).inv().apply(np.array(z['peg'][:3])-z['tcp'][:3])for z in ss])*1000
 axs[0].plot(head[:,0],color=color,label=label);axs[1].plot(np.linalg.norm(head[:,1:],axis=1),color=color);axs[2].plot(np.arange(48,101),np.linalg.norm(att[48:]-att[48],axis=1),color=color);rear=[z for z in row['decisions']if z['stage']==1];xx=[z['step']for z in rear]+[100];yy=[z['id']for z in rear]+[rear[-1]['id']];axs[3].step(xx,yy,where='post',color=color)
for ax,y in zip(axs,['Peg-head axial position (mm)','Peg-head lateral offset (mm)','TCP-relative peg drift (mm)','Executed parameter ID']):ax.set_ylabel(y);ax.axvline(48,color='gray',ls='--');ax.grid(alpha=.2)
axs[0].legend();axs[-1].set_xlabel('Actual control step (native horizon = 100)');axs[-1].set_yticks(range(8));fig.suptitle('Same native initial state and exact 48-step prefix | seed 755011\nA local-training success does not establish a full-task recovery');fig.tight_layout();(r/'figures').mkdir(exist_ok=True);fig.savefig(r/'figures/MECHANISM.png',dpi=180)
for i,ax in enumerate(axs):
 box=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted());fig.savefig(r/f'figures/MECHANISM_PANEL_{i+1}.png',dpi=180,bbox_inches=box.expanded(1.04,1.08))
plt.close(fig)
(r/'figures/MECHANISM_SOURCES.json').write_text(json.dumps([dict(path=z['path'],sha256=hashlib.sha256(Path(z['path']).read_bytes()).hexdigest())for z in selected],indent=2))
