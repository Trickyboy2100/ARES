"""Audit two observed development rescues without assigning last-action credit."""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--figures',type=Path,required=True);a=p.parse_args();a.figures.mkdir(parents=True,exist_ok=True)
h=json.loads((a.root/'sac_32201/HISTORY.json').read_text());out={};fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for row,(seed,boundary) in enumerate([(765402,65),(765423,71)]):
 ds=[];info=[]
 for ck,label,color in [(0,'Warm initialization','#888888'),(128,'Flat SAC128','#287a98')]:
  e=next(x for x in h if x['kind']=='eval' and x['checkpoint']==ck and x['row']['seed']==seed);f=Path(e['row']['path']);d=json.load(gzip.open(f,'rt'));ds.append(d);states=[d['initial']]+d['steps'];head=np.array([s['evaluation']['peg_head_pos_at_hole'][0] for s in states])*1000
  info.append(dict(checkpoint=ck,path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),success=e['row']['success'],first_native_success=next((i for i,s in enumerate(states) if s['evaluation']['success'][0]),None),boundary_actions=[x for x in e['events'] if x['boundary']]))
  axes[row,0].plot(range(101),head[:,0],label=label,color=color);axes[row,1].plot(range(101),np.linalg.norm(head[:,1:],axis=1),label=label,color=color)
 def prefix(d):
  v=dict(initial=d['initial'],steps=[{k:s[k] for k in ['q','qvel','tcp','peg','action']} for s in d['steps'][:boundary]])
  return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
 assert prefix(ds[0])==prefix(ds[1]);out[str(seed)]=dict(records=info,identical_prefix_through_step=boundary,prefix_sha=prefix(ds[0]),scope='original-reset full native reexecution; observed deployment contrast, not single-action counterfactual credit')
 for ax in axes[row]:ax.axvline(boundary,color='#b35c43',ls='--',lw=1);ax.set_title(f'Scene{seed}: common prefix through step{boundary}');ax.legend(fontsize=9)
 axes[row,0].set_ylabel('Peg-head axial coordinate (mm)');axes[row,1].set_ylabel('Peg-head lateral norm (mm)')
for ax in axes[-1]:ax.set_xlabel('Actual native physical step')
out['credit_caution']='765402 first becomes natively successful at93, before its physical5 intervention at94. Do not attribute rescue to that late action. Frozen proposal-support deployment ablation tests whether added catalogue choices are necessary for current observed scores.'
(a.root/'RESCUE_MECHANISMS.json').write_text(json.dumps(out,indent=2));fig.tight_layout()
for suffix in ['png','pdf']:fig.savefig(a.figures/('learned_rescue_physics.'+suffix),dpi=220)
fig.canvas.draw()
for i,seed in enumerate([765402,765423]):
 for j,name in enumerate(['axial','lateral']):
  box=axes[i,j].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.03,1.05)
  for suffix in ['png','pdf']:fig.savefig(a.figures/(f'rescue_{seed}_{name}.'+suffix),dpi=220,bbox_inches=box)
print(json.dumps({k:v.get('identical_prefix_through_step') for k,v in out.items() if isinstance(v,dict)}))
