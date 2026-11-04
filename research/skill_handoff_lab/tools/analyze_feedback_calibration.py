import sys,json,collections
from pathlib import Path
import torch,numpy as np
torch.set_num_threads(2)
from program_budget_feedback_sac import FeedbackSAC
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out={}
for mode in ['ordinary','suffix','budget']:
 d=r/f'train_{mode}_v4_31501';m=FeedbackSAC.load(d/'FINAL.pt');rows=json.loads((d/'EVAL_96.json').read_text());details=[]
 for row in rows:
  rear=[z for z in row['decisions']if z['stage']==1]
  if not rear:continue
  s=rear[0]['obs'];x=m.x([s])
  with torch.no_grad():h=float(m.probability(x,torch.tensor([s[41]*100])));v=float(m.scalar(x).sigmoid()[0,0])
  details.append(dict(scene=row['seed'],success=row['success'],grasp=row['decisions'][0]['id'],handoff_step=rear[0]['step'],remaining=s[41]*100,hazard_probability=h,scalar_probability=v,rear_actions=dict(collections.Counter(z['id']for z in rear))))
 out[mode]=dict(rows=details,hazard_brier=float(np.mean([(z['hazard_probability']-z['success'])**2 for z in details])),scalar_brier=float(np.mean([(z['scalar_probability']-z['success'])**2 for z in details])))
(r/'PARENT_CALIBRATION.json').write_text(json.dumps(out,indent=2));print({m:{k:v for k,v in z.items()if k!='rows'}for m,z in out.items()})
