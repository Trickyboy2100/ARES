"""Probe frozen policies on actual nominal-state support, not counterfactual success labels."""
import argparse,json,gzip
from pathlib import Path
import numpy as np,torch
from program_budget_feedback_sac import FeedbackSAC
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
states=[];groups=[]
for path in sorted((a.root/'baseline/physics').glob('*.gz')):
 d=json.loads(gzip.decompress(path.read_bytes()))
 for x in d['summary']['decisions']:
  if x['stage']==1:states.append(x['obs']);groups.append(bool(d['summary']['success']))
result={}
for directory in a.root.iterdir():
 if not directory.is_dir() or directory.name=='baseline':continue
 for path in sorted(directory.glob('*.pt')):
  m=FeedbackSAC.load(path)
  with torch.no_grad():
   x=m.x(states);p=m.probs(x).numpy();q=torch.stack([net(x) for net in m.q]).mean(0).numpy()
  result[str(path.relative_to(a.root))]={}
  for group in [True,False]:
   mask=np.array(groups)==group;z=p[mask];v=q[mask];result[str(path.relative_to(a.root))][str(group)]=dict(n=int(mask.sum()),p_continue=float(z[:,0].mean()),p_hold=float(z[:,7].mean()),p_replan=float(z[:,1:7].sum(1).mean()),actor_continue=int((z.argmax(1)==0).sum()),q_replan_advantage=float((v[:,1:7].max(1)-v[:,0]).mean()),entropy=float(-(z*np.log(z.clip(1e-9))).sum(1).mean()))
a.out.write_text(json.dumps(dict(scope='Nominal visited states; Q differences are predictions, not measured rescue',models=result),indent=2))
