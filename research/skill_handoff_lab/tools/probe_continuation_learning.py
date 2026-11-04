"""Frozen-model first-action versus measured continuation support analysis."""
import argparse,json,gzip
from pathlib import Path
import torch,numpy as np
from program_budget_feedback_sac import FeedbackSAC
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
rows=[]
for ck in sorted(a.run.glob('*CHECKPOINT_*.pt')):
 m=FeedbackSAC.load(ck)
 for pair in json.loads((a.run/'PAIRS.json').read_text()):
  if pair['delta']==0:continue
  with torch.no_grad():prob=m.probs(m.x([pair['s']]))[0].numpy()
  tails=[]
  for key in ['base_path','action_path']:
   d=json.load(gzip.open(pair[key]));ds=[x for x in d['summary']['decisions']if x['step']>pair['prefix_steps']]
   # Descriptive primitive likelihood only; not an Option policy likelihood.
   with torch.no_grad():
    pr=m.probs(m.x([x['obs'] for x in ds])).numpy() if ds else np.zeros((0,8))
   tails.append(dict(decisions=len(ds),argmax_agreement=sum(int(p.argmax())==x['id']for p,x in zip(pr,ds))))
  rows.append(dict(checkpoint=ck.name,seed=pair['scene'],delta=pair['delta'],action=pair['a'],argmax=int(prob.argmax()),p_action=float(prob[pair['a']]),p_zero=float(prob[0]),primitive_suffix_diagnostic=tails))
(a.run/'POLICY_PROBE.json').write_text(json.dumps(rows,indent=2))
for x in rows:
 if x['checkpoint'].endswith('_8.pt'):print(x['checkpoint'],x['seed'],x['delta'],x['argmax'],round(x['p_action'],4))
