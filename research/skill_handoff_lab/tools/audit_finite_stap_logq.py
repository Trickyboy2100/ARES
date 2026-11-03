"""Retain established log-Q numerical repair in new finite STAP comparator."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,legal
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);rows=json.loads((a.root/'baseline_eval/physics/RESULTS.json').read_text());checks=[]
for seed in [31301,31302]:
 m=Learner.load(a.root/f'offline/stap_{seed}_2000.pt')
 for row in rows:
  if row['tag']!=f'stap_{seed}':continue
  for d in row['decisions']:
   with torch.no_grad():
    x=m.x([d['obs']]);qs=torch.stack([q(x)for q in m.q]).masked_fill(~legal(x)[None],-1e9);v=qs.mean(0).sigmoid()
    if d['stage']==0:
     for aa in range(3):
      inp=x.clone();inp[:,-3:]=0;inp[:,-3+aa]=1;z=m.dyn(inp);z[:,44]=1;v[:,aa]*=torch.stack([q(z)for q in m.q]).mean(0)[:,:2].sigmoid().max(1).values
    old=int(v.argmax());new=m.choose(d['obs']);assert old==new==d['id'];checks.append(dict(seed=seed,scene=row['seed'],stage=d['stage'],old=old,log_domain=new))
# Underflow adversary: both products zero but log-domain must preserve order.
q=torch.tensor([[-100.,-101.],[-100.,-102.]]);prod=q.sigmoid().prod(1);stable=torch.nn.functional.logsigmoid(q).sum(1);assert prod.tolist()==[0.,0.] and stable[0]>stable[1]
(a.root/'STAP_LOGQ_AUDIT.json').write_text(json.dumps(dict(measured_decisions=checks,all_actions_identical=True,underflow_fixture=dict(product=prod.tolist(),log_scores=stable.tolist()),note='new comparator initially used products; restored inherited log-domain invariant; stored physical baseline actions verified unchanged'),indent=2));print(len(checks),'exact physical decisions; underflow fixture PASS')
