"""Analytic tests of finite legal support and measured-handoff target semantics."""
import json,tempfile
import numpy as np,torch
from pathlib import Path
from program_option_improvement import Improved,utility,suffix_support,state_key
from program_discrete_options import legal

def main():
 torch.set_num_threads(2);x=np.zeros(54);z=x.copy();z[44]=1;z[41]=.2
 rows=[dict(s=x.tolist(),next=z.tolist(),stage=0,a=0,r=0.,done=False,duration=80,outcome=0.),dict(s=z.tolist(),next=z.tolist(),stage=1,a=0,r=0.,done=True,duration=20,outcome=0.),dict(s=z.tolist(),next=z.tolist(),stage=1,a=1,r=1.,done=True,duration=20,outcome=1.)]
 support=suffix_support(rows);assert support[state_key(z)]=={0:0.,1:1.}
 q=torch.tensor([[[.2,.8,.4]],[[.4,.4,.9]]]);assert torch.allclose(utility(q,0),q.mean(0));assert torch.allclose(utility(q,-.831559),q.min(0).values,atol=2e-6)
 changed=z.copy();changed[41]=.19;assert state_key(changed)not in support
 out=[]
 for kind in ['sac_min','sac_mean','vi_mean','vi_expectile','usac','handoff','handoff_complete','lower_bound','bee']:
  m=Improved(kind,31401);m.train(rows,steps=200)
  assert m.choose(z)<2 and np.isfinite(list(m.history[-1].values())).all()
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'m.pt';m.save(f);n=Improved.load(f);assert np.array_equal(m.values(z)[0],n.values(z)[0]);m.train(rows,steps=1);n.train(rows,steps=1)
   assert all(torch.equal(v,n.q.state_dict()[k])for k,v in m.q.state_dict().items())
  out.append(dict(method=kind,legal_action=m.choose(z),updates=m.updates))
 three=[dict(t)for t in rows];three[2]['r']=0.;three[2]['outcome']=0.;third=dict(three[2],a=2,r=1.,outcome=1.);three.append(third)
 for cls in [Improved]:
  m=cls('handoff',31401,successor_actions=3);m.train(three,steps=400);assert m.choose(z)==2
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'three.pt';m.save(f);n=cls.load(f);assert n.successor_actions==3 and n.choose(z)==2
  try:cls('handoff',31401).train(three,steps=1)
  except AssertionError:pass
  else:raise AssertionError('three-action data accepted by two-action contract')
 print(json.dumps(dict(status='PASS',checks=['legal terminal support','budget-sensitive exact key','USAC min equivalence','save/load optimizer continuation'],fixtures=out),indent=2))
if __name__=='__main__':main()
