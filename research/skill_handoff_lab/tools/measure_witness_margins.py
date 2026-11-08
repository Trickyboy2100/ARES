"""Observed-state logit margin diagnosis; not a neighborhood safety certificate."""
import torch,json,numpy as np
from pathlib import Path
from continuation_learner import ContinuationLearner
from program_rule_option import Assets
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/continuation_retention_20261108');torch.set_num_threads(2);graph=torch.load(r/'WITNESS_GRAPH.pt',weights_only=False);ss=[n['s']for n in graph['nodes']];out=[]
for name in ['sac_32301','retention_32301','witness_soft_32301','witness_qp_32301']:
 for k in [0,64,128]:
  f=r/name/f'CHECKPOINT_{k}.pt'
  if not f.exists():continue
  state=torch.load(f,weights_only=False);m=ContinuationLearner(Assets(),32301,state['arm']);m.load(state);base,mask=m.context(ss);allowed=torch.zeros(len(ss),13,dtype=torch.bool)
  for i,n in enumerate(graph['nodes']):allowed[i,n['allowed']]=True
  with torch.no_grad():z=m.logits(m.norm(ss),base,mask,ss);margin=(z.masked_fill(~allowed,-1e9).max(1).values-z.masked_fill(allowed,-1e9).max(1).values).numpy();p=z.softmax(1);mass=(p*allowed).sum(1).numpy();bad=(~allowed[torch.arange(len(ss)),z.argmax(1)]).numpy()
  for group,idx in [('all',np.arange(len(ss))),('requires_nonzero',np.array([i for i,n in enumerate(graph['nodes'])if 0 not in n['allowed']]))]:
   out.append(dict(policy=name,checkpoint=k,group=group,n=len(idx),violations=int(bad[idx].sum()),margin_quantiles=np.quantile(margin[idx],[0,.1,.5,.9,1]).tolist(),mean_allowed_mass=float(mass[idx].mean()),margins=margin[idx].tolist(),history_sha=[graph['nodes'][i]['history_sha']for i in idx],scope='Initial measured training witnesses only; supports descriptive comparison, not causal generalization claim'))
(r/'WITNESS_MARGINS.json').write_text(json.dumps(out,indent=2));print(json.dumps([x for x in out if x['group']=='requires_nonzero'],indent=2))
