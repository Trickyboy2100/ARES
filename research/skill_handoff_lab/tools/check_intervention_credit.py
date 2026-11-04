"""Real checkpoint one-step isolation fixture for policy-credit changes."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from program_budget_feedback_sac import FeedbackSAC
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
data=json.loads((a.parent/'TRANSITIONS.json').read_text());labels=json.loads((a.parent/'LABELS.json').read_text());pairs=json.loads((a.pairs/'PAIRS.json').read_text());assert pairs and all(x['prefix_exact'] for x in pairs)
models=[]
for mode in ['sac','awac','measured_gap','measured_preference']:
 m=FeedbackSAC.load(a.parent/'FINAL.pt');m.train(data,labels,1,successor_only=True,intervention_pairs=pairs,actor_rule=mode);models.append(m)
for m in models[1:]:
 assert all(torch.equal(x,y) for x,y in zip(models[0].q.parameters(),m.q.parameters()))
 assert any(not torch.equal(x,y) for x,y in zip(models[0].actor.parameters(),m.actor.parameters()))
 assert all(torch.isfinite(x).all() for x in m.actor.parameters())
print('PASS same first critic update, different finite Actor gradients; real exact-prefix labels')
