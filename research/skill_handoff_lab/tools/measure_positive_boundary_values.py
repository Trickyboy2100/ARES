"""Current values at measured positive reference boundaries; no greedy-Q claim."""
import argparse,json,torch
from pathlib import Path
from continuation_learner import ContinuationLearner
from program_rule_option import Assets
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;torch.set_num_threads(2)
ps=[p for folder in ['design_a','design_b']for p in json.loads((r/folder/'PAIRS.json').read_text())if p['delta']>0];out=[]
for arm in ['sac','witness_soft','witness_qp']:
 d=torch.load(r/f'{arm}_32301/CHECKPOINT_128.pt',weights_only=False);m=ContinuationLearner(Assets(),32301,arm);m.load(d)
 for p in ps:
  with torch.no_grad():q=torch.stack([v(m.norm([p['s']]))for v in m.q]).mean(0)[0]
  out.append(dict(arm=arm,seed=p['seed'],step=p['step'],measured_better=p['action'],measured_worse=p['baseline'],current_actor=m.choose(p['s']),q_better=float(q[p['action']]),q_worse=float(q[p['baseline']]),scope='Recorded reference-policy boundary, not necessarily visited by final policy; shaped softQ is not success probability'))
(r/'POSITIVE_BOUNDARY_VALUES.json').write_text(json.dumps(out,indent=2))
