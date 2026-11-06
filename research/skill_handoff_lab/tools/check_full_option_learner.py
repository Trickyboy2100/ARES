"""Regression checks for the actual discrete learner; no simulator or research scores."""
import argparse, copy, json
from pathlib import Path
import torch
from program_rule_option import Assets
from program_full_option import legal
from train_full_option import Learner

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
assets=Assets();flat=Learner(assets,32201,'sac');other=Learner(assets,32202,'sac');factor=Learner(assets,32201,'factor')
assert not torch.equal(flat.actor[0].weight,other.actor[0].weight)
assert not torch.equal(flat.q[0][0].weight,other.q[0][0].weight)
assert torch.equal(flat.q[0][0].weight,factor.q[0][0].weight)
assert torch.linalg.matrix_rank(factor.map)==13
raw=torch.load(a.root/'SHARED_FRONTIER.pt',weights_only=False)['data'];states=[t['s'] for t in raw]
for s in states[::7]:
 assert flat.choose(s)==factor.choose(s)==flat.warm.choose(s)
 assert legal(s)[flat.choose(s)]
# Catches NumPy/Torch probability division regressions at stochastic deployment.
flat.episode=63
for i in range(200):
 s=states[i%len(states)];assert legal(s)[flat.choose(s,train=True)]
data=flat.decorate(copy.deepcopy(raw[:40]),-1000);log=flat.train(data,2,0)
state=copy.deepcopy(flat.state());other.load(state);other.episode=flat.episode
for s in states[:30]: assert flat.choose(s,True)==other.choose(s,True)
flat.train(data,1,1);other.train(data,1,1)
for module in ['actor','q','target']:
 for k,v in getattr(flat,module).state_dict().items():assert torch.equal(v,getattr(other,module).state_dict()[k]),(module,k)
result=dict(distinct_network_seeds=True,matched_cross_arm_q=True,factor_rank=13,initial_policy_checks=len(states[::7]),legal_stochastic_draws=200,exact_optimizer_and_rng_resume=True,unit_updates_per_learner=3,scope='CPU regression only; not research training or physical success')
(a.root/'LEARNER_REGRESSION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
