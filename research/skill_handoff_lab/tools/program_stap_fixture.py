"""Real CPU execution of official CEM bodies vs independent algorithm port.
Synthetic deterministic fixtures only, not robot/task success measurements.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from program_stap_kernel import *

class Dynamics:
    def __init__(self,d=2):self.d=d;self.action_space=Box([-1]*d,[1]*d);self.device=torch.device('cpu');self.calls=[]
    def to(self,device):self.device=torch.device(device);return self
    def decode(self,state,primitive=None):return state
    def rollout(self,obs,skeleton,policies,batch_size=None,time_index=False):
        n=batch_size or 1;self.calls.append(n);s=obs.unsqueeze(0).repeat(n,1);states=[s];actions=[]
        for t,p in enumerate(skeleton):
            policy=policies[t if time_index else p.idx_policy];a=policy.actor.predict(s)
            if a.ndim==1:a=a.unsqueeze(0).repeat(n,1)
            actions.append(a);s=.7*s+.3*a;states.append(s)
        states=torch.stack(states,1);actions=torch.stack(actions,1)
        return (states,actions) if batch_size else (states[0],actions[0])
class Actor:
    def predict(self,s):return torch.tanh(-.25*s+.15)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    Original,evaluate=official_kernel(a.source);rows=[]
    for d,T in [(1,2),(2,2),(2,3)]:
      for seed in [7,19]:
        for improved in [False,True]:
            kw=dict(num_iterations=5,num_samples=16,num_elites=6,standard_deviation=.8,keep_elites_fraction=.5 if improved else 0.,population_decay=.7 if improved else 1.,momentum=.2 if improved else 0.)
            policies=[Agent(Box([-1]*d,[1]*d),Actor(),Critic(lambda s,a,k=k:torch.exp(-((a-(.3 if k==0 else -.2)-.4*s)**2).sum(-1)*2))) for k in range(T)]
            sk=[Primitive(k) for k in range(T)];obs=np.linspace(-.3,.2,d,dtype=np.float32)
            dyn=Dynamics(d);torch.manual_seed(seed);x=Original(policies,dyn,device='cpu',**kw).plan(obs,sk,return_visited_samples=True)
            dyn2=Dynamics(d);torch.manual_seed(seed);y=port_plan(obs,sk,policies,dyn2,**kw)
            checks={k:bool(np.array_equal(np.asarray(getattr(x,k)),np.asarray(y[k]))) for k in ['actions','states','p_success','values','visited_actions','visited_states','p_visited_success','visited_values']}
            assert all(checks.values()),checks
            assert dyn.calls==dyn2.calls
            rows.append(dict(dim=d,skills=T,seed=seed,improved=improved,checks=checks,trajectory_queries=sum(dyn.calls),skill_model_queries=sum(dyn.calls)*T,score=float(x.p_success),iterations=y['iterations'],best_actions=x.actions.tolist()))
    # Product scoring and probability clipping are checked independently.
    states=torch.zeros(2,3,1);actions=torch.zeros(2,2,1)
    vs=[Critic(lambda s,a:torch.tensor([1.2,.5])),Critic(lambda s,a:torch.tensor([.25,-.2]))]
    prob,_,_=evaluate(vs,[lambda s:s]*2,states,actions)
    assert np.array_equal(prob.numpy(),np.array([.25,0.],np.float32))
    source={}
    for name in ['stap/planners/cem.py','stap/planners/base.py','stap/planners/utils.py','stap/dynamics/base.py','stap/utils/spaces.py','LICENSE']:
        data=(a.source/name).read_bytes();out=a.out/'official_source'/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);source[name]=hashlib.sha256(data).hexdigest()
    (a.out/'RESULTS.json').write_text(json.dumps(dict(status='PASS',fixtures=len(rows),comparisons=rows,official_revision=PIN,source_sha256=source,physical_steps=0,optimizer_updates=0,scope='Original algorithm nodes executed through explicit abstract API adapters vs independent port. No official PyBullet environment or trained-checkpoint benchmark reproduction.',probability_clip_product_check=prob.tolist()),indent=2)+'\n')
    print('PASS',len(rows),'official/port exact trajectories; product/clipping exact')
if __name__=='__main__':main()
