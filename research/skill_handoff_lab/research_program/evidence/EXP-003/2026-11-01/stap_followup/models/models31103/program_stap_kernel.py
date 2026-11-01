"""STAP CEM kernel port and execution of pinned original algorithm bodies.

Algorithm attribution: STAP (Agia et al., ICRA2023), MIT (IPRL2022).
No official environment/checkpoint reproduction claimed. Abstract API adapters
replace package imports only; original CEM class and Q-product function run.
"""
import abc,ast,dataclasses,functools,hashlib,json,types
from pathlib import Path
import numpy as np
import torch

PIN='0fe0dcbd5ede5ce4ee3d952642f70b5550a0813c'
DEFAULT_SOURCE=Path('/home/andyee/robot-research/ARES-Skilllab/external/STAP-0fe0dcbd')

class Box:
    def __init__(self,low,high):
        self.low=np.array(low,np.float32);self.high=np.array(high,np.float32);self.shape=self.low.shape;self.dtype=np.dtype('float32')

def null_tensor(space,batch_shape=(),device=None):
    if isinstance(batch_shape,int):batch_shape=(batch_shape,)
    return torch.full((*batch_shape,*space.shape),float('nan'),device=device)

class Constant:
    def __init__(self,value):self.constant=value
    def predict(self,state):return self.constant

class Critic:
    def __init__(self,fn):self.fn=fn
    def predict(self,s,a):return self.fn(s,a)

class Agent:
    def __init__(self,space,actor,critic):self.action_space=space;self.actor=actor;self.critic=critic
    def to(self,device):return self

class ConstantAgent(Agent):
    def __init__(self,action,policy):
        super().__init__(policy.action_space,types.SimpleNamespace(network=Constant(action)),policy.critic)
        self.actor.predict=self.actor.network.predict

@dataclasses.dataclass
class Primitive:
    idx_policy:int
    def get_policy_args(self):return None

def official_kernel(source=DEFAULT_SOURCE):
    """Execute original nodes without editing algorithm statements.

    Only package dependency imports, type annotations and evaluate_trajectory's
    unbatched decorator wrapper are replaced; inputs here are explicitly batched.
    """
    ns=dict(np=np,torch=torch,abc=abc,dataclasses=dataclasses,functools=functools,
            agents=types.SimpleNamespace(ConstantAgent=ConstantAgent),
            networks=types.SimpleNamespace(Constant=Constant,critics=types.SimpleNamespace(Critic=Critic,ProbabilisticCritic=type('ProbabilisticCritic',(),{}),EnsembleDetectorCritic=type('EnsembleDetectorCritic',(),{}))),
            spaces=types.SimpleNamespace(null_tensor=null_tensor),tensors=types.SimpleNamespace(device=lambda x:'cpu' if x=='auto' else x))
    def execute(file,names):
        tree=ast.parse((source/file).read_text());nodes=[x for x in tree.body if getattr(x,'name',None) in names]
        assert len(nodes)==len(names)
        for n in nodes:
            if isinstance(n,ast.FunctionDef):n.decorator_list=[]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(source/file),'exec'),ns)
    execute('stap/planners/base.py',['PlanningResult','Planner']);ns['planners']=types.SimpleNamespace(Planner=ns['Planner'],PlanningResult=ns['PlanningResult'])
    execute('stap/planners/utils.py',['evaluate_trajectory']);ns['utils']=types.SimpleNamespace(evaluate_trajectory=ns['evaluate_trajectory'])
    execute('stap/planners/cem.py',['CEMPlanner'])
    return ns['CEMPlanner'],ns['evaluate_trajectory']

def port_plan(observation,skeleton,policies,dynamics,num_iterations=8,num_samples=128,num_elites=16,standard_deviation=1.,keep_elites_fraction=0.,population_decay=1.,momentum=0.):
    """Independent compact port: identical RNG order, Q product, elite updates."""
    count_elite=max(2,min(num_elites,num_samples//2));keep=int(keep_elites_fraction*count_elite+.5)
    low=torch.stack([torch.from_numpy(policies[p.idx_policy].action_space.low) for p in skeleton]);high=torch.stack([torch.from_numpy(policies[p.idx_policy].action_space.high) for p in skeleton])
    n=num_samples*sum(policies[p.idx_policy].action_space.shape[0] for p in skeleton)
    obs=torch.from_numpy(observation);_,mean=dynamics.rollout(obs,skeleton,policies);std=standard_deviation*.5*(high-low)
    best=None;visited=[];iterations=[];calls=1
    with torch.no_grad():
        for it in range(num_iterations):
            samples=torch.distributions.Normal(mean,std).sample((n,));samples=torch.clip(samples,low,high)
            if it>0:samples[:keep]=elites[:keep]
            samples[keep]=mean
            constants=[ConstantAgent(samples[:,t],policies[p.idx_policy]) for t,p in enumerate(skeleton)]
            states,_=dynamics.rollout(obs,skeleton,constants,batch_size=n,time_index=True);calls+=n
            values=torch.stack([policies[p.idx_policy].critic.predict(dynamics.decode(states[:,t],primitive=p),samples[:,t]) for t,p in enumerate(skeleton)],-1).clamp(0,1)
            score=torch.exp(torch.log(values).sum(-1));idx=score.topk(count_elite).indices;elites=samples[idx];winner=idx[0]
            if best is None or score[winner]>best['p_success']:
                best=dict(actions=samples[winner].numpy().copy(),states=states[winner].numpy().copy(),p_success=float(score[winner]),values=values[winner].numpy().copy())
            mean=momentum*mean+(1-momentum)*elites.mean(0);std=(momentum*std+(1-momentum)*elites.std(0)).clamp(min=1e-4)
            iterations.append(dict(iteration=it,population=n,elite_indices=idx.tolist(),mean=mean.tolist(),std=std.tolist(),best_probability=best['p_success']))
            visited.append(dict(actions=samples.numpy(),states=states.numpy(),scores=score.numpy(),values=values.numpy()))
            n=max(int(population_decay*n+.5),2*count_elite)
    return dict(**best,iterations=iterations,visited_actions=np.concatenate([v['actions'] for v in visited]),visited_states=np.concatenate([v['states'] for v in visited]),p_visited_success=np.concatenate([v['scores'] for v in visited]),visited_values=np.concatenate([v['values'] for v in visited]),trajectory_queries=calls,skill_model_queries=calls*len(skeleton))
