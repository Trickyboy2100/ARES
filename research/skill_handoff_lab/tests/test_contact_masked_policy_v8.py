import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
import torch
from gymnasium.spaces import Box
from contact_masked_policy_v8 import ExecutionActivePolicy

def test_inactive_scores_and_gradients():
    policy=ExecutionActivePolicy(Box(-np.inf,np.inf,(14,),np.float32),Box(-1,1,(5,),np.float32),lambda _:3e-4,net_arch=[16,16])
    obs=torch.zeros((4,14));obs[:,-5:]=torch.tensor([1,1,0,0,0])
    action,value,logprob=policy(obs);assert torch.isfinite(logprob).all()
    other=action.detach().clone();other[:,2:]=300
    assert torch.equal(policy.evaluate_actions(obs,other)[1],logprob)
    assert torch.equal(policy.get_distribution(obs).log_prob(other),logprob)
    (-logprob.sum()).backward()
    assert torch.equal(policy.action_net.weight.grad[2:],torch.zeros_like(policy.action_net.weight.grad[2:]))
    assert torch.equal(policy.log_std.grad[2:],torch.zeros_like(policy.log_std.grad[2:]))

def test_all_inactive_unit_density_and_frozen_mask():
    policy=ExecutionActivePolicy(Box(-np.inf,np.inf,(14,),np.float32),Box(-1,1,(5,),np.float32),lambda _:3e-4)
    obs=torch.zeros((3,14));actions,values,lp=policy(obs)
    assert torch.equal(lp,torch.zeros_like(lp))
    _,newlp,entropy=policy.evaluate_actions(obs,actions)
    assert torch.equal(newlp,lp) and torch.equal(entropy,torch.zeros_like(entropy))
