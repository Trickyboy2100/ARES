"""Contract-active action likelihood for proceed/hold contact control.

Hybrid parameterized policies precede this implementation (H-PPO, IJCAI2019).
This is an experimental executed-action marginalization, not a first invention
of hybrid RL. Hold has no Cartesian parameters; phase0 has no gate decision.
"""
import torch
from torch import nn
from torch.distributions import Normal,Bernoulli
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

class GateDistribution(Distribution):
    def __init__(self,conditional=True,prior=1.):super().__init__();self.conditional=conditional;self.prior=prior
    def proba_distribution_net(self,latent_dim,log_std_init=-1.5):return nn.Linear(latent_dim,3),nn.Parameter(torch.full((2,),log_std_init))
    def proba_distribution(self,means,log_std,phase):
        self.xy=Normal(means[:,:2],torch.ones_like(means[:,:2])*log_std.exp())
        self.gate=Bernoulli(logits=means[:,2]+self.prior);self.phase=phase
        return self
    def log_prob(self,actions):
        b=(actions[:,2]>0).float();continuous=self.xy.log_prob(actions[:,:2]).sum(-1)
        active=torch.where(self.phase,b if self.conditional else torch.ones_like(b),torch.ones_like(b))
        return active*continuous+self.phase.float()*self.gate.log_prob(b)
    def entropy(self):
        weight=torch.where(self.phase,self.gate.probs if self.conditional else torch.ones_like(self.gate.probs),torch.ones_like(self.gate.probs))
        return weight*self.xy.entropy().sum(-1)+self.phase.float()*self.gate.entropy()
    def sample(self):
        b=torch.where(self.phase,self.gate.sample(),torch.ones_like(self.gate.probs))
        return torch.cat((self.xy.sample(),(2*b-1)[:,None]),-1)
    def mode(self):
        b=torch.where(self.phase,(self.gate.probs>=.5).float(),torch.ones_like(self.gate.probs))
        return torch.cat((self.xy.mean,(2*b-1)[:,None]),-1)
    def actions_from_params(self,means,log_std,phase,deterministic=False):
        return self.proba_distribution(means,log_std,phase).get_actions(deterministic)
    def log_prob_from_params(self,means,log_std,phase):
        actions=self.actions_from_params(means,log_std,phase);return actions,self.log_prob(actions)

class ContractGatePolicy(ActorCriticPolicy):
    def __init__(self,*args,gate_conditional=True,gate_prior=1.,**kwargs):
        self.gate_conditional=gate_conditional;self.gate_prior=gate_prior
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_dist=GateDistribution(self.gate_conditional,self.gate_prior)
        self.log_std=nn.Parameter(torch.full((2,),self.log_std_init))
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
    def _get_action_dist_from_latent(self,latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi),self.log_std,self._phase)
    def forward(self,obs,deterministic=False):
        self._phase=obs[:,31]>.5;return super().forward(obs,deterministic)
    def evaluate_actions(self,obs,actions):
        self._phase=obs[:,31]>.5;return super().evaluate_actions(obs,actions)
    def get_distribution(self,obs):
        self._phase=obs[:,31]>.5;return super().get_distribution(obs)
    def _get_constructor_parameters(self):
        data=super()._get_constructor_parameters();data.update(gate_conditional=self.gate_conditional,gate_prior=self.gate_prior);return data
