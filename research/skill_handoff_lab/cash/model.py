"""Small masked actor-critic with an auxiliary successor-effect head."""
from __future__ import annotations
import torch
from torch import nn

class CashActorCritic(nn.Module):
    def __init__(self,state_dim:int,action_dim:int,hidden:int=96,effect_dim:int=8):
        super().__init__()
        self.state=nn.Sequential(nn.Linear(state_dim,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.action=nn.Sequential(nn.Linear(action_dim,hidden),nn.Tanh())
        self.logit=nn.Linear(hidden,1); self.value=nn.Linear(hidden,1); self.effect=nn.Linear(hidden,effect_dim)
    def forward(self,state,action_features,action_mask):
        z=self.state(state); joint=torch.tanh(z[:,None,:]+self.action(action_features))
        logits=self.logit(joint).squeeze(-1).masked_fill(~action_mask,-torch.inf)
        return {"logits":logits,"value":self.value(z).squeeze(-1),
                "successor_effect":self.effect(joint)}
