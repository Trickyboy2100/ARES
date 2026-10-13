"""Minimal CASH actor-critic update, separated from simulator collection."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch.distributions import Categorical

@dataclass
class CashBatch:
    state:torch.Tensor; action_features:torch.Tensor; action_mask:torch.Tensor
    action:torch.Tensor; old_log_prob:torch.Tensor; advantage:torch.Tensor
    returns:torch.Tensor; successor_effect_target:torch.Tensor

def cash_update(model,optimizer,batch:CashBatch,clip=.2,value_coef=.5,effect_coef=.1,
                entropy_coef=.01):
    out=model(batch.state,batch.action_features,batch.action_mask)
    dist=Categorical(logits=out["logits"]); log_prob=dist.log_prob(batch.action)
    ratio=torch.exp(log_prob-batch.old_log_prob)
    policy=-torch.minimum(ratio*batch.advantage,
        torch.clamp(ratio,1-clip,1+clip)*batch.advantage).mean()
    value=torch.nn.functional.mse_loss(out["value"],batch.returns)
    index=batch.action[:,None,None].expand(-1,1,out["successor_effect"].shape[-1])
    selected=torch.gather(out["successor_effect"],1,index).squeeze(1)
    effect=torch.nn.functional.mse_loss(selected,batch.successor_effect_target)
    loss=policy+value_coef*value+effect_coef*effect-entropy_coef*dist.entropy().mean()
    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return {"loss":float(loss.detach()),"policy_loss":float(policy.detach()),
            "value_loss":float(value.detach()),"effect_loss":float(effect.detach()),
            "entropy":float(dist.entropy().mean().detach())}

def save_checkpoint(path,model,optimizer,metadata):
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "metadata":metadata},path)
