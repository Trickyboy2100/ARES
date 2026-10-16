"""On-policy GAE / bootstrapped lambda=1 mixture; NOT a novel MC algorithm.

Every current-rollout sample is used once. No stale policy episode cache.
At a rollout boundary, lambda=1 targets bootstrap the *current* value; actual
finite-horizon terminations do not bootstrap. Both critic and actor use the
same mixed target. This is explicitly not a full-episode Monte Carlo return.
"""
import numpy as np

def bootstrap_mix(advantages,returns,values,rewards,ends,phases,mode='all',weight=.5,gamma=1.):
    targets=returns.copy()
    # The final GAE return is exactly r + gamma V(next) (or r at termination).
    boot=returns[-1].copy()
    for t in range(len(rewards)-1,-1,-1):
        if t<len(rewards)-1:boot=rewards[t]+gamma*(~ends[t])*boot
        targets[t]=boot
    mask=np.ones_like(ends) if mode=='all' else phases==0
    mixed=np.where(mask,(1-weight)*advantages+weight*(targets-values),advantages)
    return mixed,mixed+values,{'samples':int(mask.sum()),'total_samples':int(mask.size),
        'phase0_samples':int(((phases==0)&mask).sum()),'phase1_samples':int(((phases==1)&mask).sum()),
        'max_advantage_change':float(np.max(abs(mixed-advantages))),
        'rewards_modified':False,'bootstrap':'current rollout value, lambda=1; no stale samples'}

def policy_class():
    from stable_baselines3 import PPO
    class BootstrapMixPPO(PPO):
        def collect_rollouts(self,*args,**kwargs):
            self.outcome_flags=[]
            return super().collect_rollouts(*args,**kwargs)
        def train(self):
            import torch
            b=self.rollout_buffer
            if len(self.outcome_flags)!=self.n_steps:raise RuntimeError('missing current-rollout endings')
            phases=np.argmax(b.observations[:,:,30:32],axis=-1)
            before=[p.detach().clone() for p in self.policy.parameters()]
            old_rewards=b.rewards.copy();adv_before=b.advantages.copy()
            b.advantages,b.returns,audit=bootstrap_mix(b.advantages,b.returns,b.values,b.rewards,
                np.array(self.outcome_flags,bool),phases,self.credit_mode,self.credit_mix,self.gamma)
            if not np.array_equal(old_rewards,b.rewards):raise RuntimeError('environment truth changed')
            audit.update(advantage_before_std=float(adv_before.std()),advantage_after_std=float(b.advantages.std()),
                phase0_modified=int(((abs(b.advantages-adv_before)>1e-9)&(phases==0)).sum()),
                phase1_modified=int(((abs(b.advantages-adv_before)>1e-9)&(phases==1)).sum()))
            super().train()
            audit['parameter_update_l2']=float(torch.sqrt(sum(((p-q)**2).sum() for p,q in zip(self.policy.parameters(),before))))
            actor=list(self.policy.mlp_extractor.policy_net.parameters())+list(self.policy.action_net.parameters())
            audit['last_minibatch_actor_gradient_l2']=float(torch.sqrt(sum((p.grad**2).sum() for p in actor if p.grad is not None)))
            audit['policy_gradient_loss']=float(self.logger.name_to_value['train/policy_gradient_loss'])
            self.credit_audits=getattr(self,'credit_audits',[])+[audit]
    return BootstrapMixPPO
