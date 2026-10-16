"""Completed-episode outcome credit in actual Actor advantages, not an unused head.

Prototype event-conditioned GAE/Monte-Carlo mixing. This is an empirical
extension of established eligibility-trace ideas, NOT a claim of inventing
skill values or Monte Carlo returns. Environment rewards remain byte-identical.
"""
import numpy as np

def mix_advantages(advantages,rewards,values,episode_starts,phases,ends,mode='boundary',weight=.5,gamma=1.):
    result=advantages.copy();changed=0;episodes=0
    for env in range(rewards.shape[1]):
        start=None
        for t in range(rewards.shape[0]):
            if episode_starts[t,env]:start=t
            if ends[t,env] and start is not None:
                value=0.;episodes+=1
                for j in range(t,start-1,-1):
                    value=float(rewards[j,env])+gamma*value
                    if mode=='all' or (mode=='boundary' and phases[j,env]==0):
                        result[j,env]=(1-weight)*advantages[j,env]+weight*(value-values[j,env]);changed+=1
                start=None
    return result,{'completed_episodes':episodes,'mixed_actor_samples':changed,
        'max_advantage_change':float(abs(result-advantages).max()),'environment_rewards_modified':False}


def policy_class():
    from stable_baselines3 import PPO
    class BoundaryCreditPPO(PPO):
        def collect_rollouts(self,*args,**kwargs):
            self.outcome_flags=[]
            return super().collect_rollouts(*args,**kwargs)
        def train(self):
            if len(self.outcome_flags)!=self.n_steps:raise RuntimeError('actual episode endings missing')
            buffer=self.rollout_buffer;original=buffer.rewards.copy()
            phase=np.argmax(buffer.observations[:,:,30:32],axis=-1)
            buffer.advantages,audit=mix_advantages(buffer.advantages,buffer.rewards,buffer.values,
                buffer.episode_starts,phase,np.array(self.outcome_flags,dtype=bool),
                mode=self.credit_mode,weight=self.credit_mix,gamma=self.gamma)
            if not np.array_equal(original,buffer.rewards):raise RuntimeError('credit estimator must never alter task truth/rewards')
            if not hasattr(self,'credit_audits'):self.credit_audits=[]
            self.credit_audits.append(audit)
            return super().train()
    return BoundaryCreditPPO
