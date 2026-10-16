"""Synthetic algorithm fixtures, NOT robot success labels or training data."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from boundary_credit import mix_advantages

class BoundaryCreditTests(unittest.TestCase):
    def test_real_later_reward_reaches_prior_actor_without_changing_rewards(self):
        reward=np.array([[0.],[0.],[0.],[10.]])
        old=np.zeros_like(reward);start=np.array([[1],[0],[0],[0]],bool)
        phase=np.array([[0],[0],[1],[1]]);end=np.array([[0],[0],[0],[1]],bool)
        adv,info=mix_advantages(old,reward,old,start,phase,end,weight=.5)
        np.testing.assert_array_equal(adv[:,0],[5,5,0,0]);np.testing.assert_array_equal(reward[:,0],[0,0,0,10])
        self.assertEqual(info['mixed_actor_samples'],2)
    def test_partial_episode_is_not_given_invented_return(self):
        x=np.ones((4,1));starts=np.zeros((4,1),bool);end=starts.copy();end[-1]=True
        out,info=mix_advantages(x,x,x,starts,np.zeros_like(x),end)
        np.testing.assert_array_equal(out,x);self.assertEqual(info['completed_episodes'],0)
    def test_zero_mix_is_exact_standard_ablation(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(4,1));s=np.zeros((4,1),bool);s[0]=True;e=s[::-1].copy()
        out,_=mix_advantages(x,x,x,s,np.zeros_like(x),e,weight=0.)
        np.testing.assert_array_equal(out,x)

    def test_sb3_actual_actor_optimization_uses_mixed_advantages(self):
        import gymnasium as gym
        import torch
        from stable_baselines3.common.callbacks import BaseCallback
        from boundary_credit import policy_class
        class Fixture(gym.Env):
            observation_space=gym.spaces.Box(-1,1,(33,),np.float32)
            action_space=gym.spaces.Box(-1,1,(2,),np.float32)
            def reset(self,seed=None,options=None):
                super().reset(seed=seed);self.t=0;return self.obs(),{}
            def obs(self):
                v=np.zeros(33,np.float32);v[30+int(self.t>=2)]=1;return v
            def step(self,action):
                self.t+=1;return self.obs(),float(self.t==4),self.t==4,False,{}
        class Endings(BaseCallback):
            def _on_step(self):self.model.outcome_flags.append(self.locals['dones'].copy());return True
        model=policy_class()('MlpPolicy',Fixture(),n_steps=8,batch_size=8,n_epochs=1,seed=9)
        model.credit_mode='boundary';model.credit_mix=.5
        before={k:v.clone() for k,v in model.policy.state_dict().items()}
        model.learn(16,callback=Endings())
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.policy.state_dict().items()))
        self.assertTrue(all(x['mixed_actor_samples']>0 for x in model.credit_audits))
        self.assertTrue(all(not x['environment_rewards_modified'] for x in model.credit_audits))
