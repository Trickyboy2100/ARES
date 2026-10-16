"""Synthetic estimator fixtures only; never used as robot training labels."""
import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from contact_credit_v2 import bootstrap_mix,policy_class
class CreditV2Tests(unittest.TestCase):
    def test_rollout_cut_uses_current_bootstrap_without_complete_episode(self):
        r=np.ones((3,1));v=np.zeros_like(r);a=np.zeros_like(r);ret=np.zeros_like(r);ret[-1]=6
        out,targets,audit=bootstrap_mix(a,ret,v,r,np.zeros_like(r,bool),np.zeros_like(r),weight=1)
        np.testing.assert_array_equal(targets[:,0],[8,7,6]);self.assertEqual(audit['samples'],3)
    def test_actual_done_blocks_cross_episode_leakage(self):
        r=np.array([[1.],[2.],[100.] ]);ret=r.copy();z=np.zeros_like(r);ends=np.array([[0],[1],[1]],bool)
        _,target,_=bootstrap_mix(z,ret,z,r,ends,z,weight=1)
        np.testing.assert_array_equal(target[:,0],[3,2,100])
    def test_stage_and_all_have_different_actor_gradients(self):
        import torch
        r=np.array([[0.],[0.],[0.],[10.]]);z=np.zeros_like(r);ret=z.copy();ret[-1]=10
        phase=np.array([[0],[0],[1],[1]]);done=np.array([[0],[0],[0],[1]],bool)
        gradients=[]
        for mode in ['boundary','all']:
            adv,_,_=bootstrap_mix(z,ret,z,r,done,phase,mode=mode)
            theta=torch.tensor(.2,requires_grad=True);features=torch.tensor([1.,2.,3.,4.])
            loss=-(theta*features*torch.tensor(adv[:,0])).mean();loss.backward();gradients.append(theta.grad.item())
        self.assertNotEqual(*gradients)
    def test_real_sb3_optimization_uses_every_current_window(self):
        import gymnasium as gym,torch
        from stable_baselines3.common.callbacks import BaseCallback
        class Fixture(gym.Env):
            observation_space=gym.spaces.Box(-1,1,(56,),np.float32);action_space=gym.spaces.Box(-1,1,(3,),np.float32)
            def reset(self,seed=None,options=None):super().reset(seed=seed);self.t=0;return self.obs(),{}
            def obs(self):
                x=np.zeros(56,np.float32);x[30+int(self.t>=3)]=1;return x
            def step(self,action):self.t+=1;return self.obs(),float(self.t==7),self.t==7,False,{}
        class Record(BaseCallback):
            def _on_step(self):self.model.outcome_flags.append(self.locals['dones'].copy());return True
        models=[]
        for mode in ['all','boundary']:
            model=policy_class()('MlpPolicy',Fixture(),n_steps=4,batch_size=4,n_epochs=1,seed=8)
            model.credit_mode=mode;model.credit_mix=.5;model.learn(16,callback=Record());models.append(model)
        self.assertEqual(len(models[0].credit_audits),4)
        self.assertTrue(all(a['samples']==4 for a in models[0].credit_audits))
        self.assertTrue(all(a['parameter_update_l2']>0 for a in models[0].credit_audits))
        self.assertFalse(torch.equal(models[0].policy.action_net.weight,models[1].policy.action_net.weight))
