"""Synthetic distribution tests, never used as physical success labels."""
import unittest,sys
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from contact_hybrid_policy import GateDistribution,ContractGatePolicy
class HybridTests(unittest.TestCase):
    def test_hold_has_zero_unused_xy_gradient(self):
        means=torch.tensor([[.2,.3,.1]],requires_grad=True)
        d=GateDistribution().proba_distribution(means,torch.zeros(2),torch.tensor([True]))
        d.log_prob(torch.tensor([[.8,-.7,-1.]])).sum().backward()
        torch.testing.assert_close(means.grad[0,:2],torch.zeros(2));self.assertNotEqual(means.grad[0,2],0)
    def test_joint_ablation_trains_inactive_xy(self):
        m=torch.tensor([[.2,.3,.1]],requires_grad=True)
        d=GateDistribution(False).proba_distribution(m,torch.zeros(2),torch.tensor([True]))
        d.log_prob(torch.tensor([[.8,-.7,-1.]])).sum().backward();self.assertGreater(m.grad[0,:2].norm(),0)
    def test_phase0_gate_is_legally_masked(self):
        m=torch.zeros((8,3),requires_grad=True);d=GateDistribution().proba_distribution(m,torch.zeros(2),torch.zeros(8,dtype=torch.bool))
        action=d.sample();self.assertTrue((action[:,2]==1).all());d.log_prob(action).sum().backward()
        torch.testing.assert_close(m.grad[:,2],torch.zeros(8))
    def test_sb3_online_updates_and_roundtrip(self):
        import gymnasium as gym,tempfile
        from stable_baselines3 import PPO
        class Fixture(gym.Env):
            observation_space=gym.spaces.Box(-1,1,(56,),np.float32);action_space=gym.spaces.Box(-1,1,(3,),np.float32)
            def reset(self,seed=None,options=None):super().reset(seed=seed);self.t=0;return self.obs(),{}
            def obs(self):
                x=np.zeros(56,np.float32);x[30+int(self.t>=2)]=1;return x
            def step(self,a):self.t+=1;return self.obs(),float(a[2]>0),self.t>=4,False,{}
        env=Fixture();m=PPO(ContractGatePolicy,env,n_steps=8,batch_size=8,n_epochs=1,seed=2)
        before=m.policy.action_net.weight.detach().clone();m.learn(16)
        self.assertFalse(torch.equal(before,m.policy.action_net.weight))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'m.zip';m.save(path);loaded=PPO.load(path);obs,_=env.reset()
            np.testing.assert_array_equal(m.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])
