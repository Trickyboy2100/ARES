"""Isolated synthetic unit fixtures only; never used as training evidence.

Compile actual methods' AST without importing the Isaac application. GPU
physics tests separately verify execution; these assert projection/reward.
"""
import ast,unittest
from pathlib import Path
from types import SimpleNamespace
import torch

SOURCE=Path(__file__).parents[1]/'tools/isaac_lab_smoke.py'
def actual_method(name,args):
    node=next(n for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[node],type_ignores=[]);space={'torch':torch,'a':args}
    exec(compile(ast.fix_missing_locations(module),str(SOURCE),'exec'),space)
    return space[name]

class IndustrialContractTests(unittest.TestCase):
    def fixture(self):
        c=torch.tensor([-.1,-.1,-.1,.1,.1,.1,1.,0.,1.,.055,.007,.05,.001,100.,4.]).repeat(2,1)
        return SimpleNamespace(contract=c,center=torch.zeros(2,3),command=torch.zeros(2,3),goal=torch.ones(2,3))
    def test_mask_applies_to_nominal_and_residual(self):
        for controller in ['nominal','residual','contract','contract_blind','ppo']:
            args=SimpleNamespace(industrial=True,controller=controller,residual_mode='learned',action_audit=False)
            env=self.fixture();actual_method('_pre_physics_step',args)(env,torch.ones(2,3)*3)
            self.assertTrue(torch.equal(env.command[:,1],torch.zeros(2)))
            self.assertLessEqual(float(env.command.abs().max()),.00100001)
            self.assertTrue(torch.equal(env.actions,torch.ones(2,3)))
    def test_workspace_projection(self):
        env=self.fixture();env.command[:,0]=.0998
        args=SimpleNamespace(industrial=True,controller='residual',residual_mode='learned',action_audit=False)
        actual_method('_pre_physics_step',args)(env,torch.ones(2,3))
        self.assertLessEqual(float(env.command[:,0].max()),.10000001)
    def test_zero_residual_is_nominal(self):
        left=self.fixture();right=self.fixture()
        for env,controller,mode in [(left,'nominal','learned'),(right,'residual','zero')]:
            args=SimpleNamespace(industrial=True,controller=controller,residual_mode=mode,action_audit=False)
            actual_method('_pre_physics_step',args)(env,torch.ones(2,3))
        self.assertTrue(torch.equal(left.command,right.command))
    def test_frozen_reward_penalizes_stationary_episode(self):
        env=SimpleNamespace(previous_distance=torch.full((2,),float('nan')),step_dt=1/60,
            actions=torch.zeros(2,3),task_success=torch.zeros(2,dtype=torch.bool),
            task_failed=torch.zeros(2,dtype=torch.bool),task_timeout=torch.zeros(2,dtype=torch.bool),
            episode_return=torch.zeros(2),task_records=[])
        env._task_state=lambda:(torch.zeros(2,3),torch.ones(2,3)*.01,torch.zeros(2),torch.zeros(2),torch.zeros(2))
        fn=actual_method('_get_rewards',SimpleNamespace(industrial=True,reward_revision='v2'))
        self.assertTrue(torch.all(fn(env)<0));self.assertTrue(torch.all(fn(env)<0))
        env.task_timeout[:]=True
        self.assertTrue(torch.all(fn(env)<-10))

if __name__=='__main__':unittest.main()
