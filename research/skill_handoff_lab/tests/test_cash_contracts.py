import unittest
import torch
from research.skill_handoff_lab.cash.contracts import HandoffAction,HandoffObservation
from research.skill_handoff_lab.cash.env import SkillHandoffEnv
from research.skill_handoff_lab.cash.model import CashActorCritic
from research.skill_handoff_lab.cash.trainer import CashBatch,cash_update

class Backend:
    def reset(self,seed):
        return HandoffObservation("e","GRASP_BRANCH",(0.,),(1.,),(2.,),
          (HandoffAction("a","GRASP_BRANCH",(0.,),True),),"sha256:s","sha256:h")
    def execute(self,observation,action_id): raise AssertionError("not called")

class CashTests(unittest.TestCase):
    def test_mask_exposed(self):
        env=SkillHandoffEnv(Backend()); obs,_=env.reset(seed=3)
        self.assertEqual(obs["action_ids"],("a",)); self.assertTrue(env.action_masks()[0])
    def test_masked_actor_critic_updates(self):
        torch.manual_seed(3); model=CashActorCritic(4,3,hidden=16,effect_dim=2)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        state=torch.randn(3,4); af=torch.randn(3,4,3)
        mask=torch.tensor([[1,1,0,0],[1,0,1,0],[0,1,1,0]],dtype=torch.bool)
        with torch.no_grad():
            logits=model(state,af,mask)["logits"]
            actions=torch.tensor([0,0,1]); old=torch.distributions.Categorical(logits=logits).log_prob(actions)
        before=next(model.parameters()).detach().clone()
        metrics=cash_update(model,opt,CashBatch(state,af,mask,actions,old,
            torch.ones(3),torch.ones(3),torch.zeros(3,2)))
        self.assertFalse(torch.equal(before,next(model.parameters()).detach()))
        self.assertIn("effect_loss",metrics)

if __name__=="__main__": unittest.main()
