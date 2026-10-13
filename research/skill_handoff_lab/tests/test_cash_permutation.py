"""Candidate permutation invariance on artificial tensor fixtures only."""
import unittest
import torch
from research.skill_handoff_lab.cash.model import CashActorCritic

class PermutationTests(unittest.TestCase):
    def test_actor_and_effect_are_equivariant(self):
        torch.manual_seed(19);model=CashActorCritic(4,3,hidden=8,effect_dim=2)
        state=torch.randn(2,4);features=torch.randn(2,5,3)
        mask=torch.tensor([[True,False,True,True,True],[True,True,False,True,True]])
        order=torch.tensor([3,0,4,1,2]);base=model(state,features,mask)
        other=model(state,features[:,order],mask[:,order])
        torch.testing.assert_close(other['logits'],base['logits'][:,order])
        torch.testing.assert_close(other['successor_effect'],base['successor_effect'][:,order])
        torch.testing.assert_close(other['value'],base['value'])

if __name__=='__main__':unittest.main()
