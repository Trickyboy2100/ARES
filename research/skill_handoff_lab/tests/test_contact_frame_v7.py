import unittest
import torch
from research.skill_handoff_lab.tools.contact_task_frame_v7 import to_task_xy, from_task_xy, rear_alignment, task_observation


class FrameTest(unittest.TestCase):
    def test_roundtrip(self):
        p=torch.tensor([[.03,-.04],[.01,.02]])
        phi=torch.tensor([.4,-.6]);n=torch.stack((phi.cos(),phi.sin()),-1)
        torch.testing.assert_close(from_task_xy(to_task_xy(p,n),n),p)

    def test_alignment_equivariance(self):
        phi=torch.tensor([0.,.3]);n=torch.stack((phi.cos(),phi.sin()),-1)
        p=from_task_xy(torch.tensor([[-.05,.01],[-.05,.01]]),n)
        o=torch.zeros_like(p);yes,d=rear_alignment(p,o,p,n)
        self.assertTrue(yes.all())
        torch.testing.assert_close(to_task_xy(d,n),torch.tensor([[0.,-.0012],[0.,-.0012]]),atol=1e-8,rtol=1e-5)

    def test_body_vs_task_action_really_differs(self):
        n=torch.tensor([[0.,1.]])
        actual=from_task_xy(torch.tensor([[.001,0.]]),n)
        torch.testing.assert_close(actual,torch.tensor([[0.,.001]]))

    def test_observation_geometry_dwell_phase_and_vectors(self):
        phi=torch.tensor([.3,-.2]);n=torch.stack((phi.cos(),phi.sin()),-1)
        old=torch.arange(66,dtype=torch.float32).reshape(2,33)/100
        extra=torch.arange(46,dtype=torch.float32).reshape(2,23)/100
        extra[:,:4]=torch.stack(((phi/2).cos(),torch.zeros_like(phi),torch.zeros_like(phi),(phi/2).sin()),-1)
        a,b=task_observation(old,extra,n)
        torch.testing.assert_close(b[:,18:20],torch.tensor([[1.,0.],[1.,0.]]))
        torch.testing.assert_close(b[:,20:],extra[:,20:])
        torch.testing.assert_close(a[:,30:33],old[:,30:33])
        torch.testing.assert_close(a[:,:12],old[:,:12])
        for lo in [12,15,18,21,24,27]:
            torch.testing.assert_close(from_task_xy(a[:,lo:lo+2],n),old[:,lo:lo+2])
            torch.testing.assert_close(a[:,lo+2],old[:,lo+2])
        for lo in [4,7,10]:torch.testing.assert_close(from_task_xy(b[:,lo:lo+2],n),extra[:,lo:lo+2])
        torch.testing.assert_close(b[:,:4],torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]),atol=1e-6,rtol=1e-6)

if __name__=='__main__':unittest.main()
