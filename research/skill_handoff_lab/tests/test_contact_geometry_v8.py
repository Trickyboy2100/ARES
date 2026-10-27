import unittest,math
import torch
from research.skill_handoff_lab.tools.contact_geometry_v8 import contact_ray_clearance,projected_halfwidth

class GeometryTest(unittest.TestCase):
    def test_axis_aligned(self):
        n=torch.tensor([[1.,0.]])
        d=contact_ray_clearance(n,torch.tensor([0.]),torch.tensor([0.]),[.02,.02],[.015,.015])
        self.assertAlmostEqual(d.item(),.035,7)
    def test_rotated_cube_true_envelope(self):
        n=torch.tensor([[1.,0.]])
        d=contact_ray_clearance(n,torch.tensor([math.pi/4]),torch.tensor([0.]),[.02,.02],[.015,.015])
        self.assertAlmostEqual(d.item(),.02*math.sqrt(2)+.015,7)
    def test_rotated_both_and_ray(self):
        phi=torch.tensor([.3]);n=torch.stack((phi.cos(),phi.sin()),-1)
        d=contact_ray_clearance(n,phi,phi,[.02,.02],[.015,.015],torch.tensor([.008]))
        self.assertAlmostEqual(d.item(),.035,6)
        w=projected_halfwidth(phi,n,[.02,.02]);self.assertAlmostEqual(w.item(),.02,6)
    def test_lateral_contact_on_rotated_polygon(self):
        n=torch.tensor([[1.,0.]])
        centered=contact_ray_clearance(n,torch.tensor([.3]),torch.tensor([-.2]),[.02,.02],[.015,.015])
        shifted=contact_ray_clearance(n,torch.tensor([.3]),torch.tensor([-.2]),[.02,.02],[.015,.015],torch.tensor([.025]))
        self.assertTrue(torch.isfinite(shifted).all());self.assertNotEqual(centered.item(),shifted.item())
