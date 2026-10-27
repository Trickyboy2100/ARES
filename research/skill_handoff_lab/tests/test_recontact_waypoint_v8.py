"""Analytic geometry fixtures only; not physical experiment evidence."""
import importlib.util
from pathlib import Path
import unittest
import torch

source = Path(__file__).parents[1] / 'tools' / 'recontact_waypoint_v8.py'
spec = importlib.util.spec_from_file_location('recontact_waypoint', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RecontactTests(unittest.TestCase):
    def test_backoff_does_not_cut_across_object(self):
        paddle = torch.tensor([[-.035, 0.]])
        target, mode, ready = module.recontact_waypoint(paddle, torch.zeros_like(paddle),
            torch.tensor([[0., 1.]]), torch.tensor([0]), torch.tensor([.035]))
        self.assertLess(target[0, 0], paddle[0, 0])
        self.assertAlmostEqual(float(target[0, 1]), 0.)
        self.assertEqual(mode.item(), 0)
        self.assertFalse(ready.item())

    def test_orbit_chord_clears_circumscribed_envelope(self):
        ring = (.02 ** 2 + .02 ** 2) ** .5 + (.015 ** 2 + .02 ** 2) ** .5 + .006
        paddle = torch.tensor([[-ring, 0.]])
        target, mode, ready = module.recontact_waypoint(paddle, torch.zeros_like(paddle),
            torch.tensor([[0., 1.]]), torch.tensor([1]), torch.tensor([.035]))
        points = paddle + torch.linspace(0, 1, 101)[:, None] * (target - paddle)
        self.assertGreater(float(points.norm(dim=-1).min()), ring - .006)
        self.assertEqual(mode.item(), 1)
        self.assertFalse(ready.item())

    def test_ready_only_after_real_reference_reached(self):
        paddle = torch.tensor([[0., -.035]])
        target, mode, ready = module.recontact_waypoint(paddle, torch.zeros_like(paddle),
            torch.tensor([[0., 1.]]), torch.tensor([2]), torch.tensor([.035]))
        self.assertTrue(ready.item())
        self.assertEqual(mode.item(), 3)
        torch.testing.assert_close(paddle, target)

    def test_degenerate_direction_is_finite(self):
        value = torch.zeros((1, 2))
        target, _, ready = module.recontact_waypoint(value, value, value, torch.tensor([0]), torch.tensor([.035]))
        self.assertTrue(torch.isfinite(target).all())
        self.assertFalse(ready.item())


if __name__ == '__main__':
    unittest.main()
