"""Analytic unit fixtures, not physical success labels."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
from recontact_minkowski_v8 import recontact_waypoint_exact, segment_hits_envelope


class MinkowskiRecontactTests(unittest.TestCase):
    def test_side_shell_fits_where_circle_does_not(self):
        obj = torch.tensor([[.535, -.514]])
        paddle = obj + torch.tensor([[0., -.040]])
        target, _, _ = recontact_waypoint_exact(paddle, obj, torch.tensor([[0., 1.]]),
            torch.tensor([0]), torch.tensor([.041]), torch.zeros(1), torch.zeros(1))
        self.assertAlmostEqual(float(target[0, 1]), -.560, places=5)

    def test_segment_through_object_rejected(self):
        self.assertTrue(segment_hits_envelope(torch.tensor([[-.06, 0.]]), torch.tensor([[.06, 0.]]),
            torch.zeros(1), torch.zeros(1), [.02, .02], [.015, .02]).item())

    def test_outer_segment_does_not_hit_object(self):
        self.assertFalse(segment_hits_envelope(torch.tensor([[-.06, .06]]), torch.tensor([[.06, .06]]),
            torch.zeros(1), torch.zeros(1), [.02, .02], [.015, .02]).item())

    def test_rotated_orbit_chords_are_checked(self):
        angle = torch.linspace(-3.14, 3.14, 81)
        radius = .07
        start = radius * torch.stack((angle.cos(), angle.sin()), -1)
        obj = torch.zeros_like(start)
        target, next_mode, _ = recontact_waypoint_exact(start, obj, torch.tensor([[0., 1.]]).repeat(81, 1),
            torch.ones(81, dtype=torch.long), torch.full((81,), .041),
            torch.full((81,), .3), torch.full((81,), -.1))
        orbit = next_mode == 1
        # Approach intentionally enters the 2-mm expanded margin, unlike orbit.
        self.assertFalse(segment_hits_envelope(start, target, torch.full((81,), .3),
            torch.full((81,), -.1), [.02, .02], [.015, .02])[orbit].any())


if __name__ == '__main__':
    unittest.main()
