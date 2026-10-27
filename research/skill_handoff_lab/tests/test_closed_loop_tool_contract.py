import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.tools.validate_closed_loop_physics import orientation_errors


class ToolContractTests(unittest.TestCase):
    def test_world_yaw_preserves_level_with_nonidentity_tool_frame(self):
        reference = Rotation.from_euler('xyz', [.4, -.7, 1.2]).as_matrix()
        actual = Rotation.from_euler('z', .14).as_matrix() @ reference
        tilt, yaw, full = orientation_errors(actual, reference)
        self.assertAlmostEqual(tilt, 0., places=7)
        self.assertAlmostEqual(yaw, .14, places=7)
        self.assertAlmostEqual(full, .14, places=7)

    def test_roll_violation_is_not_hidden_by_yaw_permission(self):
        actual = Rotation.from_euler('xyz', [.08, 0, .12]).as_matrix()
        tilt, yaw, full = orientation_errors(actual, np.eye(3))
        self.assertGreater(tilt, np.deg2rad(3.5))
        self.assertLess(yaw, np.deg2rad(8))
        self.assertGreater(full, tilt)

    def test_yaw_sign_and_wrap(self):
        for angle in [.14, -.14, 2*np.pi-.14]:
            _, yaw, _ = orientation_errors(Rotation.from_euler('z', angle).as_matrix(), np.eye(3))
            self.assertAlmostEqual(yaw, .14, places=7)


if __name__ == '__main__':
    unittest.main()
