"""Offline assertions on frozen audit evidence, not physical safety tests."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results/real_aligned_20261014_v1'


class FrozenAuditTests(unittest.TestCase):
    def read(self, name):
        return json.loads((ROOT / name).read_text())

    def test_model_fk_and_mount_disagreement(self):
        f = self.read('FRAME_ALIGNMENT.json')
        self.assertEqual(len(f['samples']), 28)
        self.assertLess(f['usd_urdf_max_position_error_m'], 1e-6)
        self.assertFalse(f['single_global_transform_consistent'])
        self.assertAlmostEqual(f['global_candidate_disagreement']['rotation_deg'], 120, places=3)
        for side in ['left', 'right']:
            self.assertIsNone(f['arms'][side]['october_q_current'])

    def test_geometry_not_claimed_field_verified(self):
        g = self.read('TRAY_GEOMETRY.json')
        self.assertFalse(g['same_field_object_verified'])
        self.assertIsNone(g['october_attached_object'])
        for actual, expected in zip(g['world_bounds']['dimensions_m'], [.0995, .1795, .03]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertLess(g['legacy_marker_separation_m'], .003)
        ear = g['markers']['/World/Tray/GraspFrames/YMinusEar']
        self.assertGreater(ear['offset_from_mesh_bbox_center_m'][2], .14)

    def test_static_capture_not_physics_or_october_reconstruction(self):
        m = self.read('isaac51_static_v2/capture_manifest.json')
        p = self.read('isaac51_static_v2/process_manifest.json')
        self.assertEqual(m['physics_steps'], 0)
        self.assertFalse(m['physics_task_completed'])
        self.assertFalse(m['october_scene_reconstructed'])
        self.assertEqual(p['returncode'], 0)
        self.assertFalse(p['timeout'])


if __name__ == '__main__':
    unittest.main()
