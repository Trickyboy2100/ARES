"""Pure-stdlib regression tests for the offline skill-handoff alignment substrate."""
import json
from pathlib import Path
import tempfile
import unittest

from research.skill_handoff_lab.tools.align_repos import (
    compare_assets, joint_limits, soft_limit_overlap, sha256,
)


URDF = """<robot name="test">
%s
</robot>""" % "\n".join(
    '<joint name="joint%d" type="revolute"><parent link="l%d"/><child link="l%d"/>'
    '<axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>' % (i,i-1,i)
    for i in range(1,7)
)


class OfflineAlignmentContractTest(unittest.TestCase):
    def test_reads_six_joint_axes_and_limits(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"robot.urdf"
            p.write_text(URDF)
            v=joint_limits(p)
            self.assertEqual(len(v),6)
            self.assertEqual(v["joint2"]["axis"],(0.,0.,1.))
            self.assertEqual(v["joint6"]["upper"],2.)

    def test_soft_limit_intersection(self):
        limits={("joint%d"%i):{"lower":-2.0,"upper":2.0} for i in range(1,7)}
        site={"joint_names":["joint%d"%i for i in range(1,7)],
              "lower_rad":[-2.1]*6,"upper_rad":[2.1]*6,
              "soft_limit_margin_rad":.15}
        ok, ranges=soft_limit_overlap(limits,site)
        self.assertTrue(ok)
        self.assertAlmostEqual(ranges["joint1"]["sample_lower_rad"],-1.95)

    def test_reports_bad_assets_instead_of_auto_repair(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"robot.urdf"
            p.write_text("original")
            manifest={"asset_sha256":{"robot.urdf":sha256(p)}}
            self.assertTrue(compare_assets(Path(root),manifest)[0]["match"])
            p.write_text("changed")
            self.assertFalse(compare_assets(Path(root),manifest)[0]["match"])

    def test_missing_mesh_blocks(self):
        with tempfile.TemporaryDirectory() as root:
            entries=compare_assets(Path(root),{"asset_sha256":{"missing.STL":"abc"}})
            self.assertFalse(entries[0]["exists"])
            self.assertFalse(entries[0]["match"])


if __name__=="__main__":
    unittest.main()
