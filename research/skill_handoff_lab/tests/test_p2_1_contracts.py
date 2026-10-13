import unittest
from research.skill_handoff_lab.bench.contracts import digest
from research.skill_handoff_lab.bench.p2_1_contracts import P2_1_REVISION,ScenarioV2
from research.skill_handoff_lab.bench.oracle_adapter import qualification_status

class P21Tests(unittest.TestCase):
    def test_scene_revision_rejects_stale_geometry(self):
        obj={"object_id":"x","revision":"sha256:x"}; obstacles=()
        core={"scenario_id":"s","seed":1,"family":"TRAY_TO_GROOVE",
              "obstacles":obstacles,"rear":(0.,0.,0.),"preplace":(0.,0.,0.),"object":obj}
        row=ScenarioV2(P2_1_REVISION,"s",1,"TRAY_TO_GROOVE","TEST",
            (0.,0.,0.),(0.,0.,0.),(0.,0.,0.),(0.,0.,0.),0.,obstacles,obj,{},digest(core))
        self.assertTrue(row.scene_revision.startswith("sha256:"))
        with self.assertRaises(ValueError):
            ScenarioV2(P2_1_REVISION,"s",1,"TRAY_TO_GROOVE","TEST",
                (0.,0.,0.),(0.,0.,0.),(0.,0.,0.),(0.,0.,0.),0.,obstacles,obj,{},"bad")
    def test_oracle_is_execution_blocked_without_gpu(self):
        value=qualification_status()
        self.assertFalse(value["execution_allowed"])
        self.assertFalse(value["curobo_validated_this_round"])

if __name__=="__main__": unittest.main()
