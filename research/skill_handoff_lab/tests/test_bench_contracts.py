import unittest
from research.skill_handoff_lab.bench.contracts import CONTRACT_REVISION,Scenario
class ContractTests(unittest.TestCase):
    def test_revision_enforced(self):
        with self.assertRaises(ValueError): Scenario("wrong","x",1,"TRAY_TO_GROOVE","BODY","right",(0.,0.,0.),(0.,0.,0.),0.,(),{})
    def test_digest_stable(self):
        row=Scenario(CONTRACT_REVISION,"x",1,"TRAY_TO_GROOVE","BODY","right",(0.,0.,0.),(0.,0.,0.),0.,(),{})
        self.assertEqual(row.to_dict()["record_digest"],row.to_dict()["record_digest"])
if __name__=="__main__": unittest.main()
