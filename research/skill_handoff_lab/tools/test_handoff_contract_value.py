"""Information and censoring tests, not physical outcome evidence."""
import unittest
import numpy as np
from handoff_contract_value import feature,records
class ValueTests(unittest.TestCase):
    def test_history_ablation_keeps_live_geometry(self):
        x=np.arange(57,dtype=np.float32);y=feature(x,'contract_no_history')
        np.testing.assert_array_equal(y[:39],x[:39]);np.testing.assert_array_equal(y[48:],x[48:]);self.assertTrue(np.all(y[39:48]==0));self.assertTrue(np.any(x[39:48]!=0))
    def test_unreached_contract_has_masked_continuous_targets(self):
        e={(1,a,c):dict(y=0,boundaries=[],decisions=[dict(stage=0,observation=[0.]*57,action=a)]) for a in range(3) for c in range(3)}
        rows=records(e);self.assertEqual(len(rows[0]),3);self.assertEqual(rows[1],[])
        for r in rows[0]:np.testing.assert_array_equal(r['mask'],[1,0,0,0,0]);self.assertIsNone(r['next'])
if __name__=='__main__':unittest.main(verbosity=2)
