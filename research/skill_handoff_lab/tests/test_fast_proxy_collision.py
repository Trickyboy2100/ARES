"""Synthetic collision math fixtures, isolated from research outcomes."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fast_proxy_collision import global_clearances,signed,gpu_per_sample_clearances,rotation


class ExactCollisionTests(unittest.TestCase):
    def test_kd_matches_full_signed_distance(self):
        rng=np.random.default_rng(17);p=rng.normal(size=(5,41,3));r=rng.uniform(.01,.1,41)
        boxes={str(i):{'pose':[*rng.normal(size=3),1,0,0,0],'dims':rng.uniform(.05,2,3).tolist()} for i in range(15)}
        actual,_=global_clearances(p,r,boxes)
        for name,box in boxes.items():
            expected=float(signed(p.reshape(-1,3),np.tile(r,5),box).min())
            self.assertAlmostEqual(actual[name],expected,places=12)
    def test_batched_torch_formula_matches_cpu(self):
        import torch
        rng=np.random.default_rng(18);p=rng.normal(size=(3,11,4));p[:,:,3]=rng.uniform(.01,.1,(3,11))
        boxes={'a':{'pose':[.2,.3,-.5,.5,.5,.5,.5],'dims':[.8,.3,.5]}}
        actual=gpu_per_sample_clearances(torch.tensor(p),boxes)['a']
        expected=[signed(row[:,:3],row[:,3],boxes['a']).min() for row in p]
        np.testing.assert_allclose(actual,expected,atol=1e-12)


if __name__=='__main__':unittest.main()
