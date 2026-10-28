"""CPU contract checks, explicitly NOT physical benchmark evidence."""
import ast,hashlib,json,unittest
from pathlib import Path
import numpy as np
import torch
from handoff_learn import ActorCritic,Head,transform,initial_model,load_prior,dataset,policy_for
class ContractTests(unittest.TestCase):
    def test_exact_information_masks(self):
        initial=np.arange(57,dtype=np.float32);initial[-1]=0;post=initial+100;post[-1]=1
        next_hidden=transform(initial,'no_next',initial)
        self.assertTrue(np.all(next_hidden[3:9]==0));self.assertTrue(np.all(next_hidden[17:21]==0));self.assertTrue(np.all(next_hidden[45:48]==0))
        posterior=transform(post,'no_posterior',initial);np.testing.assert_array_equal(posterior[:21],initial[:21]);np.testing.assert_array_equal(posterior[21:39],post[21:39]);self.assertTrue(np.all(posterior[39:48]==0))
        committed=transform(post,'no_second',initial);np.testing.assert_array_equal(committed[:-1],initial[:-1]);self.assertEqual(committed[-1],1)
    def test_no_state_teleport_or_virtual_outcome_in_executor(self):
        source=Path(__file__).with_name('handoff_vector.py').read_text();tree=ast.parse(source);calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        forbidden={'set_state','set_state_dict','set_pose','set_qpos','set_qvel'}
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden and not (n.func.attr=='set_base_pose') for n in calls))
        self.assertIn('self.env.step(action)',source);self.assertIn('for step in range(250)',source);self.assertIn('consecutive>=3',source)
    def test_no_second_actor_input_is_committed_snapshot(self):
        torch.manual_seed(4);model=ActorCritic();b={'mean':np.zeros(57),'std':np.ones(57)};pol=policy_for(model,b,'no_second');o=np.zeros(57,np.float32);a,meta=pol(o,0,0);p=np.ones(57,np.float32)*100;p[-1]=1;a2,m2=pol(p,0,1)
        self.assertEqual(a2,meta['committed_second']);self.assertEqual(m2['policy_input'][-1],1);np.testing.assert_array_equal(m2['policy_input'][:-1],o[:-1])
    def test_no_unvisited_second_token(self):
        root=Path(__file__).parents[3].parents[1]/'runs/single_paper_20261028/opportunity64/RESULTS.json'
        if not root.exists():self.skipTest('raw public-physics evidence not available')
        rows=json.loads(root.read_text())['rows']
        for r in rows:
            for ds,bs in zip(r['decisions'],r['boundaries']):
                self.assertIn(len(ds),[1,2])
                if len(ds)==2:self.assertTrue(any(b['stage']==0 and b['valid'] for b in bs))
if __name__=='__main__':unittest.main(verbosity=2)
