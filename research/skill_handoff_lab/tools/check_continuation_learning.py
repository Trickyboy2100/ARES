import sys,copy,torch
sys.path.insert(0,str(__import__('pathlib').Path(__file__).parent))
from continuation_learner import ContinuationLearner
from continuation_data import build
from program_rule_option import Assets
from pathlib import Path
torch.set_num_threads(2);assets=Assets();r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/continuation_retention_20261108');m=ContinuationLearner(assets,32301,'retention');shared=build([r/'design_a',r/'design_b'],m,scope='full_reference');data=m.decorate(shared['data'],0);m.set_evidence(shared['pairs'],shared['anchors'],shared['reference_sha']);print('data',len(data),'non_ties',len(m.pairs),'anchors',len(m.anchors));print(m.train(data,2,0));saved=copy.deepcopy(m.state());result=m.train(data,2,0);end=copy.deepcopy(m.state());n=ContinuationLearner(assets,32301,'retention');n.load(saved);n.set_evidence(shared['pairs'],shared['anchors'],shared['reference_sha']);n.train(data,2,0);assert all(torch.equal(v,n.actor.state_dict()[k])for k,v in end['actor'].items());assert all(torch.equal(v,n.q.state_dict()[k])for k,v in end['q'].items());print('resume exact, nonzero physical pair gradients exercised')

a=ContinuationLearner(assets,32301,'sac');b=ContinuationLearner(assets,32302,'sac');assert not torch.equal(a.actor[0].weight,b.actor[0].weight)
c=ContinuationLearner(assets,32301,'ranking');assert torch.equal(a.q[0][0].weight,c.q[0][0].weight)
assert all(p['delta']!=0 for p in n.pairs)
print('independent network initialization; equal cross-arm initial critics; ties excluded from preference but retained in replay')
