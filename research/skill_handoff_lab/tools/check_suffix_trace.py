"""Analytic return counterexamples and exact optimizer/RNG resume."""
import copy,json
from pathlib import Path
import numpy as np,torch
from suffix_trace_learner import Learner,backup
from program_rule_option import Assets

def main():
 torch.set_num_threads(2)
 # Terminal must suppress bootstrap; coefficients belong to successor actions.
 assert backup([1,2],[3,99],[7,0],[0,1],[False,True])==[6.,2.]
 assert backup([1,2],[3,99],[7,0],[0,0],[False,True])==[4.,2.]
 assert backup([1],[99],[7],[1],[True])==[1.]
 # Full on-policy lambda=1 collapses to MC when V(next)=Q(next).
 assert np.allclose(backup([.1,.2,1],[.8,.7,0],[.9,.8,.7],[1,1,1],[False,False,True]),[1.3,1.2,1.],atol=1e-12)
 root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004')
 prior=torch.load(root/'action_frontier_20261107/sac_32202/CURRENT.pt',weights_only=False);assets=Assets();m=Learner(assets,32401,'trace');m.load(prior['model'])
 rows=copy.deepcopy(prior['data'][:16]);yt,_=m.targets(rows,list(range(16)));m.mode='sac';ys,_=m.targets(rows,list(range(16)));assert torch.equal(yt,ys)
 # Exercise real freshly collected chains, no invented behavior probabilities.
 run=root/'suffix_trace_20261108/trace_32401';current=torch.load(run/'CURRENT.pt',weights_only=False)
 assert any('mu'in r for r in current['data']),'Need one completed own-policy episode'
 for mode in ['sac','trace','lagged']:
  a=Learner(assets,32401,mode);a.load(current['model']);a.mode=mode;a.refresh(1);a.train(current['data'],2,1);saved=copy.deepcopy(a.state())
  b=Learner(assets,999,mode);b.load(saved);b.mode=mode
  a.train(current['data'],2,1);b.train(current['data'],2,1)
  for name in ['actor','q','target','reference']:
   assert all(torch.equal(x,y)for x,y in zip(getattr(a,name).parameters(),getattr(b,name).parameters())),name
 print(json.dumps({'analytic_backup':'PASS','unknown_mu_one_step':'PASS','exact_resume_three_modes':'PASS'}))
if __name__=='__main__':main()
