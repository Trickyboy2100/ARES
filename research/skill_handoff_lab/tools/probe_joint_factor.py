"""Selected training mechanism: separate changed grasp from rear correction.
Original native initial state, no simulation-state restore. No population claim.
"""
import argparse,json,torch,gzip
from pathlib import Path
from train_execution_joint import Assets,Learner,Joint,AtomicRuntime,atomic_json
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);assets=Assets();m=Learner(assets,32001,'flat');m.load(torch.load(a.root/'identity_flat_32001/INITIAL.pt',weights_only=False));rt=AtomicRuntime(a.out/'physics');rows=[]
class Proxy:
 def __init__(self):self.override=False
 def proposals(self,s):return [2,assets.proposals(s)[1]]if self.override else assets.proposals(s)
 def reference(self,s):return assets.reference(s)
try:
 for grasp,correct in [(1,False),(0,True),(1,True)]:
  proxy=Proxy();used=[False]
  def actor(s):
   proxy.override=False
   if s[44]<.5:return grasp
   if correct and not used[0]:used[0]=True;proxy.override=True;return 1
   return m.choose(s)
  pol=Joint(proxy,actor,lambda:len(rt.env.records));row,ts,d=rt.run(764712,pol,f'g{grasp}_correct{int(correct)}');row.update(grasp=grasp,force_first_rear_local_y_plus=correct,events=pol.events);rows.append(row);atomic_json(a.out/'RESULTS.json',rows);print(grasp,correct,row['success'],flush=True)
finally:rt.close()
