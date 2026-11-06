"""New natural training-design frontier, native resets only, shared warm suffix."""
import argparse,json,torch,hashlib
from pathlib import Path
from program_full_option import FullJoint,Warm,legal
from program_rule_option import Assets,prefix_hash
from program_atomic_feedback import AtomicRuntime
from train_option_online import atomic_json,atomic_torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=8);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
assets=Assets();warm=Warm(assets);rt=AtomicRuntime(a.out/'physics');rows=[];saved=[]
atomic_json(a.out/'MANIFEST.json',dict(source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scenes=list(range(a.start,a.start+a.count)),purpose='training-design only; not evaluation',native_horizon=100,trigger='first reachable rear boundary; no future labels',continuation='common recurrent warm policy',actions='12 physical options with legal mask; baseline plus fixed three grasps',prefix='complete native reexecution, no mid-state reset'))
try:
 for seed in range(a.start,a.start+a.count):
  reference=None
  for label,forced,grasp in [('warm',None,None)]+[(f'a{k}',k,None)for k in range(12)]+[(f'g{k}',None,k)for k in range(3)]:
   used=[False];entry=[None]
   def actor(s):
    if s[44]<.5:return warm.choose(s)if grasp is None else grasp
    if forced is not None and not used[0]:
     used[0]=True;sel=forced+1
     if sel==1 and not legal(s)[1]:sel=0
     entry[0]=dict(step=len(rt.env.records),obs=s.tolist(),requested=forced,legal=bool(legal(s)[sel]),proposals=assets.proposals(s))
     if legal(s)[sel]:return sel
    return warm.choose(s)
   pol=FullJoint(assets,actor,lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,label);ts=pol.transitions(ts)
   if label=='warm':reference=d
   prefix=None
   if entry[0] is not None:
    n=entry[0]['step'];prefix=prefix_hash(reference,n)==prefix_hash(d,n);assert prefix,'prefix changed before selected first intervention'
   row.update(label=label,grasp=pol.events[0]['a'],entry=entry[0],prefix_equal=prefix,events=pol.events);rows.append(row);saved.append(dict(row=row,ts=ts));atomic_json(a.out/'RESULTS.json',rows);atomic_torch(a.out/'REPLAY.pt',saved);print(seed,label,row['success'],entry[0]['step']if entry[0]else None,flush=True)
finally:rt.close()
