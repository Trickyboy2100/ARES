"""Frozen full native comparison; all permissions shared, no updates."""
import argparse,torch,json
from pathlib import Path
from program_full_option import FullJoint,Warm
from program_rule_option import Assets,rule
from program_atomic_feedback import AtomicRuntime
from train_full_option import Learner
from train_option_online import atomic_json
from train_joint_continuation import sha
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--mode',choices=['warm','rule','learned'],required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=24);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False);assets=Assets();warm=Warm(assets);m=None
if a.model:
 d=torch.load(a.model,weights_only=False);m=Learner(assets,32201,d['arm']);m.load(d)
assert (a.mode=='learned')==(m is not None)
atomic_json(a.out/'MANIFEST.json',dict(mode=a.mode,model=str(a.model),model_sha=sha(a.model)if a.model else None,source=sha(__file__),seeds=list(range(a.start,a.start+a.count)),native100=True,updates=0))
rt=AtomicRuntime(a.out/'physics');rows=[]
try:
 for seed in range(a.start,a.start+a.count):
  actor=m.choose if m else (lambda s:assets.front(s)if s[44]<.5 else 0)if a.mode=='rule'else warm.choose
  pol=FullJoint(assets,actor,lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,'frozen');row['events']=pol.events;rows.append(row);atomic_json(a.out/'RESULTS.json',rows);print(a.mode,seed,row['success'],flush=True)
finally:rt.close()
