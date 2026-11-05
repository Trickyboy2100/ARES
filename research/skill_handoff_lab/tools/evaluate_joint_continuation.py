"""Frozen native evaluation of joint continuation learner; no updates."""
import argparse
from pathlib import Path
import torch
from train_joint_continuation import Learner,Joint,Assets,AtomicRuntime,atomic_json,sha
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=24);p.add_argument('--front',type=int);a=p.parse_args();torch.set_num_threads(2)
a.out.mkdir(parents=True,exist_ok=False);assets=Assets();m=Learner(assets,31901);m.load(torch.load(a.model,weights_only=False))
atomic_json(a.out/'MANIFEST.json',dict(model=str(a.model),sha=sha(a.model),start=a.start,count=a.count,front_override=a.front,training=False,source=sha(__file__)))
rt=AtomicRuntime(a.out/'physics');rows=[]
try:
 for seed in range(a.start,a.start+a.count):
  pol=Joint(assets,lambda s:a.front if a.front is not None and s[44]<.5 else m.choose(s),lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,'frozen');row['events']=pol.events;rows.append(row);atomic_json(a.out/'RESULTS.json',rows);print(a.out.name,seed,row['success'],flush=True)
finally:rt.close()
