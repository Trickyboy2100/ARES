"""Frozen complete native evaluation of a trained single-deviation policy."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from train_rule_gate import Gate
from program_rule_option import Assets, SingleDeviation, gate_features
from program_atomic_feedback import AtomicRuntime
from train_option_online import atomic_json

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=24);p.add_argument('--sample',action='store_true');p.add_argument('--front',type=int,choices=[0,1,2]);a=p.parse_args();torch.set_num_threads(2)
d=torch.load(a.model,weights_only=False);w=d['model'];model=Gate(w['mu'].numpy(),w['sd'].numpy(),d['manifest']['seed']);model.load_state_dict(w);model.eval()
a.out.mkdir(parents=True,exist_ok=False)
atomic_json(a.out/'MANIFEST.json',dict(model=str(a.model),model_sha=hashlib.sha256(a.model.read_bytes()).hexdigest(),start=a.start,count=a.count,selection='sample exact training mixture'if a.sample else 'argmax',source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training=False,native100=True,front_override=a.front))
assets=Assets();rt=AtomicRuntime(a.out/'physics');rows=[]
if a.front is not None:assets.front=lambda s:a.front
try:
    for seed in range(a.start,a.start+a.count):
        rng=np.random.default_rng(seed+71000)
        def selector(s,ab,cs,step):
            if cs[0]==ab and cs[1]==ab:return None
            x=np.stack([gate_features(s,ab,c)for c in cs]);mask=[True,cs[0]!=ab,cs[1]!=ab]
            with torch.no_grad():pr=model.probs(torch.tensor(x[None]),torch.tensor([mask]),a.sample)[0].numpy()
            k=int(rng.choice(3,p=pr/pr.sum()))if a.sample else int(pr.argmax())
            return k-1 if k else None
        pol=SingleDeviation(assets,selector,lambda:len(rt.env.records));row,_,_=rt.run(seed,pol,'frozen');row['option']=pol.metadata();rows.append(row);atomic_json(a.out/'RESULTS.json',rows)
        print(a.out.name,seed,row['success'],row['option']['entry']and row['option']['entry']['step'],flush=True)
finally:rt.close()
