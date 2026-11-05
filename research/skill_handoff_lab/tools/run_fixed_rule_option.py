"""Training-selected fixed proposal at its first disagreement; no learned gate."""
import argparse,hashlib
from pathlib import Path
import torch
from program_rule_option import Assets,SingleDeviation
from program_atomic_feedback import AtomicRuntime
from train_option_online import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--proposal',type=int,choices=[0,1],default=1);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
atomic_json(a.out/'MANIFEST.json',dict(start=a.start,count=a.count,proposal=a.proposal,selection='first disagreement; frozen proposal; complete Option then rule',source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),training=False))
assets=Assets();rt=AtomicRuntime(a.out/'physics');rows=[]
try:
 for seed in range(a.start,a.start+a.count):
  pol=SingleDeviation(assets,lambda s,b,c,t:a.proposal if c[a.proposal]!=b else None,lambda:len(rt.env.records))
  row,_,_=rt.run(seed,pol,'fixed_first');row['option']=pol.metadata();rows.append(row);atomic_json(a.out/'RESULTS.json',rows);print(a.out.name,seed,row['success'],flush=True)
finally:rt.close()
