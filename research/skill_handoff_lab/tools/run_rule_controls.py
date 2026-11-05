import argparse,json,torch
from pathlib import Path
from program_rule_option import Assets,SingleDeviation,rule
from program_atomic_feedback import AtomicRuntime
from peg_atomic_macro import AtomicMacro
from train_option_online import atomic_json
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,required=True);p.add_argument('--front',type=int,choices=[0,1,2]);p.add_argument('--mode',choices=['rule','macro0','macro1','qgate'],default='rule');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);assets=Assets();rt=AtomicRuntime(a.out/'physics');rows=[]
if a.front is not None:assets.front=lambda s:a.front
try:
 for seed in range(a.start,a.start+a.count):
  if a.mode.startswith('macro'):pol=AtomicMacro(lambda s:assets.front(s)if s[44]<.5 else assets.models[int(a.mode[-1])].choose(s),assets.reference)
  else:
   def selector(s,ab,cs,step):
    if a.mode=='rule':return None
    scores=[]
    with torch.no_grad():
     for j,m in enumerate(assets.models):
      x=m.x([s]);q=torch.stack([v(x)for v in m.q]).mean(0)[0];scores.append(float(q[cs[j]]-q[ab])if cs[j]!=ab else -1e9)
    k=max(range(2),key=lambda j:scores[j]);return k if scores[k]>0 else None
   pol=SingleDeviation(assets,selector,lambda:len(rt.env.records))
  row,ts,d=rt.run(seed,pol,a.mode);row['option']=pol.metadata()if hasattr(pol,'metadata')else dict(events=pol.events);rows.append(row);atomic_json(a.out/'RESULTS.json',rows);print(a.mode,seed,row['success'],flush=True)
finally:rt.close()
