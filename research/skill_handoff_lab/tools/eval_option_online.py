"""Frozen own-policy checkpoints on measured native branches, plus optional actual rollout.
All logged results label their provenance. Direct verification reexecutes full physics.
"""
import argparse,json,hashlib,gzip
from pathlib import Path
import torch
from program_option_improvement import Improved
from eval_option_improvement import panel,evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--direct',action='store_true');p.add_argument('--direct-n',type=int,default=8);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 files=sorted(a.root.glob('online_*/MODEL_*.pt'));assert files;(a.out/'FROZEN_MODELS.json').write_text(json.dumps([dict(path=str(f),sha=hashlib.sha256(f.read_bytes()).hexdigest())for f in files],indent=2));pp=panel(a.panel);scores=[];pred=[]
 rt=None
 if a.direct:
  from program_peg_option_runtime import Runtime
  rt=Runtime(a.out/'physics',successor_mode='split_refresh')
 try:
  for f in files:
   m=Improved.load(f);k=int(f.stem.split('_')[-1]);rows=evaluate(m,pp);entry=dict(method=m.kind,seed=m.seed,checkpoint=k,updates=m.updates,success=sum(r['success']for r in rows),n=len(rows),model=str(f),evaluation='FROZEN_POLICY_ON_REAL_BRANCHES_NOT_ADDITIONAL_PHYSICS')
   if a.direct and k==32:
    for row in rows[:a.direct_n]:
     rr,tt,d=rt.run(row['scene'],m.choose,f'{m.kind}_{m.seed}');ref=json.load(gzip.open(row['source'],'rt'));assert d['initial']==ref['initial']and d['steps']==ref['steps'];row['direct_source']=rr['path'];row['exact100']=True
    entry['evaluation']='LOGGED_PANEL_WITH_DIRECT_PREFIX_SCENES_EXACT100_CHECK';entry['direct_scenes']=min(a.direct_n,len(rows))
   scores.append(entry);pred.append(dict(**entry,rows=rows));(a.out/'SCORES.json').write_text(json.dumps(scores,indent=2));(a.out/'PREDICTIONS.json').write_text(json.dumps(pred,indent=2));print(m.kind,m.seed,k,entry['success'],flush=True)
 finally:
  if rt:rt.close()
if __name__=='__main__':main()
