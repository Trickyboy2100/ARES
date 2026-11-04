"""Full native-task accounting and matched-prefix retention, never oracle selection."""
import argparse,json,gzip,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
base=json.loads((a.root/'baseline/EVAL_0.json').read_text());b={r['seed']:r for r in base};result={}
for directory in sorted(a.root.iterdir()):
 if not directory.is_dir():continue
 evals={}
 for f in sorted(directory.glob('EVAL_*.json'),key=lambda p:int(p.stem.split('_')[1])):
  rows=json.loads(f.read_text());rescue=[r['seed'] for r in rows if r['success'] and not b[r['seed']]['success']];regress=[r['seed'] for r in rows if not r['success'] and b[r['seed']]['success']]
  acts=collections.Counter(d['id'] for r in rows for d in r['decisions'] if d['stage']==1)
  evals[f.stem]=dict(success=sum(r['success'] for r in rows),n=len(rows),rescues=rescue,regressions=regress,successor_actions=dict(acts),replans=sum(d.get('replanned',False) for r in rows for d in r['decisions']))
 if not evals:continue
 raw=list((directory/'physics').glob('*.json.gz'));steps=0;queries=0;planning_s=0;prefix_errors=[]
 base_files={json.loads(gzip.decompress(p.read_bytes()))['summary']['seed']:p for p in (a.root/'baseline/physics').glob('*.json.gz')}
 for f in raw:
  d=json.loads(gzip.decompress(f.read_bytes()));steps+=len(d['steps']);assert len(d['steps'])==100
  # Inspect only actual planner entries; no inferred successful-query costs.
  queries+=len(d['summary']['queries'])
  planning_s+=sum(x.get('wall_s',0) for x in d['summary']['queries'])
  if d['summary']['seed'] in base_files:
   ref=json.loads(gzip.decompress(base_files[d['summary']['seed']].read_bytes()));rear=[x for x in d['summary']['decisions'] if x['stage']==1];br=[x for x in ref['summary']['decisions'] if x['stage']==1]
   if rear and br:
    assert rear[0]['step']==br[0]['step'];end=rear[0]['step']
    for field in ['q','qvel','tcp','peg']:
     err=max((float(np.max(np.abs(np.asarray(x[field])-np.asarray(y[field])))) for x,y in zip(d['steps'][:end],ref['steps'][:end])),default=0.)
     prefix_errors.append(err)
 train=json.loads((directory/'TRAIN.json').read_text()) if (directory/'TRAIN.json').exists() else []
 protocol=json.loads((directory/'PROTOCOL.json').read_text()) if (directory/'PROTOCOL.json').exists() else {}
 result[directory.name]=dict(evaluations=evals,full_episodes=len(raw),physical_steps=steps,training_success=sum(r['success']for r in train),training_episodes=len(train),training_steps=len(train)*100,optimizer_iterations=protocol.get('updates',len(train)*40),prefix_max_error=max(prefix_errors,default=0),planning_queries=queries,planning_wall_s=planning_s)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
