"""Read only actual saved native episodes and checkpoints; no synthetic scores."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out={};cost=dict(episodes=0,control_steps=0,decisions=0,planner_queries=0,planner_wall_s=0,errors={});errors=collections.Counter();cost_by={}
for f in sorted(r.glob('*/physics/RESULTS.json'))+sorted(r.glob('*/legacy/RESULTS.json'))+sorted(r.glob('*/feedback/RESULTS.json')):
 rows=json.loads(f.read_text());name=str(f.parent.parent.relative_to(r))+'/'+f.parent.name;cc=dict(episodes=len(rows),control_steps=sum(x['steps']for x in rows),decisions=sum(len(x['decisions'])for x in rows),planner_queries=sum(len(x['queries'])for x in rows),planner_wall_s=sum(q['wall_s']for x in rows for q in x['queries']))
 cost_by[name]=cc
 for k,v in cc.items():cost[k]+=v
 errors.update(str(x['error'])for x in rows if x['error'] not in [None,'HORIZON','PLAN_FAILED'])
 tags={}
 for t in sorted({x['tag']for x in rows}):
  rr=[x for x in rows if x['tag']==t];tags[t]=dict(n=len(rr),success=sum(x['success']for x in rr),scenes=[x['seed']for x in rr],success_scenes=[x['seed']for x in rr if x['success']])
 if name.startswith('train_'):continue
 out[name]=tags
cost['errors']=dict(errors);training={}
for d in sorted(r.glob('train_*')):
 if not d.is_dir():continue
 evals={}
 for f in sorted(d.glob('EVAL_*.json')):
  rows=json.loads(f.read_text());evals[int(f.stem.split('_')[-1])]=dict(n=len(rows),success=sum(x['success']for x in rows),success_scenes=[x['seed']for x in rows if x['success']],choices={str(x['seed']):[z['id']for z in x['decisions']]for x in rows})
 actual=json.loads((d/'physics/RESULTS.json').read_text())if(d/'physics/RESULTS.json').exists()else[]
 online=[x for x in actual if x['tag'].startswith('online_')]
 training[d.name]=dict(evals=evals,online_episodes=len(online),online_success=sum(x['success']for x in online),online_steps=sum(x['steps']for x in online),online_decisions=sum(len(x['decisions'])for x in online),checkpoints=len(list(d.glob('*.pt'))))
final=dict(cost=cost,cost_by=cost_by,baselines=out,training=training,note='All scores actual continuous native100 physics, same scene across methods is paired not independent. Early failed interface debug episodes included in costs, excluded scientific method comparisons.')
(r/'SUMMARY.json').write_text(json.dumps(final,indent=2));print(json.dumps(dict(cost=cost,training={k:{'evals':{t:z['success']for t,z in v['evals'].items()},'online':v['online_episodes']}for k,v in training.items()}),indent=2))
if training:
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4.5));colors={'ordinary':'#1261a0','budget':'#b85c00','suffix':'#23733e'}
 for k,v in training.items():
  ks=sorted(v['evals']);mode=k.split('_')[1];ax.plot(ks,[v['evals'][n]['success']/v['evals'][n]['n']for n in ks],marker='o',label=k.replace('train_',''),color=colors.get(mode))
 ax.set(xlabel='New full native100 training episodes per method',ylabel='Full-task native success rate',ylim=(0,1),title='Closed-loop skill learning — actual full-prefix development rollouts');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(r/'LEARNING.png',dpi=180);plt.close(fig)
