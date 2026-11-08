"""Recount actual native evaluation, training and sampling-correction activity."""
import argparse,json,collections,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
summary={'arms':{},'controls':{},'cost':dict(episodes=0,steps=0,queries=0,query_wall_s=0.,episode_wall_s=0.,own_episodes=0,updates=0,effective_decisions=0)}
def charge(rows):
 for h in rows:
  r=h.get('row',h);c=summary['cost'];c['episodes']+=1;c['steps']+=r['steps'];c['queries']+=len(r['queries']);c['query_wall_s']+=sum(q['wall_s']for q in r['queries']);c['episode_wall_s']+=r['wall_s']
  if h.get('kind')=='train':c['own_episodes']+=1;c['updates']+=h['log']['updates'];c['effective_decisions']+=h['effective_decisions']
for f in sorted(a.root.glob('*/HISTORY.json')):
 h=json.loads(f.read_text());charge(h);evals=collections.defaultdict(dict);train=[]
 for row in h:
  if row['kind']=='eval':evals[row['checkpoint']][str(row['row']['seed'])]=bool(row['row']['success'])
  else:train.append(row)
 base=evals.get(0,{});curve={}
 for cp,rs in evals.items():
  curve[str(cp)]=dict(n=len(rs),success=sum(rs.values()),rescue=[s for s,v in rs.items()if v and not base[s]],regression=[s for s,v in rs.items()if not v and base[s]])
 summary['arms'][f.parent.name]=dict(curve=curve,own_episodes=len(train),own_success=sum(r['row']['success']for r in train),updates=sum(r['log']['updates']for r in train),effective_decisions=sum(r['effective_decisions']for r in train),update_wall_s=sum(r['log']['wall_s']for r in train),last_update=train[-1]['log']if train else None,own_rear_boundary_histogram=dict(collections.Counter(sum(e.get('stage')==1 and e.get('boundary',False) for e in r['events'])for r in train)),own_first_rear_actions=dict(collections.Counter(next((e['a']for e in r['events']if e.get('stage')==1),-1)for r in train)),eval_outcomes=evals,training_curve=[dict(episode=r['checkpoint'],success=r['row']['success'],**r['log']['last'])for r in train])
for f in sorted(a.root.glob('*/RESULTS.json')):
 rs=json.loads(f.read_text());charge(rs);summary['controls'][f.parent.name]=dict(n=len(rs),success=sum(r['success']for r in rs),outcomes={str(r['seed']):bool(r['success'])for r in rs},model_manifest=json.loads((f.parent/'MANIFEST.json').read_text()),failure_without_rear_boundary=sum(not r['success'] and not any(e.get('boundary') and e.get('stage')==1 for e in r.get('events',[])) for r in rs))
probe=a.root/'mechanism/CASES.json'
if probe.exists():
 rs=json.loads(probe.read_text());charge(rs);summary['mechanism']=[dict(mode=r['mode'],seed=r['seed'],success=r['success'],path=r['path'])for r in rs]
(a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:{'curve':v['curve'],'own':v['own_episodes']}for k,v in summary['arms'].items()},indent=2));print(summary['cost'])
