"""Read immutable native results; never select historical endpoints retroactively."""
import argparse,json,collections,gzip,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
nom={r['seed']:r['success'] for r in json.loads((a.old/'controls/RESULTS.json').read_text()) if r['method']=='nominal'}
out={};cost=[]
paths=list(a.root.glob('*/HISTORY.json'))+[a.old/n/'HISTORY.json' for n in ['online_sac','online_success_bc']]
for f in paths:
 h=json.loads(f.read_text());m=json.loads((f.parent/'MANIFEST.json').read_text());tr=[x for x in h if x['job']['kind']=='train'];ev={}
 for cp in sorted(set(x['job']['checkpoint']for x in h if x['job']['kind']=='eval')):
  rows=[x['row']for x in h if x['job']['kind']=='eval' and x['job']['checkpoint']==cp]
  ev[cp]=dict(success=sum(x['success']for x in rows),n=len(rows),rescue=[x['seed']for x in rows if x['success'] and not nom.get(x['seed'],False)],regression=[x['seed']for x in rows if not x['success'] and nom.get(x['seed'],False)])
 out[f.parent.name]=dict(frame=m['frame'],arm=m['arm'],seed=m['optimizer_rng_seed'],training_n=len(tr),training_success=sum(x['row']['success']for x in tr),decisions=sum(x['effective_decisions']for x in tr),updates=sum(x['updates']for x in tr),front=dict(collections.Counter(x['row']['decisions'][0]['id'] for x in tr)),evaluation=ev,episode_curve=[dict(index=i+1,success=x['row']['success'],steps=x['row']['steps'],queries=len(x['row']['queries']))for i,x in enumerate(tr)],manifest_sha=hashlib.sha256((f.parent/'MANIFEST.json').read_bytes()).hexdigest())
 if f.is_relative_to(a.root):cost.extend(x['row']for x in h)
for f in a.root.glob('*/RESULTS.json'):
 rows=json.loads(f.read_text());cost.extend(rows);out[f.parent.name]=dict(success=sum(r['success']for r in rows),n=len(rows),rescue=[r['seed']for r in rows if r['success'] and not nom.get(r['seed'],False)],regression=[r['seed']for r in rows if not r['success'] and nom.get(r['seed'],False)])
for r in cost:
 p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(p,'rt'));assert len(d['steps'])==100 and bool(d['steps'][-1]['evaluation']['success'][0])==r['success']
s=dict(arms=out,new_cost=dict(episodes=len(cost),steps=sum(r['steps']for r in cost),planning_queries=sum(len(r['queries'])for r in cost),planning_wall_s=sum(q['wall_s']for r in cost for q in r['queries']),episode_wall_s=sum(r['wall_s']for r in cost)),scope='seen development; historical measured final64 unchanged')
(a.root/'SUMMARY.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='episode_curve'}for k,v in out.items()},indent=2))
