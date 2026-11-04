"""Read immutable native results; never select historical endpoints retroactively."""
import argparse,json,collections,gzip,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);a=p.parse_args()
nom={r['seed']:r['success'] for r in json.loads((a.old/'controls/RESULTS.json').read_text()) if r['method']=='nominal'}
for f in a.root.glob('*/RESULTS.json'):
 mf=f.parent/'MANIFEST.json'
 if mf.exists():
  m=json.loads(mf.read_text())
  if m.get('control')=='nominal' and not m.get('checkpoint_sha'):
   nom.update({r['seed']:r['success']for r in json.loads(f.read_text())})
out={};cost=[];new_training=[]
paths=list(a.root.glob('*/HISTORY.json'))+[a.old/n/'HISTORY.json' for n in ['online_sac','online_success_bc']]
for f in paths:
 h=json.loads(f.read_text());m=json.loads((f.parent/'MANIFEST.json').read_text());tr=[x for x in h if x['job']['kind']=='train'];ev={}
 for cp in sorted(set(x['job']['checkpoint']for x in h if x['job']['kind']=='eval')):
  rows=[x['row']for x in h if x['job']['kind']=='eval' and x['job']['checkpoint']==cp]
  ev[cp]=dict(success=sum(x['success']for x in rows),n=len(rows),rescue=[x['seed']for x in rows if x['success'] and x['seed'] in nom and not nom[x['seed']]],regression=[x['seed']for x in rows if not x['success'] and nom.get(x['seed'],False)])
 out[f.parent.name]=dict(frame=m['frame'],arm=m['arm'],seed=m['optimizer_rng_seed'],training_n=len(tr),training_success=sum(x['row']['success']for x in tr),training_steps=sum(x['row']['steps']for x in tr),training_planning_queries=sum(len(x['row']['queries'])for x in tr),training_planning_wall_s=sum(q['wall_s']for x in tr for q in x['row']['queries']),training_episode_wall_s=sum(x['row']['wall_s']for x in tr),optimization_wall_s=sum(x['log']['wall_s']for x in tr if x.get('log')),evaluation_planning_queries=sum(len(x['row']['queries'])for x in h if x['job']['kind']=='eval'),decisions=sum(x['effective_decisions']for x in tr),updates=sum(x['updates']for x in tr),front=dict(collections.Counter(x['row']['decisions'][0]['id'] for x in tr)),evaluation=ev,episode_curve=[dict(index=i+1,success=x['row']['success'],steps=x['row']['steps'],queries=len(x['row']['queries']))for i,x in enumerate(tr)],manifest_sha=hashlib.sha256((f.parent/'MANIFEST.json').read_bytes()).hexdigest())
 if f.is_relative_to(a.root):
  cost.extend(x['row']for x in h);new_training.extend(tr)
for f in a.root.glob('*/RESULTS.json'):
 rows=json.loads(f.read_text());cost.extend(rows);mf=json.loads((f.parent/'MANIFEST.json').read_text());out[f.parent.name]=dict(scope=mf.get('scope','control'),paired_baseline_n=sum(r['seed']in nom for r in rows),success=sum(r['success']for r in rows),n=len(rows),rescue=[r['seed']for r in rows if r['success'] and r['seed'] in nom and not nom[r['seed']]],regression=[r['seed']for r in rows if not r['success'] and nom.get(r['seed'],False)])
for r in cost:
 p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(p,'rt'));assert len(d['steps'])==100 and bool(d['steps'][-1]['evaluation']['success'][0])==r['success']
s=dict(arms=out,new_cost=dict(training_episodes=len(new_training),training_steps=sum(x['row']['steps']for x in new_training),effective_decisions=sum(x['effective_decisions']for x in new_training),actor_updates=sum(x['updates']for x in new_training),critic_updates=sum(x['updates']for x in new_training),unused_aux_updates=sum(x['updates']for x in new_training),optimization_wall_s=sum(x['log']['wall_s']for x in new_training if x.get('log')),episodes=len(cost),steps=sum(r['steps']for r in cost),planning_queries=sum(len(r['queries'])for r in cost),planning_wall_s=sum(q['wall_s']for r in cost for q in r['queries']),episode_wall_s=sum(r['wall_s']for r in cost)),scope='seen development; historical measured final64 unchanged')
(a.root/'SUMMARY.json').write_text(json.dumps(s,indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='episode_curve'}for k,v in out.items()},indent=2))
