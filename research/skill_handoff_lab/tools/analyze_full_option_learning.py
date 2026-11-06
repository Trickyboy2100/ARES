"""Recount native outcomes, learning costs and scene-paired changes; no oracle score."""
import argparse, collections, gzip, json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--partial',action='store_true'); a=p.parse_args()
groups={}; learning={}; raw={}; skipped=[]
for folder in sorted(a.root.iterdir()):
 if not folder.is_dir(): continue
 hp=folder/'HISTORY.json'; rp=folder/'RESULTS.json'
 if hp.exists():
  h=json.loads(hp.read_text()); tr=[x for x in h if x['kind']=='train']
  learning[folder.name]=dict(episodes=len(tr),success=sum(x['row']['success'] for x in tr),physical_steps=sum(x['row']['steps'] for x in tr),effective_decisions=sum(x['effective_decisions'] for x in tr),updates=sum((x['log'] or {}).get('updates',0) for x in tr),optimizer_wall_s=sum((x['log'] or {}).get('wall_s',0) for x in tr),planning_queries=sum(len(x['row']['queries']) for x in tr),front_counts=dict(collections.Counter(str(e['a']) for x in tr for e in x['events'] if e.get('boundary') and e.get('stage')==0)),rear_selector_counts=dict(collections.Counter(str(e['a']) for x in tr for e in x['events'] if e.get('boundary') and e.get('stage')==1)),rear_physical_counts=dict(collections.Counter(str(e['physical']) for x in tr for e in x['events'] if e.get('boundary') and e.get('stage')==1)),curve=[dict(episode=x['checkpoint']+1,seed=x['row']['seed'],success=bool(x['row']['success']),decisions=x['effective_decisions'],queries=len(x['row']['queries']),updates=x['log']['updates']) for x in tr])
  for ck in sorted({x['checkpoint'] for x in h if x['kind']=='eval'}): groups[f'{folder.name}@{ck}']=[x['row'] for x in h if x['kind']=='eval' and x['checkpoint']==ck]
 elif rp.exists() and not folder.name.startswith('design_'): groups[folder.name]=json.loads(rp.read_text())
 for f in folder.rglob('*.json.gz'):
  try: d=json.load(gzip.open(f,'rt'))
  except (EOFError,json.JSONDecodeError):
   if not a.partial: raise
   skipped.append(str(f));continue
  if 'summary' not in d: continue
  r=d['summary']; assert r['steps']==len(d['steps'])==100, f
  assert bool(r['success'])==bool(d['steps'][-1]['evaluation']['success'][0]), f
  raw[str(f.resolve())]=r
out=dict(groups={},learning=learning,paired={},cost={},partial=a.partial,skipped=skipped)
for k,rs in groups.items():
 assert len({r['seed'] for r in rs})==len(rs),k
 for r in rs:
  nr=raw[str(Path(r['path']).resolve())];assert nr['success']==r['success'] and nr['steps']==r['steps']
 out['groups'][k]=dict(n=len(rs),success=sum(r['success'] for r in rs),outcomes={str(r['seed']):bool(r['success']) for r in rs},planning_queries=sum(len(r['queries']) for r in rs),planner_wall_s=sum(q['wall_s'] for r in rs for q in r['queries']),physical_steps=sum(r['steps'] for r in rs),summed_episode_wall_s=sum(r['wall_s'] for r in rs))
rng=np.random.default_rng(32297)
for name,g in out['groups'].items():
 for ref,b in out['groups'].items():
  if name==ref or set(g['outcomes'])!=set(b['outcomes']):continue
  if not (ref.endswith('@0') or any(t in ref for t in ['rule','fixed5','old_','sac_'])):continue
  ids=list(g['outcomes']);delta=np.array([int(g['outcomes'][s])-int(b['outcomes'][s]) for s in ids]);bs=delta[rng.integers(len(ids),size=(10000,len(ids)))].mean(1)
  out['paired'][name+' versus '+ref]=dict(rescue=[int(s) for s in ids if g['outcomes'][s] and not b['outcomes'][s]],regression=[int(s) for s in ids if not g['outcomes'][s] and b['outcomes'][s]],difference=float(delta.mean()),scene_bootstrap95=np.quantile(bs,[.025,.975]).tolist(),scope='descriptive seen development/declared confirmation; not sealed test')
out['cost']=dict(unique_native_episodes=len(raw),physical_steps=sum(r['steps'] for r in raw.values()),planning_queries=sum(len(r['queries']) for r in raw.values()),planner_wall_s=sum(q['wall_s'] for r in raw.values() for q in r['queries']),summed_episode_wall_s=sum(r['wall_s'] for r in raw.values()),shared_design_episodes=sum('/design_' in f for f in raw),startup_failed_episodes=[f for f,r in raw.items() if '00025_765100_train_0' in f],accounting='Unique run cost includes design, every evaluation, and two startup failures. Each learner also consumed the common 256-episode design replay. Inherited pretraining is separately pinned, not free online interaction.')
(a.root/'LEARNING_SUMMARY.json').write_text(json.dumps(out,indent=2))
print(json.dumps(dict(groups={k:[v['success'],v['n']] for k,v in out['groups'].items()},training={k:v['episodes'] for k,v in learning.items()},cost=out['cost']),indent=2))
