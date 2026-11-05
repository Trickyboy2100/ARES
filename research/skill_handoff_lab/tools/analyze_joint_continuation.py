"""Native terminal re-count and scene-cluster paired descriptive intervals."""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();groups={};curves={};raw={};training={}
for folder in sorted(a.root.iterdir()):
 if not folder.is_dir():continue
 h=folder/'HISTORY.json';r=folder/'RESULTS.json'
 if h.exists():
  hist=json.loads(h.read_text());train=[v for v in hist if v['kind']=='train'];training[folder.name]=dict(episodes=len(train),success=sum(v['row']['success']for v in train),effective_decisions=sum(v['effective_decisions']for v in train),updates=sum((v['log']or{}).get('updates',0)for v in train),optimizer_wall_s=sum((v['log']or{}).get('wall_s',0)for v in train),front_choices=[v['events'][0]['a']for v in train]);curves[folder.name]={}
  for ck in sorted({v['checkpoint']for v in hist if v['kind']=='eval'}):
   rows=[v['row']for v in hist if v['kind']=='eval'and v['checkpoint']==ck];groups[f'{folder.name}@{ck}']=rows;curves[folder.name][str(ck)]=dict(n=len(rows),success=sum(v['success']for v in rows))
 elif r.exists():groups[folder.name]=json.loads(r.read_text())
 for f in folder.rglob('*.json.gz'):
  d=json.load(gzip.open(f,'rt'));row=d['summary'];assert row['steps']==len(d['steps'])==100;assert bool(d['steps'][-1]['evaluation']['success'][0])==bool(row['success']);raw[str(f.resolve())]=row
for name,rows in groups.items():
 for row in rows:
  native=raw[str(Path(row['path']).resolve())];assert bool(row['success'])==bool(native['success']) and row['steps']==native['steps']
summary=dict(groups={k:dict(n=len(v),unique_scenes=len({x['seed']for x in v}),records=[dict(seed=x['seed'],success=bool(x['success']),path=x['path'])for x in v],success=sum(x['success']for x in v),seeds=[x['seed']for x in v],outcomes={str(x['seed']):bool(x['success'])for x in v})for k,v in groups.items()},learning_curves=curves,training=training,paired={},design={})
# Intervals resample scene clusters, not individual steps or duplicated methods.
rng=np.random.default_rng(86931)
for name,rows in groups.items():
 reference='frozen_rule' if set(x['seed']for x in rows)==set(range(764400,764432)) else ('identity_dev_rule' if any(764600<=x['seed']<764616 for x in rows) else 'dev_rule')
 if reference not in groups or len({x['seed']for x in rows})!=len(rows):continue
 b={x['seed']:bool(x['success'])for x in groups[reference]};rows=[x for x in rows if x['seed']in b]
 if not rows:continue
 delta=np.array([int(x['success'])-int(b[x['seed']])for x in rows]);bs=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
 summary['paired'][name]=dict(reference=reference,rescue=[x['seed']for x in rows if x['success']and not b[x['seed']]],regression=[x['seed']for x in rows if not x['success']and b[x['seed']]],mean_difference=float(delta.mean()),scene_bootstrap95=np.quantile(bs,[.025,.975]).tolist())
for mode in ['rule','warm']:
 names=[f'design_{mode}_g{g}'for g in range(3)]
 if all(n in groups and len(groups[n])==16 for n in names):
  table={s:[next(bool(x['success'])for x in groups[n]if x['seed']==s)for n in names]for s in range(764000,764016)}
  summary['design'][mode]=dict(fixed=[sum(row[g]for row in table.values())for g in range(3)],union=sum(any(row)for row in table.values()),rows=table,sensitive=[s for s,row in table.items()if len(set(row))>1])
summary['cost']=dict(episodes=len(raw),steps=sum(x['steps']for x in raw.values()),queries=sum(len(x['queries'])for x in raw.values()),planner_wall_s=sum(q['wall_s']for x in raw.values()for q in x['queries']),summed_episode_wall_s=sum(x['wall_s']for x in raw.values()))
(a.root/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(cost=summary['cost'],groups={k:[v['success'],v['n']]for k,v in summary['groups'].items()},design=summary['design'],curves=curves),indent=2))
