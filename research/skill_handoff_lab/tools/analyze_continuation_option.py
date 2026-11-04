"""Raw native outcomes, matched rescue/regression and actual compute accounting."""
import argparse,json,gzip,collections,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
controls=json.loads((r/'controls/RESULTS.json').read_text());zero={x['seed']:x['success']for x in controls if x['method']=='zero'};table=[]
def add(name,rows):
 assert len({x['seed']for x in rows})==len(rows)
 table.append(dict(method=name,success=sum(x['success']for x in rows),n=len(rows),rescue=[x['seed']for x in rows if x['success']and not zero[x['seed']]],regression=[x['seed']for x in rows if not x['success']and zero[x['seed']]]))
for name in sorted({x['method']for x in controls}):add(name,[x for x in controls if x['method']==name])
for name in ['shared_train','joint_train','smdp_train']:
 path=r/name/'EVALUATION.json'
 if not path.exists():continue
 data=json.loads(path.read_text())
 for ck in sorted({x['checkpoint']for x in data}):
  for method in sorted({x['method']for x in data}):add(f'{name}/{method}/{ck}',[x for x in data if x['checkpoint']==ck and x['method']==method])
cost=collections.Counter();groups={}
for directory in r.glob('*/physics'):
 c=collections.Counter()
 for f in directory.glob('*.json.gz'):
  d=json.load(gzip.open(f));s=d['summary'];assert len(d['steps'])==100 and s['steps']==100
  assert bool(d['steps'][-1]['evaluation']['success'][0])==s['success']
  c['episodes']+=1;c['physical_steps']+=100;c['planner_queries']+=len(s['queries']);c['planner_wall_s']+=sum(q['wall_s']for q in s['queries']);c['episode_wall_s']+=s['wall_s']
 groups[directory.parent.name]=dict(c);cost.update(c)
for name in ['shared_train','joint_train','smdp_train']:
 path=r/name/'HISTORY.json'
 if path.exists():
  hist=json.loads(path.read_text());cost['critic_updates']+=80*len(hist);cost['actor_updates']+=120*len(hist);cost['legacy_auxiliary_updates']+=80*len(hist)
models=[p for p in r.rglob('*.pt')];cost['saved_models']=len(models)
summary=dict(table=table,cost=dict(cost),physical_groups=groups,scope='758200..758215 natural development, now seen; no sealed data; one inherited neural seed',independent_gain='UNVERIFIED')
(r/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(r/'MODEL_SHA.json').write_text(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest()for p in models},indent=2))
for row in table:print(row['method'],f"{row['success']}/{row['n']}",len(row['rescue']),len(row['regression']))
print('COST',dict(cost))
