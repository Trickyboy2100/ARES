"""Post-freeze trace attribution; no policy or threshold is fitted to confirmation."""
import argparse,gzip,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
base={r['seed']:r for r in json.loads((a.root/'confirm_warm/RESULTS.json').read_text())};out={}
def load(r):return json.load(gzip.open(r['path'],'rt'))
def signature(s):return {k:s[k]for k in ['q','qvel','tcp','peg','action']}
for f in sorted(a.root.glob('confirm_*/RESULTS.json')):
 rows=json.loads(f.read_text());assert len(rows)==32
 cases=[]
 for r in rows:
  b=base[r['seed']]
  if r['success']==b['success']:continue
  x,y=load(b),load(r);initial_equal=x['initial']==y['initial'];prefix=0
  if initial_equal:
   for u,v in zip(x['steps'],y['steps']):
    if signature(u)!=signature(v):break
    prefix+=1
  def brief(row,d):
   return dict(path=row['path'],sha256=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest(),success=row['success'],error=row['error'],first_success=next((s['step']for s in d['steps']if s['evaluation']['success'][0]),None),last_phase=d['steps'][-1]['phase'],last_grasped=d['steps'][-1]['grasped'],boundary_events=[v for v in row.get('events',[])if v.get('boundary')])
  cases.append(dict(seed=r['seed'],change='rescue'if r['success']else'regression',initial_equal=initial_equal,exact_prefix_steps=prefix,reference=brief(b,x),policy=brief(r,y)))
 out[f.parent.name]=cases
(a.root/'CONFIRMATION_BEHAVIOR.json').write_text(json.dumps(dict(scope='Frozen full-policy observational contrasts from exact native resets; not isolated one-action causality or training data',methods=out),indent=2))
print(json.dumps({k:[(v['seed'],v['change'],v['exact_prefix_steps'])for v in vs]for k,vs in out.items()},indent=2))
front={}
for f in sorted(a.root.glob('*_322*/HISTORY.json')):
 h=json.loads(f.read_text())
 def choices(ck):return {x['row']['seed']:(next(e['a']for e in x['events']if e.get('boundary')and e['stage']==0),x['row']['success']) for x in h if x['kind']=='eval'and x['checkpoint']==ck}
 b,v=choices(0),choices(128);front[f.parent.name]=[(s,b[s][0],v[s][0],v[s][1])for s in v if b[s][0]!=v[s][0]]
(a.root/'PREDECESSOR_DEPLOYMENT_CHANGES.json').write_text(json.dumps(dict(scope='Final128 versus initial on seen24; actual parameter changes do not imply independent joint benefit',changes=front),indent=2))
