"""Whole-policy paired reexecutions; first difference is not isolated action credit."""
import argparse,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=[]
def load(path):return json.load(gzip.open(path,'rt'))
def prefix(x,y):
 if x['initial']!=y['initial']:return -1
 n=0
 for u,v in zip(x['steps'],y['steps']):
  if any(u[k]!=v[k]for k in ['q','qvel','tcp','peg','action']):break
  n+=1
 return n
for f in sorted(a.root.glob('*/HISTORY.json')):
 h=json.loads(f.read_text());ref={x['row']['seed']:x for x in h if x['kind']=='eval'and x['checkpoint']==0}
 for x in h:
  if x['kind']!='eval'or x['checkpoint']==0:continue
  b=ref[x['row']['seed']];r=x['row'];rr=b['row'];events=[e for e in x['events']if e.get('boundary')];baseline=[e for e in b['events']if e.get('boundary')];first=None
  for i,(u,v)in enumerate(zip(baseline,events)):
   if any(u[k]!=v[k]for k in ['a','step','stage']):first=dict(index=i,reference=u,learned=v);break
  if first is None and len(events)!=len(baseline):first=dict(index=min(len(events),len(baseline)),different_length=True)
  if bool(r['success'])!=bool(rr['success'])or first:
   out.append(dict(policy=f.parent.name,checkpoint=x['checkpoint'],seed=r['seed'],reference_success=bool(rr['success']),success=bool(r['success']),rescue=bool(r['success']and not rr['success']),regression=bool(rr['success']and not r['success']),same_measured_prefix_steps=prefix(load(rr['path']),load(r['path'])),first_boundary_difference=first,reference_path=rr['path'],path=r['path'],scope='Whole-policy change; first differing boundary does not isolate later-policy causal effects'))
(a.root/'DEPLOYMENT_BEHAVIOR.json').write_text(json.dumps(out,indent=2));print(len(out),'changed native policy executions')
