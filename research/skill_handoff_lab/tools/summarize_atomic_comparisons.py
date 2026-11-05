"""Paired native outcomes and executor probes, retaining every adverse result."""
import argparse,json,collections,gzip
from pathlib import Path
def resolve_raw(path, root):
 source=Path(path)
 if source.exists():return source
 for n in range(2,len(source.parts)):
  candidate=root.joinpath(*source.parts[-n:])
  if candidate.is_file():return candidate
 raise FileNotFoundError(path)
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
read=lambda f:json.loads(f.read_text())
def compare(rows,nom):
 z={x['seed']:x for x in rows};assert set(z)==set(nom)
 return dict(n=len(z),success=sum(x['success']for x in z.values()),rescue=[s for s in z if z[s]['success']and not nom[s]['success']],regression=[s for s in z if not z[s]['success']and nom[s]['success']],front=dict(collections.Counter(x['decisions'][0]['id']for x in z.values()if 'decisions'in x)))
out={}
for label,base,names in [('development','queue_bc64',['control_rule_chunk','control_rule_phase','control_rule_insertion_queue','control_budget_guard_chunk']),('confirmation','confirm_nominal',['confirm_queue_bc64','confirm_macro_bc64','confirm_queue_seed31702','confirm_macro_seed31702','confirm_rule'])]:
 if label=='development':
  nom={x['job']['seed']:x['row']for x in read(r/base/'HISTORY.json')if x['job']['kind']=='eval'and x['job']['checkpoint']==0}
 elif (r/base/'RESULTS.json').exists():nom={x['seed']:x for x in read(r/base/'RESULTS.json')}
 else:continue
 out[label]={}
 for name in names:
  f=r/name/'RESULTS.json'
  if f.exists():
   rows=read(f)
   if len(rows)==len(nom):out[label][name]=compare(rows,nom)
 out[label]['nominal']=dict(n=len(nom),success=sum(x['success']for x in nom.values()))
f=r/'progress_splice/RESULTS.json'
if f.exists():
 rows=read(f);details=[d['splice']for x in rows for d in x['decisions']if 'splice'in d]
 out['splice']=dict(episodes=len(rows),success=sum(x['success']for x in rows),attempts=len(details),accepted=sum(d['accepted']for d in details),rejections=dict(collections.Counter(d['rejection']for d in details)))
f=r/'learned_rescue/RESULTS.json'
if f.exists():out['learned_rescue']={x['mode']:x['success']for x in read(f)}
f=r/'contact_sequence/RESULTS.json'
if f.exists():
 out['contact_sequence']=[]
 for row in read(f):
  d=json.load(gzip.open(resolve_raw(row['path'],a.root),'rt'));force=[(i+1,sum(v*v for v in z['peg_fixture_force_N'])**.5)for i,z in enumerate(d['steps'])];contact=[i for i,v in force if v>1e-6]
  out['contact_sequence'].append(dict(termination=row['termination'],success=row['success'],first_nonzero_contact_step=min(contact)if contact else None,max_force_N=max(v for i,v in force),note='net pairwise force; nonzero threshold is diagnostic, not controller input'))
(r/'COMPARISONS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
