"""Recount actual runs; shared evidence is charged per learner and once globally."""
import argparse,collections,csv,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def stats(rows):
 return dict(episodes=len(rows),success=sum(bool(r['success'])for r in rows),steps=sum(r['steps']for r in rows),planner_calls=sum(len(r['queries'])for r in rows),planner_seconds=sum(q['wall_s']for r in rows for q in r['queries']),execution_seconds=sum(r['wall_s']for r in rows),errors=[dict(seed=r['seed'],error=r['error'])for r in rows if r['error']])
def paired(rows,base):
 ys={r['seed']:bool(r['success'])for r in rows};assert len(ys)==len(rows)
 return dict(**stats(rows),outcomes=ys,rescue=[s for s,y in ys.items()if y and not base[s]],regression=[s for s,y in ys.items()if not y and base[s]])
groups={};training={};shared=[];physical={};pending=[]
for f in sorted(a.root.glob('design*/RESULTS.json')):shared+=read(f)
# Source manifests are retained; no second counting of embedded replay/transactions.
for f in sorted(a.root.rglob('physics/*.json.gz')):
 try:d=json.load(gzip.open(f,'rt'))['summary']
 except (EOFError,OSError,json.JSONDecodeError):pending.append(str(f));continue
 physical[str(f.relative_to(a.root))]=d
for f in sorted(a.root.glob('*/HISTORY.json')):
 h=read(f);name=f.parent.name;ev=[x for x in h if x['kind']=='eval'];base={x['row']['seed']:bool(x['row']['success'])for x in ev if x['checkpoint']==0};tr=[x for x in h if x['kind']=='train'];refresh=[v for k,v in physical.items()if k.startswith(name+'/refresh64/')]
 for k in sorted(set(x['checkpoint']for x in ev)):
  rows=[x['row']for x in ev if x['checkpoint']==k];groups[f'{name}@{k}']=paired(rows,base)
 logs=[x['log'].get('projection',{})for x in tr];training[name]=dict(own=stats([x['row']for x in tr]),own_decisions=sum(x['effective_decisions']for x in tr),updates=sum(x['log']['updates']for x in tr),optimizer_seconds=sum(x['log']['wall_s']for x in tr),shared_charge=stats(shared),refresh=stats(refresh),projection=dict(cuts=sum(x.get('cuts',0)for x in logs),zero_steps=sum(x.get('zero_steps',0)for x in logs),attempts=sum(x.get('attempts',0)for x in logs),seconds=sum(x.get('wall_s',0)for x in logs)))
conf={f.parent.name:read(f)for f in sorted(a.root.glob('confirm_*/RESULTS.json'))}
if 'confirm_warm'in conf:
 base={r['seed']:bool(r['success'])for r in conf['confirm_warm']}
 for name,rows in conf.items():
  if all(r['seed']in base for r in rows):groups[name]=paired(rows,base)
else:
 for name,rows in conf.items():groups[name]=dict(**stats(rows),pairing_pending=True)
initial_base={r['seed']:bool(r['success'])for r in shared if r['tag']=='reference'}
for f in sorted(a.root.glob('training_witness_*/RESULTS.json')):groups[f.parent.name]=paired(read(f),initial_base)
disagreement=None
if (a.root/'disagreement_design/RESULTS.json').exists():
 rows=read(a.root/'disagreement_design/RESULTS.json');ps=read(a.root/'disagreement_design/PAIRS.json');ref={r['seed']:bool(r['success'])for r in rows if r['tag']=='reference'};union={k:v for k,v in ref.items()}
 for r in rows:union[r['seed']]=union[r['seed']]or bool(r['success'])
 disagreement=dict(cost=stats(rows),reference_scenes=len(ref),reference_success=sum(ref.values()),finite_union=sum(union.values()),pairs=len(ps),positive=sum(p['delta']>0 for p in ps),negative=sum(p['delta']<0 for p in ps),ties=sum(p['delta']==0 for p in ps),positive_pairs=[p for p in ps if p['delta']>0],scope='Additional design, not used by the primary128-episode trained arms; different scenes from cyclic acquisition, not a controlled acquisition-efficiency comparison')
summary=dict(disagreement=disagreement,groups=groups,training=training,global_unique=stats(list(physical.values())),pending_files=pending,shared=stats(shared),scope='Public native simulation; shared cost charged per arm but unique physical executions counted once globally. Confirmation is new development, not sealed test. Variable legal branch availability changes actual refresh cost, reported explicitly.')
(a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
with(a.out/'RESULTS.csv').open('w')as f:
 w=csv.writer(f);w.writerow(['policy','episodes','success','rescue','regression','planner_calls','planner_seconds'])
 for k,v in groups.items():w.writerow([k,v['episodes'],v['success'],len(v.get('rescue',[])),len(v.get('regression',[])),v['planner_calls'],v['planner_seconds']])
print(json.dumps(dict(groups={k:(v['success'],v['episodes'])for k,v in groups.items()},global_unique={k:v for k,v in summary['global_unique'].items()if k!='errors'},termination_counts=dict(collections.Counter(x['error']for x in summary['global_unique']['errors'])),pending=len(pending)),indent=2))
