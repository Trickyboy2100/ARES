"""Recompute complete-task paired outcomes and unique physical costs."""
import argparse,json,collections,csv,gzip
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
result=[];curves=[];cost=[];pairs=[]
for f in sorted(a.root.glob('*/HISTORY.json')):
 h=read(f);name=f.parent.name;train=[r for r in h if r['kind']=='train'];ev=[r for r in h if r['kind']=='eval'];baseline={r['row']['seed']:bool(r['row']['success'])for r in ev if r['checkpoint']==0}
 for k in sorted(set(r['checkpoint']for r in ev)):
  rows=[r['row']for r in ev if r['checkpoint']==k];ys={r['seed']:bool(r['success'])for r in rows};rescue=[s for s,y in ys.items()if y and not baseline[s]];regress=[s for s,y in ys.items()if not y and baseline[s]]
  result.append(dict(policy=name,checkpoint=k,n=len(rows),success=sum(ys.values()),rescue=len(rescue),regression=len(regress),rescue_scenes=rescue,regression_scenes=regress,planner_calls=sum(len(r['queries'])for r in rows)))
 for r in train:curves.append(dict(policy=name,episode=r['checkpoint']+1,success=int(r['row']['success']),decisions=r['effective_decisions'],planner_calls=len(r['row']['queries']),updates=r['log']['updates'],**r['log']['last']))
 cost.append(dict(policy=name,own_episodes=len(train),own_steps=sum(r['row']['steps']for r in train),own_decisions=sum(r['effective_decisions']for r in train),updates=sum(r['log']['updates']for r in train),own_planner_calls=sum(len(r['row']['queries'])for r in train)))
for f in a.root.rglob('PAIRS.json'):
 x=read(f);c=collections.Counter(t['delta']for t in x);pairs.append(dict(source=str(f.relative_to(a.root)),scenes=len(set(t['seed']for t in x)),pairs=len(x),rescues=c[1],regressions=c[-1],ties=c[0]))
(a.out/'RESULTS.json').write_text(json.dumps(result,indent=2));(a.out/'COST.json').write_text(json.dumps(cost,indent=2));(a.out/'PAIR_DENSITY.json').write_text(json.dumps(pairs,indent=2))
for name,rows in [('LEARNING_CURVES',curves),('RESULTS',result)]:
 if rows:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with (a.out/(name+'.csv')).open('w')as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
print(json.dumps(dict(results=result,cost=cost,pairs=pairs),indent=2))
