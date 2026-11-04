"""Summarize actual native outcomes/costs without treating decisions as scenes."""
import argparse,json,gzip,hashlib
from pathlib import Path
from collections import Counter,defaultdict
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
refs=json.loads((r/'references/RESULTS.json').read_text());base={x['seed']:x['success'] for x in refs if x['tag']=='zero_suffix'}
rows=[]
def add(name,rs):
 rs=[x for x in rs if x['seed'] in base];rows.append(dict(method=name,n=len(rs),success=sum(x['success'] for x in rs),rescue=sum(x['success'] and not base[x['seed']] for x in rs),regression=sum(not x['success'] and base[x['seed']] for x in rs),actions=dict(Counter(str(y['id']) for x in rs for y in x['decisions'] if y['stage']==1))))
for tag in ['zero_suffix','budget_refine_zplus']:add(tag,[x for x in refs if x['tag']==tag])
for f in sorted(r.glob('*/final_eval.json')):
 for tag in ['offline_eval','final_eval']:
  path=f.parent/(tag+'.json')
  if path.exists():add(f.parent.name+'/'+tag,json.loads(path.read_text()))
raw=list(r.glob('*/physics/*.json.gz'));nstep=0;queries=0;planning_s=0;decisions=0;manifest=[]
for f in raw:
 d=json.load(gzip.open(f,'rt'));assert len(d['steps'])==100
 assert bool(d['steps'][-1]['evaluation']['success'][0])==d['summary']['success']
 nstep+=len(d['steps']);queries+=len(d['summary']['queries']);planning_s+=sum(x['wall_s'] for x in d['summary']['queries']);decisions+=len(d['summary']['decisions'])
 manifest.append(dict(path=str(f.relative_to(r)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
summary=dict(results=rows,cost=dict(episodes=len(raw),physical_steps=nstep,planner_queries=queries,planner_wall_s=planning_s,decisions=decisions),claim='UNVERIFIED')
(r/'SUMMARY.json').write_text(json.dumps(summary,indent=2));(r/'RAW_SHA.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(summary,indent=2))
