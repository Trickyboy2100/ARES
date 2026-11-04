"""Check actual archived control-step accounting, native horizon and source links."""
import argparse,gzip,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();cost=json.loads((a.run/'COST.json').read_text());steps=0;queries=0;n=0;prefix_steps=0;init_steps=0
for c in cost['captures']:
 d=a.run/f"capture_{c['seed']}";prefix=json.load(gzip.open(d/'PREFIX.json.gz','rt'));assert len(prefix['steps'])==c['actual_steps'];steps+=c['actual_steps'];prefix_steps+=c['actual_steps'];queries+=len(prefix['queries'])
 for f in sorted(d.glob('suffix_*.json.gz')):
  z=json.load(gzip.open(f,'rt'));s=z['summary'];assert len(z['steps'])==s['actual_steps'];assert len(prefix['steps'])+len(z['steps'])==100;assert bool(z['steps'][-1]['evaluation']['success'][0])==s['success'];assert s['native_deadline']==100
  assert s['initialization_steps']==int(s['state_reset']);assert s['decisions'][0]['step']==s['start_step']+s['initialization_steps'];steps+=len(z['steps']);queries+=len(s['queries']);init_steps+=s['initialization_steps'];n+=1
assert steps==cost['actual_training_steps'];assert queries==cost['planner_queries'];assert n==cost['local_rollouts']
print(json.dumps(dict(status='PASS',actual_training_steps=steps,prefix_steps=prefix_steps,local_suffix_steps=steps-prefix_steps,charged_initialization_steps=init_steps,local_rollouts=n,planning_queries=queries,exact_counterfactual_restore='NOT_CLAIMED'),indent=2))
