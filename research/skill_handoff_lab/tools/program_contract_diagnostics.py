"""Physics and learning diagnostics computed from retained files, not planned budgets."""
import argparse,json,gzip,hashlib,collections
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True);groups={};seen=set();records=[]
 for file in sorted(a.root.rglob('RESULTS.json')):
  if 'analysis'in file.parts or 'matching_512_data' in file.parts:continue
  rows=json.loads(file.read_text())
  if not isinstance(rows,list):continue
  group=str(file.parent.relative_to(a.root));gg=[]
  for r in rows:
   if not isinstance(r,dict)or 'path'not in r:continue
   path=Path(r['path'])
   if str(path)in seen:continue
   seen.add(str(path));assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(path,'rt'));steps=d['steps'];assert len(steps)==r['steps'];assert bool(steps[-1]['native_success'])==bool(r['success']);r=dict(r,group=group,physics_substeps=sum(len(z.get('physics_substep_contacts',[]))for z in steps),planner_calls=sum('position'in z for z in d['plans']),planner_wall_s=sum(z.get('planning_wall_s',0)for z in d['plans']),solver_failures=sum(z['phase']=='SOLVER_FAILURE'for z in d['plans']),decisions=len(d.get('policy_actions',d['decisions'])),observed_boundary=d['skill_results'][0]['boundary_step']if d['skill_results']else None)
   if r['success']:failure='native_success'
   elif not d['skill_results']:failure='predecessor_not_verified'
   elif 'Horizon'in str(r['error']):failure='budget_censored'
   elif r['error']:failure='successor_plan_failure'
   else:failure='successor_finished_native_fail'
   r['failure_class']=failure;gg.append(r);records.append(r)
  if gg:groups[group]=dict(episodes=len(gg),success=sum(x['success']for x in gg),control_steps=sum(x['steps']for x in gg),physical_substeps=sum(x['physics_substeps']for x in gg),decisions=sum(x['decisions']for x in gg),planning_wall_s=sum(x['planner_wall_s']for x in gg),wall_s=sum(x['wall_s']for x in gg),failure_classes=dict(collections.Counter(x['failure_class']for x in gg)))
 summary=dict(groups=groups,complete_physical_episodes=len(records),complete_control_steps=sum(x['steps']for x in records),complete_boundary_decisions=sum(x['decisions']for x in records),completed_training_rl_decisions=sum(x['decisions']for x in records if x['group'] in ['sac/episodes','sac_recovered/episodes']),rows=records,unknown_partial_note='First SAC attempt crashed before checkpoint:67 complete episodes retained; one partial episode not saved,0..100 additional steps not falsely counted as0. Startup missingSB3 import added0physics. Recovered run stores interruption state.')
 (a.out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items()if k!='rows'},indent=2))
 # Per-scene physical trajectories identify actual action interactions, not learned causal labels.
 rows=[r for r in records if r['group']=='depth_grid/episodes'];grid={}
 for seed in sorted({r['seed']for r in rows}):
  rr=[r for r in rows if r['seed']==seed];grid[str(seed)]=[dict(grasp=r['grasp'],height=r['hook_height'],success=r['success'],failure=r['failure_class'])for r in rr]
 (a.out/'DEPTH_OPPORTUNITY.json').write_text(json.dumps(grid,indent=2))
if __name__=='__main__':main()
