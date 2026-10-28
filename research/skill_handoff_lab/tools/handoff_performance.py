"""Separate measured compute, actual native time and logical completion."""
import argparse,csv,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root,out=a.root,a.output;rows=[];timings=[];sources=[]
 for split in ['opportunity64','development64']:
  for j,r in enumerate(json.loads((root/split/'RESULTS.json').read_text())['rows']):sources.append((split,r.get('method',r.get('policy')),j,r))
 for f in sorted(root.glob('ppo_*/rollout_*.json')):sources.append(('online_training',f.parent.name,int(f.stem.split('_')[1]),json.loads(f.read_text())))
 for split,method,batch,r in sources:
  rows.append(dict(split=split,method=method,batch=batch,n=len(r['seeds']),physical_interactions=r['physical_interactions'],batch_wall_s=r['wall_s'],planning_wall_s=r['planning_wall_s'],logical_planning_requests=r['planner_calls'],plan_cache_hits=r.get('plan_cache_hits','NOT_RECORDED_IN_SUMMARY'),RRT_fallback_calls=r.get('rrt_fallback_calls','NOT_RECORDED_IN_SUMMARY'),native_physical_seconds_per_episode=12.5,wall_is_not_algorithm_speed_comparison=True))
  for i,seed in enumerate(r['seeds']):
   ds=r['decisions'][i];A=next((b for b in r['boundaries'][i] if b['stage']==0),{});C=next((b for b in r['boundaries'][i] if b['stage']==1),{});timings.append(dict(split=split,method=method,seed=seed,decision_tokens=len(ds),A_verified=A.get('valid',False),A_boundary_step=A.get('step',''),A_verified_physical_duration_s=A.get('physical_steps',0)*.05 if A.get('valid') else '',C_verified=C.get('valid',False),C_verified_logical_duration_s=C.get('physical_steps',0)*.05 if C.get('valid') else '',last_option_duration_including_terminal_hold_s=(250-ds[-1]['step'])*.05,logical_finish_s=r['logical_finish_steps'][i]*.05,actual_native_episode_s=12.5))
 for name,data in [('PERFORMANCE.csv',rows),('PHYSICAL_BOUNDARY_TIMINGS.csv',timings)]:
  with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 print(f'{len(rows)} physical batches, {len(timings)} episodes; no new physics')
if __name__=='__main__':main()
