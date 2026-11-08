"""Mature second-task physical coverage; parameter union is not deployed learning."""
import argparse,json,gzip,numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();folder=a.root/'charger_design';rows=json.loads((folder/'RESULTS.json').read_text());assert len(rows)==72
by={(r['seed'],r['angle'],r['policy']):r for r in rows};seeds=sorted({r['seed']for r in rows});groups={};checks=[]
for angle in [15,30,45]:
 for policy in ['nominal','feedback']:
  rr=[by[s,angle,policy]for s in seeds];base={s:by[s,15,'nominal']['success']for s in seeds};groups[f'{angle}_{policy}']=dict(episodes=len(rr),success=sum(r['success']for r in rr),rescue=[r['seed']for r in rr if r['success']and not base[r['seed']]],regression=[r['seed']for r in rr if not r['success']and base[r['seed']]],physical_steps=sum(r['steps']for r in rr),planner_calls=sum(r['plan_calls']for r in rr),planning_seconds=sum(r['planning_wall_s']for r in rr),successor_decisions=sum(r['decisions']for r in rr))
 for seed in seeds:
  left=json.load(gzip.open(by[seed,angle,'nominal']['path'],'rt'));right=json.load(gzip.open(by[seed,angle,'feedback']['path'],'rt'));n=left['summary']['prefix_steps'];same_n=n==right['summary']['prefix_steps'];err=max((float(np.max(np.abs(np.asarray(x[k])-np.asarray(y[k]))))for x,y in zip(left['steps'][:n],right['steps'][:n])for k in ['q','qvel','tcp','tool','cube']),default=0.)
  checks.append(dict(seed=seed,angle=angle,prefix_steps=n,same_length=same_n,max_recorded_prefix_error=err,paired_prefix_within1e5=same_n and err<=1e-5))
retake_path=a.root/'charger_prefix_retakes/RESULTS.json'
retakes=json.loads(retake_path.read_text())if retake_path.exists()else[];rb={(r['seed'],r['angle']):r for r in retakes};paired_groups={}
for angle in [15,30,45]:
 valid=[]
 for check in checks:
  if check['angle']!=angle:continue
  key=(check['seed'],angle)
  if check['paired_prefix_within1e5']:valid.append(by[key[0],angle,'feedback'])
  elif key in rb and rb[key]['prefix_replay']['max_physical_error']<=1e-5:valid.append(rb[key])
 paired_groups[str(angle)]=dict(valid_pairs=len(valid),nominal_success=sum(by[r['seed'],angle,'nominal']['success']for r in valid),feedback_success=sum(r['success']for r in valid),rescue=[r['seed']for r in valid if r['success']and not by[r['seed'],angle,'nominal']['success']],regression=[r['seed']for r in valid if not r['success']and by[r['seed'],angle,'nominal']['success']])
repeat_path=a.root/'charger_crossover_repeats/RESULTS.json'
repeats=json.loads(repeat_path.read_text())if repeat_path.exists()else[]
probe_path=a.root/'charger_successor_probe/RESULTS.json'
probe=json.loads(probe_path.read_text())if probe_path.exists()else[]
result=dict(successor_probe=[dict(seed=r['seed'],angle=r['angle'],policy=r['policy'],success=r['success'],error=r['error'],prefix_error=r['prefix_replay']['max_physical_error'])for r in probe],crossover_repeats=[dict(seed=r['seed'],angle=r['angle'],repeat=r['repeat'],success=r['success'],error=r['error'],prefix_steps=r['prefix_steps'])for r in repeats],paired_groups=paired_groups,retake_episodes=len(retakes),initial_sha_variants_per_seed={str(seed):len({r['initial_sha256']for r in rows if r['seed']==seed})for seed in seeds},groups=groups,natural_scenes=len(seeds),native_horizon=200,finite_union=sum(any(r['success']for r in rows if r['seed']==s)for s in seeds),prefix_checks=checks,all_prefixes_match=all(x['paired_prefix_within1e5']for x in checks),optimizer_updates=0,scope='New physical design only. Native physics unchanged; no learned transfer. Original independent replans are not automatically paired. paired_groups uses only matched records or physical command-prefix reexecution; not an assertion about every hidden PhysX variable. Different grasp angles are real complete predecessor interventions from original native resets.')
(a.root/'SECOND_TASK_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!='prefix_checks'},indent=2))
