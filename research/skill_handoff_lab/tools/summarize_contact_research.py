"""Recompute final native outcomes, paired contrasts and real budgets from source files."""
import argparse,json,gzip,hashlib,collections,math,csv,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'analysis';out.mkdir(exist_ok=True);groups={};summaries=[];pairs=[]
for folder in ['eval_offline','eval_online128','eval_online256','eval_matched_prefix']:
 path=r/folder/'RESULTS.json'
 if not path.exists():continue
 rows=json.loads(path.read_text());by=collections.defaultdict(list)
 for z in rows:by[z['method']].append(z)
 groups[folder]=by
 for method,rs in by.items():
  summaries.append(dict(cohort=folder,method=method,n=len(rs),success=sum(x['success']for x in rs),physical_steps=sum(x['steps']for x in rs),planner_queries=sum(x['planner_queries']for x in rs),planner_wall_s=sum(x['query_wall_s']for x in rs),median_episode_wall_s=float(np.median([x['wall_s']for x in rs])),model_selection_wall_s=sum(y['selection_wall_s']for x in rs for y in x['selection']),errors=dict(collections.Counter(x['error']or'NONE'for x in rs))))
 if 'ordinary'in by and 'conditional'in by:
  left={x['seed']:x for x in by['ordinary']};right={x['seed']:x for x in by['conditional']};keys=sorted(left.keys()&right.keys());rescue=sum(not left[k]['success']and right[k]['success']for k in keys);regress=sum(left[k]['success']and not right[k]['success']for k in keys);n=rescue+regress;pval=1. if not n else min(1.,2*sum(math.comb(n,k)for k in range(min(rescue,regress)+1))/2**n)
  detail=dict(cohort=folder,matched_initial_scenes=len(keys),same_initial=all(left[k]['initial_sha256']==right[k]['initial_sha256']for k in keys),candidate_rescues=rescue,candidate_regressions=regress,exact_mcnemar_p=pval)
  if folder=='eval_matched_prefix':
   checks=[]
   for seed in keys:
    ld=json.load(gzip.open(left[seed]['path'],'rt'));rd=json.load(gzip.open(right[seed]['path'],'rt'));at=left[seed]['contact_step']
    if at is not None:checks.append(dict(seed=seed,steps=at,same_prefix=ld['steps'][:at]==rd['steps'][:at]))
   detail['prefix_checks']=checks;assert all(x['same_prefix']for x in checks)
  pairs.append(detail)
allrows={}
for path in r.rglob('RESULTS.json'):
 if path.relative_to(r).parts[0]in ['analysis','video_rescue','video_regression','ORIGINAL_METADATA']:continue
 data=json.loads(path.read_text())
 if not isinstance(data,list):continue
 for z in data:
  if 'path'in z and 'steps'in z:allrows[z['path']]=z
for path,z in allrows.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==z['sha256'],path
partials=[]
for path in r.glob('*/PARTIAL.json.gz'):
 d=json.load(gzip.open(path,'rt'));partials.append(dict(path=str(path),steps=len(d['steps']),seed=d['seed']))
models=[]
for folder,name in [('ordinary','INTERRUPTED.zip'),('sac_ordinary','SAC_256.zip'),('sac_conditional','SAC_256.zip'),('sac_global','OFFLINE.zip'),('sac_generic_aux','OFFLINE.zip')]:
 path=r/folder/name;d=json.loads(zipfile.ZipFile(path).read('data'));models.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),online_decisions=d['num_timesteps'],sac_gradient_iterations=d['_n_updates'],actor_optimizer_steps=d['_n_updates'],critic_optimizer_steps=d['_n_updates']))
budget=dict(unique_completed_physical_episodes=len(allrows),completed_control_steps=sum(x['steps']for x in allrows.values()),saved_partial_control_steps=sum(x['steps']for x in partials),partial_episodes=partials,planner_queries=sum(x['planner_queries']for x in allrows.values()),planner_wall_s=sum(x['query_wall_s']for x in allrows.values()),SAC_gradient_iterations=sum(x['sac_gradient_iterations']for x in models),SAC_optimizer_step_calls=sum(x['actor_optimizer_steps']+x['critic_optimizer_steps']for x in models),joint_comparator_optimizer_steps=json.load(open(r/'comparators/TRAINING.json'))['optimizer_steps'],effect_model_fitting='ridge linear solves, not neural optimizer steps',shared_offline_data_episodes=190,shared_offline_replay_transitions=320,shared_data_counting='counted once globally; charged to each learning method as shared190episodes/19000steps',models=models)
result=dict(claim='UNVERIFIED_INDEPENDENT_ALGORITHM_GAIN',task='official PullCubeTool-v1',native_horizon=100,development_seeds=list(range(748100,748108)),sealed=False,results=summaries,pairs=pairs,budget=budget)
(out/'SUMMARY.json').write_text(json.dumps(result,indent=2));fields=['cohort','method','n','success','physical_steps','planner_queries','planner_wall_s','median_episode_wall_s','model_selection_wall_s']
with(out/'RESULTS.csv').open('w')as f:
 w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows({k:z[k]for k in fields}for z in summaries)
print(json.dumps({k:budget[k]for k in ['unique_completed_physical_episodes','completed_control_steps','saved_partial_control_steps','SAC_gradient_iterations','joint_comparator_optimizer_steps']}));print([(x['cohort'],x['method'],x['success'],x['n'])for x in summaries])
