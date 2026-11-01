"""CPU evidence aggregation for EXP-003 A; live runs never counted as final costs."""
import argparse,csv,gzip,hashlib,itertools,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def flag(x):return bool(np.asarray(x).reshape(-1)[0])
def read(p):return json.loads(p.read_text())
def write(out,name,data):(out/name).write_text(json.dumps(data,indent=2,allow_nan=False))
def scalar(v):return float(v) if v is not None else None

def scene_cluster_interval(details, bootstrap_seed=20261101, replicates=20000):
 """Exploratory percentile CI; resample whole scenes, never individual bias cases."""
 clusters=defaultdict(list)
 for row in details:
  if row['same_initial_sha256']:clusters[row['seed']].append(row)
 result=dict(method='paired scene-cluster percentile bootstrap',direction='right success rate minus left success rate',bootstrap_seed=bootstrap_seed,replicates=replicates,confidence=.95,independent_scene_clusters=len(clusters),engineering_reused_scene=732000,engineering_scene_present=732000 in clusters,interpretation='EXPLORATORY ONLY: primary design has six independent scenes, one used for stop engineering. Thirty-six conditions are not 36 independent samples. This interval is not final statistical evidence or a novelty claim.')
 signatures=[sorted((r['family'],tuple(r['bias_mm'])) for r in rows) for rows in clusters.values()]
 balanced=bool(signatures) and all(x==signatures[0] for x in signatures) and len(signatures[0])==6 and len(set(signatures[0]))==6
 result['six_conditions_per_scene_complete']=balanced
 result['clusters']=[dict(seed=seed,conditions=len(rows),right_minus_left_successes=sum(int(r['right_success'])-int(r['left_success']) for r in rows)) for seed,rows in sorted(clusters.items())]
 if not balanced or len(clusters)<2:
  result.update(status='NOT_ESTIMATED_INCOMPLETE_OR_TOO_FEW_SCENES',difference=None,percentile_95_interval=None)
  return result
 means=np.asarray([x['right_minus_left_successes']/x['conditions'] for x in result['clusters']],dtype=float)
 rng=np.random.default_rng(bootstrap_seed)
 draws=means[rng.integers(0,len(means),size=(replicates,len(means)))].mean(axis=1)
 result.update(status='EXPLORATORY_ESTIMATED',difference=float(means.mean()),percentile_95_interval=np.quantile(draws,[.025,.975]).tolist(),unit='success-rate fraction',weighting='equal scene weight; all six matched conditions retained together per sampled scene',limitation='Percentile interval can be degenerate when all observed scene differences agree; it does not establish zero uncertainty beyond these scenes.')
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 runs=[];episodes=[];raw_index=[];warnings=[];boundaries={};training=[]
 for folder in sorted(a.root.iterdir()):
  if not folder.is_dir() or not (folder/'MANIFEST.json').exists():continue
  try:manifest=read(folder/'MANIFEST.json')
  except (OSError,json.JSONDecodeError) as exc:warnings.append(dict(run=folder.name,error=str(exc),status='UNREADABLE_MANIFEST'));continue
  args=manifest['args'];result=None
  if (folder/'RESULTS.json').exists():
   try:result=read(folder/'RESULTS.json')
   except (OSError,json.JSONDecodeError) as exc:warnings.append(dict(run=folder.name,error=str(exc),status='RESULT_WRITE_IN_PROGRESS'))
  status='FINAL_ERROR' if result and result.get('error') else 'FINAL' if result else 'UNFINISHED_NO_FINAL_COUNTERS'
  index=[]
  if (folder/'EPISODES.json').exists():
   try:index=read(folder/'EPISODES.json')
   except (OSError,json.JSONDecodeError) as exc:warnings.append(dict(run=folder.name,error=str(exc),status='EPISODE_INDEX_WRITE_IN_PROGRESS'))
  source_qa={n:(folder/n).exists() and sha(folder/n)==digest for n,digest in manifest.get('sources',{}).items()}
  if not all(source_qa.values()):warnings.append(dict(run=folder.name,status='SOURCE_HASH_MISMATCH',checks=source_qa))
  run=dict(run=folder.name,status=status,path=str(folder),mode=args['mode'],authority=args['authority'],completion=args['completion'],training_seed=args.get('seed'),requested_steps=args.get('steps'),sources=manifest.get('sources',{}),source_hash_checks=source_qa,final_counters=result,recorded_complete_episode_lower_bound=0,recorded_physics_lower_bound=0,excluded_from_final_costs=result is None)
  local=[];seen=set()
  for row in index:
   path=Path(row['path'])
   if str(path) in seen:warnings.append(dict(run=folder.name,status='DUPLICATE_INDEX_PATH',path=str(path)));continue
   seen.add(str(path))
   try:
    actual=sha(path)
    if actual!=row['sha256']:raise ValueError('trajectory SHA mismatch')
    with gzip.open(path,'rt') as f:t=json.load(f)
   except (OSError,ValueError,EOFError) as exc:warnings.append(dict(run=folder.name,status='TRACE_INVALID',path=str(path),error=str(exc)));continue
   steps=t['steps'];native=[flag(s.get('native_success',False)) for s in steps];success=flag(row['success']);hold=row.get('hold_entry');hold_valid=hold is not None and 0<int(hold)<=len(steps)
   final_agrees=bool(native and native[-1]==success);complete=len(steps)==200 and row['steps']==200
   if not final_agrees or not complete:warnings.append(dict(run=folder.name,path=str(path),status='NATIVE_TERMINAL_OR_HORIZON_MISMATCH'))
   policy=row.get('policy','model' if args['mode']=='eval' else 'training' if args['mode']=='train' else 'UNKNOWN')
   group=folder.name+'::'+policy
   e=dict(run=folder.name,group=group,run_status=status,mode=args['mode'],authority=row.get('authority',args['authority']),family=row.get('task_family','UNKNOWN'),completion=args['completion'],policy=policy,seed=row['seed'],bias_mm=row['bias_mm'],initial_sha256=row['initial_sha256'],success=success,ever_success=any(native),transient_success_lost=any(native) and not success,hold_entry=hold,hold_entered=hold_valid,hold_entry_native_success=native[int(hold)-1] if hold_valid else None,hold_final_failure=hold_valid and not success,hold_entered_before_native=hold_valid and not native[int(hold)-1],hold_confirmed_and_final_success=hold_valid and native[int(hold)-1] and success,hold_then_lost=hold_valid and native[int(hold)-1] and not success,steps=len(steps),prefix_steps=row['prefix_steps'],decisions=row.get('decisions',len(t['transitions'])),settling_probe_steps=row.get('settling_probe_steps',0),truncated_tail=bool(row.get('truncated_training_tail',False)),completion_reason=row.get('completion_reason'),error=row.get('error'),path=str(path),sha256=actual,trace_hash_pass=True,terminal_check_pass=final_agrees,full_horizon_pass=complete,first_native_success_step=next((i+1 for i,x in enumerate(native) if x),None))
   if args['mode']=='controls' and policy=='UNKNOWN':e['comparison_excluded']='control interrupted before policy label recorded'
   boundaries[str(path)]=t.get('prefix_boundary');episodes.append(e);local.append(e)
  run['expected_evaluation_episodes']=(len(manifest.get('seeds',[]))*len(args.get('conditions','0,1,2,3,4,5').split(','))*(len(args.get('policies','geometry,search,rls').split(',')) if args['mode']=='controls' else 1)) if args['mode']!='train' else None
  run['evaluation_coverage_complete']=(len(local)==run['expected_evaluation_episodes'] and status=='FINAL') if args['mode']!='train' else None
  run['recorded_complete_episode_lower_bound']=len(local);run['recorded_physics_lower_bound']=sum(e['steps'] for e in local)
  if result:
   run['counter_physics_matches_index']=result.get('physics_steps')==run['recorded_physics_lower_bound'];run['counter_episode_matches_index']=result.get('complete_episodes')==len(local)
   if not run['counter_physics_matches_index'] or not run['counter_episode_matches_index']:warnings.append(dict(run=folder.name,status='FINAL_COUNTER_INDEX_DISAGREEMENT'))
  else:run['unknown_live_or_crash_cost']='Current in-progress episode and optimizer updates unavailable; recorded episodes are only a lower bound.'
  run['unindexed_trajectory_files']=[str(f) for f in folder.rglob('*.json.gz') if str(f) not in seen]
  if run['unindexed_trajectory_files']:warnings.append(dict(run=folder.name,status='UNINDEXED_TRAJECTORIES_NOT_COUNTED',files=run['unindexed_trajectory_files']))
  runs.append(run)
  if args['mode']=='train':
   cumulative_physics=0;cumulative_decisions=0
   for i,e in enumerate(local):
    cumulative_physics+=e['steps'];cumulative_decisions+=e['decisions'];window=[x for x in local[max(0,i-19):i+1] if not x['truncated_tail']]
    training.append(dict(run=folder.name,episode_index=i+1,physics_steps=cumulative_physics,skill_decisions=cumulative_decisions,window_episodes=len(window),window_success=sum(x['success'] for x in window),window_success_rate=sum(x['success'] for x in window)/len(window) if window else None,interpretation='online exploration training episodes; NOT validation/convergence'))
  for f in sorted(folder.rglob('*')):
   if f.is_file() and '__pycache__' not in f.parts and f.suffix in ['.json','.gz','.py','.zip','.pkl']:
    try:raw_index.append(dict(run=folder.name,path=str(f),bytes=f.stat().st_size,sha256=sha(f),live_file_may_change=result is None))
    except OSError as exc:warnings.append(dict(run=folder.name,path=str(f),status='FILE_CHANGED_DURING_SCAN',error=str(exc)))
 # Preserve every historical run separately. Only final non-smoke rows form evaluation comparisons.
 groups=defaultdict(list)
 for e in episodes:
  if e['mode'] in ['eval','controls'] and e['run_status']=='FINAL' and not e['run'].startswith('smoke') and not e['truncated_tail'] and not e.get('comparison_excluded'):groups[e['group']].append(e)
 scores=[]
 for name,es in sorted(groups.items()):
  for family in ['ALL']+sorted({e['family'] for e in es}):
   subset=[e for e in es if family=='ALL' or e['family']==family]
   scores.append(dict(group=name,authority=es[0]['authority'],completion=es[0]['completion'],family=family,episodes=len(subset),distinct_scene_seeds=len({e['seed'] for e in subset}),success=sum(e['success'] for e in subset),ever_success=sum(e['ever_success'] for e in subset),transient_success_lost=sum(e['transient_success_lost'] for e in subset),hold_entered=sum(e['hold_entered'] for e in subset),hold_entry_native_success=sum(e['hold_entry_native_success'] is True for e in subset),hold_entered_before_native=sum(e['hold_entered_before_native'] for e in subset),hold_final_failure=sum(e['hold_final_failure'] for e in subset)))
 pairs=[]
 for left,right in itertools.combinations(sorted(groups),2):
  def key(e):return (e['seed'],tuple(e['bias_mm']),e['family'])
  ll={key(e):e for e in groups[left]};rr={key(e):e for e in groups[right]};details=[]
  for k in sorted(ll.keys()&rr.keys()):
   x,y=ll[k],rr[k];same=x['initial_sha256']==y['initial_sha256'];bx,by=boundaries[x['path']],boundaries[y['path']];eq=None
   if bx and by:
    fields=['q','qvel','tcp','tool','cube','tool_linear_velocity','tool_angular_velocity','cube_linear_velocity'];eq=all(np.array_equal(np.asarray(bx.get(f)),np.asarray(by.get(f))) for f in fields)
   details.append(dict(seed=k[0],bias_mm=list(k[1]),family=k[2],same_initial_sha256=same,recorded_prefix_equal=eq,left_success=x['success'],right_success=y['success']))
  valid=[x for x in details if x['same_initial_sha256']]
  if details:pairs.append(dict(left=left,right=right,matched_conditions=len(valid),excluded_initial_mismatches=len(details)-len(valid),recorded_prefix_equal=sum(x['recorded_prefix_equal'] is True for x in valid),right_rescues=sum(not x['left_success'] and x['right_success'] for x in valid),right_regressions=sum(x['left_success'] and not x['right_success'] for x in valid),scene_cluster_bootstrap=scene_cluster_interval(valid),scene_cluster_bootstrap_excluding_engineering_scene=scene_cluster_interval([x for x in valid if x['seed']!=732000]),details=details,limitation='Matches initial SHA; prefix equality reported separately. Across completion modes this is explicit shared-controller intervention, not algorithm-only effect.'))
 # Frozen checkpoint evaluations share actual conditions; never infer performance from training windows.
 checkpoint_evals=[]
 for run in runs:
  if run['mode']!='eval' or run['status']!='FINAL':continue
  manifest=read(Path(run['path'])/'MANIFEST.json');model=Path(manifest['args']['model']);train_manifest_path=model.parent/'MANIFEST.json'
  train_manifest=read(train_manifest_path) if train_manifest_path.exists() else {}
  if model.stem.startswith('CHECKPOINT_'):
   try:decisions=int(model.stem.split('_')[-1])
   except ValueError:decisions=None
  elif (model.parent/'RESULTS.json').exists():decisions=read(model.parent/'RESULTS.json').get('policy_decisions')
  else:decisions=None
  es=[e for e in episodes if e['run']==run['run'] and not e['truncated_tail']]
  sensitivity=[e for e in es if e['seed']!=732000]
  checkpoint_evals.append(dict(run=run['run'],checkpoint=str(model),checkpoint_sha256=sha(model) if model.exists() else None,training_decisions=decisions,training_authority=train_manifest.get('args',{}).get('authority'),evaluation_authority=run['authority'],completion=run['completion'],episodes=len(es),success=sum(e['success'] for e in es),condition_identity=sorted([dict(seed=e['seed'],bias_mm=e['bias_mm'],family=e['family'],initial_sha256=e['initial_sha256']) for e in es],key=lambda x:(x['seed'],x['family'],x['bias_mm'])),sensitivity_excluding_stop_development_seed732000=dict(episodes=len(sensitivity),success=sum(e['success'] for e in sensitivity),distinct_scenes=len({e['seed'] for e in sensitivity})),interpretation='Actual completed checkpoint physical evaluation, not training-window interpolation; whole36 dwell remains primary; legacy12 is a separate paired-stop diagnostic, not a learning-curve point; remove732000 only as prespecified engineering-reuse sensitivity.'))
 write(a.out,'CHECKPOINT_EVAL_CURVE.json',checkpoint_evals)
 expected={f'{mode}_{authority}':next((r['status'] for r in runs if r['run']==f'{mode}_{authority}'),'MISSING') for mode in ['train','eval'] for authority in ['full','y_only','task']}
 final_runs=[r for r in runs if r['final_counters'] is not None];unfinished=[r for r in runs if r['final_counters'] is None]
 totals={k:sum(r['final_counters'].get(k,0) for r in final_runs) for k in ['complete_episodes','physics_steps','policy_decisions','optimizer_updates','tail_hold_steps','tail_episodes']}
 summary=dict(oracle_boundary='Native success is allowed as terminal training reward and evaluation label; actor input/online stop use observed geometry, frame velocity and declared family, not true bias or native success.',native_success_contract='position<=.005m and angle<=.2rad at physical step200',observable_completion_contract='current source:4mm/.12rad enter,4.6mm/.16rad exit,20Hz pose-delta velocity<=.02m/s and .3rad/s,3 unique-step dwell; historical smoke source versions retained',claim='COMPONENT_BASELINE_NOT_MULTISKILL_METHOD',finalized_cost_including_failed_smokes=totals,final_run_count=len(final_runs),unfinished_runs_excluded_from_final_costs=[r['run'] for r in unfinished],unfinished_recorded_physics_lower_bound=sum(r['recorded_physics_lower_bound'] for r in unfinished),unfinished_optimizer_cost=None if unfinished else 0,expected_runs=expected,controls_expected='full geometry/search; all-authority original and fast deduplicated search; full RLS and legacy controls on two-scene diagnostic panel; absence is not zero cost',warnings=warnings,source_sha256=sha(Path(__file__)),completion_definitions={'ever_success':'any recorded native criterion true; auxiliary only','success':'native criterion on step200','hold_entered_before_native':'hold entered while native criterion false; may later become successful, so not automatically false termination','hold_final_failure':'observable hold entered but native final failure','hold_then_lost':'native success at hold entry but native final failure'})
 for name,data in [('SUMMARY.json',summary),('RUNS.json',runs),('EPISODES.json',episodes),('RAW_SHA_INDEX.json',raw_index),('EVAL_SCORES.json',scores),('PAIRS.json',pairs),('TRAINING_WINDOWS.json',training)]:write(a.out,name,data)
 with (a.out/'EVAL_EPISODES.csv').open('w') as f:
  fields=['run','authority','family','completion','policy','seed','bias_mm','success','ever_success','hold_entry','hold_entry_native_success','hold_final_failure','initial_sha256'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:e[k] for k in fields} for e in episodes if e['mode']!='train')
 text=['# Component A evidence snapshot','', 'CPU aggregation only: no new simulator interactions. Historical smoke versions remain separate; failed smoke costs are retained.','',f"Finalized recorded cost: {totals['complete_episodes']} episodes, {totals['physics_steps']} physical steps, {totals['optimizer_updates']} optimizer updates.",f"Unfinished excluded from final totals: {[r['run'] for r in unfinished]}; their indexed physical-step lower bound is {summary['unfinished_recorded_physics_lower_bound']}. Unrecorded current episodes and updates are unknown.",'','| Group | Family | Success | Distinct initial scenes |','|---|---|---:|---:|']
 text += [f"| {x['group']} | {x['family']} | {x['success']}/{x['episodes']} | {x['distinct_scene_seeds']} |" for x in scores]
 text += ['','Training windows describe collected exploratory training episodes only (SAC optimization is off-policy); they do not show validation convergence. Same scene with multiple biases is not an independent scene draw. Pair tables retain initial-SHA and physical-prefix checks. PAIRS.json adds exploratory paired scene-cluster bootstrap intervals (20,000 resamples; fixed seed 20261101): all six bias/family conditions move together. Only six independent scenes exist, one used for stop engineering; a separate five-scene sensitivity excludes it. These intervals are not final statistical proof. No novelty or generalization claim follows from this component comparison.','',f'Warnings requiring review: {len(warnings)}. Exact source/trace/checkpoint provenance is in RAW_SHA_INDEX.json.','']
 (a.out/'README.md').write_text('\n'.join(text))
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(8,4))
  for run in sorted({x['run'] for x in training}):
   ts=[x for x in training if x['run']==run and x['window_success_rate'] is not None];ax.plot([x['skill_decisions'] for x in ts],[x['window_success_rate'] for x in ts],label=run)
  ax.set(xlabel='Recorded training skill decisions',ylabel='Trailing20-episode training success',title='Training exploration windows — NOT validation convergence',ylim=(-.02,1.02))
  if training:ax.legend()
  fig.tight_layout();fig.savefig(a.out/'TRAINING_WINDOWS.png',dpi=170);plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4))
  for authority in sorted({x['evaluation_authority'] for x in checkpoint_evals}):
   items=sorted([x for x in checkpoint_evals if x['evaluation_authority']==authority and x['training_decisions'] is not None and x['episodes']==36 and x['completion']=='dwell'],key=lambda x:x['training_decisions'])
   for item in items:
    ax.scatter(item['training_decisions'],item['success']/item['episodes'],label=f"{item['run']} ({item['success']}/{item['episodes']})")
  ax.set(xlabel='Training decisions at saved checkpoint',ylabel='Actual development physical success',title='Shared dwell contract / same36 conditions — actual checkpoints',ylim=(-.02,1.02))
  if checkpoint_evals:ax.legend(fontsize=7)
  fig.tight_layout();fig.savefig(a.out/'CHECKPOINT_EVAL.png',dpi=170);plt.close(fig)
 except ImportError:warnings.append({'status':'MATPLOTLIB_UNAVAILABLE'})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
