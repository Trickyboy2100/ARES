"""Source-backed CPU analysis of true two-boundary native100 tool episodes."""
import argparse,csv,gzip,hashlib,itertools,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];raws={};checks=[]
 for path in sorted(a.root.glob('*/RESULTS.json')):
  r=json.loads(path.read_text())[0];trace=Path(r['path']);d=json.load(gzip.open(trace,'rt'));raws[path.parent.name]=d
  dec=d['decisions'];pre=next((x for x in d['skill_results'] if x['skill']=='PICK_TOOL'),None);boundary=dec[1]['boundary_step'] if len(dec)>1 else None
  search=next((x for x in d['plans'] if x['phase']=='SEARCH'),{});cs=search.get('candidates',[]);valid=[x for x in cs if x['valid']]
  selected=dec[1]['parameters'] if len(dec)>1 else {};finish=d['steps'][-1]
  matching=[c for c in cs if abs(c['angle']-selected.get('tool_yaw_rad',999))<1e-8 and abs(c['hook_y']-selected.get('hook_y_m',999))<1e-8];chosen=matching[0] if matching else {}
  predicted_feasible=bool(chosen.get('valid',False) and boundary is not None and chosen.get('predicted_steps',10000)<=100-boundary)
  row=dict(run=path.parent.name,seed=r['seed'],label=r['label'],success=r['success'],error=r['error'],steps=r['steps'],
   decisions=len(dec),initial_sha256=r['initial_sha256'],grasp_x_m=dec[0]['parameters']['grasp_x_m'] if dec else None,
   selected_hook_y_m=selected.get('hook_y_m'),selected_yaw_rad=selected.get('tool_yaw_rad'),
   selected_pull_distance_m=selected.get('pull_distance_m',.27),pull_policy=selected.get('pull_policy','fixed'),
   predecessor_success=pre['physical_success'] if pre else False,boundary_step=boundary,
   selected_predicted_budget_feasible=predicted_feasible,selected_predicted_steps=chosen.get('predicted_steps'),
   outcome_class='SUCCESS' if r['success'] else ('PREDECESSOR_FAILURE' if not pre or not pre['physical_success'] else ('PREFIX_SUCCESS_PREDICTED_SUFFIX_INFEASIBLE' if not predicted_feasible else 'PREFIX_SUCCESS_PREDICTED_FEASIBLE_EXECUTION_FAILURE')),
   actual_lift_m=pre['measurement']['tool'][2]-d['initial_measure']['tool'][2] if pre else None,
   attachment_x_m=pre['measurement']['tcp_T_tool'][0] if pre else None,
   boundary_tool_speed_m_s=float(np.linalg.norm(pre['measurement']['tool_linear_velocity'])) if pre else None,
   kinematic_candidates=len(valid),predicted_budget_feasible_candidates=sum(c['predicted_steps']<=100-boundary for c in valid) if boundary is not None else 0,
   shortest_predicted_suffix_steps=min((c['predicted_steps'] for c in valid),default=None),
   final_cube_base_distance_m=finish['cube_base_distance_m'],cube_displacement_m=float(np.linalg.norm(np.array(finish['cube'][:3])-d['initial_measure']['cube'][:3])),
   max_tool_cube_force_N=max(s['tool_cube_force_N'] for s in d['steps']),
   pull_steps=sum(s['phase']=='PULL' for s in d['steps']),wall_s=r['wall_s'],planning_wall_s=r['planning_wall_s'],source_path=str(trace),sha256=r['sha256'])
  rows.append(row);checks.append(dict(run=row['run'],sha=hashlib.sha256(trace.read_bytes()).hexdigest()==r['sha256'],native_horizon=len(d['steps'])==100,
   native_terminal=bool(finish['native_success'])==bool(r['success']),two_real_boundaries=len(dec)==2 and dec[0]['boundary_step']==0 and dec[1]['boundary_step']>0))
 pairs=[]
 for l,r in itertools.combinations(rows,2):
  if l['seed']!=r['seed'] or l['grasp_x_m']!=r['grasp_x_m'] or l['boundary_step'] is None or r['boundary_step'] is None:continue
  lm=raws[l['run']]['decisions'][1]['observation'];rm=raws[r['run']]['decisions'][1]['observation'];fields=['q','qvel','tcp','tool','cube','tcp_T_tool','tool_linear_velocity','tool_angular_velocity']
  delta={k:float(np.max(np.abs(np.array(lm[k])-np.array(rm[k])))) for k in fields}
  pairs.append(dict(left=l['run'],right=r['run'],exact_prefix_boundary=all(x==0 for x in delta.values()) and l['boundary_step']==r['boundary_step'],difference=delta))
 summary=dict(episodes=len(rows),physical_steps=sum(r['steps'] for r in rows),optimizer_updates=0,scene_seeds=sorted({r['seed'] for r in rows}),
  initial_hash_consistent=all(len({r['initial_sha256'] for r in rows if r['seed']==s})==1 for s in {r['seed'] for r in rows}),
  strong_baseline=[{k:r[k] for k in ['seed','success','error','grasp_x_m','selected_hook_y_m']} for r in rows if r['label']=='lookahead'],
  two_boundary_episodes=sum(c['two_real_boundaries'] for c in checks),scope='PHYSICAL_PARAMETER_INTERVENTION_NOT_LEARNED_METHOD',
  note='Kinematic/budget predictions are not contact success labels. Native goal cube/base distance <0.6m, horizon100.')
 policies={}
 for row in rows:
  policies.setdefault(row['label'],[]).append(row)
 summary['outcome_classes']={k:sum(r['outcome_class']==k for r in rows) for k in sorted({r['outcome_class'] for r in rows})}
 summary['selected_contract_false_positive']=sum(r['selected_predicted_budget_feasible'] and not r['success'] for r in rows)
 summary['selected_contract_false_negative']=sum(not r['selected_predicted_budget_feasible'] and r['success'] for r in rows)
 summary['policy_outcomes']={k:dict(episodes=len(v),successes=sum(x['success'] for x in v),seeds=[x['seed'] for x in v],mean_distance_m=float(np.mean([x['final_cube_base_distance_m'] for x in v]))) for k,v in policies.items()}
 comparisons=[]
 for left,right in itertools.combinations(sorted(policies),2):
  lm={r['seed']:r for r in policies[left]};rm={r['seed']:r for r in policies[right]};shared=sorted(lm.keys()&rm.keys())
  if not shared:continue
  comparisons.append(dict(left=left,right=right,seeds=shared,left_only_success=sum(lm[x]['success'] and not rm[x]['success'] for x in shared),right_only_success=sum(rm[x]['success'] and not lm[x]['success'] for x in shared),both_success=sum(lm[x]['success'] and rm[x]['success'] for x in shared),both_failure=sum(not lm[x]['success'] and not rm[x]['success'] for x in shared),same_initial=all(lm[x]['initial_sha256']==rm[x]['initial_sha256'] for x in shared)))
 def diagnostic(items):
  pos=[100-r['boundary_step']-r['selected_predicted_steps'] for r in items if r['success'] and r['selected_predicted_steps'] is not None]
  neg=[100-r['boundary_step']-r['selected_predicted_steps'] for r in items if not r['success'] and r['selected_predicted_steps'] is not None]
  auc=float(np.mean([float(x>y)+.5*float(x==y) for x in pos for y in neg])) if pos and neg else None
  return dict(episodes=len(items),successes=sum(r['success'] for r in items),TP=sum(r['success'] and r['selected_predicted_budget_feasible'] for r in items),FP=sum(not r['success'] and r['selected_predicted_budget_feasible'] for r in items),FN=sum(r['success'] and not r['selected_predicted_budget_feasible'] for r in items),TN=sum(not r['success'] and not r['selected_predicted_budget_feasible'] for r in items),budget_margin_in_panel_AUC=auc)
 unique={}
 for r in rows:
  key=(r['seed'],r['grasp_x_m'],r['selected_hook_y_m'],r['selected_yaw_rad'],r['selected_pull_distance_m'])
  unique.setdefault(key,r)
 contract=dict(implemented_diagnostic='selected kinematic and remaining-step margin as naive physical-value proxy',proposed_distribution_or_relabeling_method='PROPOSED_NOT_IMPLEMENTED_NOT_TESTED',all_executed_episodes=diagnostic(rows),unique_seed_and_applied_parameters=diagnostic(list(unique.values())),scene_draws=len({r['seed'] for r in rows}),scope='Descriptive only, no fitting or held-out claim. Several goal rules apply exactly the same parameters as fixed250; repeated physical trials counted in budget but deduplicated in second diagnostic. Feasibility is not a probability; no probability calibration asserted.',observable_state='Actual TCP-relative attachment derives from current TCP/tool poses already in full state; actual tool velocity is observed. No POMDP or privileged posterior-information claim.')
 for name,data in [('SUMMARY',summary),('EPISODES',rows),('QA',checks),('SUFFIX_PREFIX_PAIRS',pairs),('PAIRED_POLICY_OUTCOMES',comparisons),('CONTRACT_FALSIFIER',contract)]:
  (a.out/f'{name}.json').write_text(json.dumps(data,indent=2))
 if rows:
  with (a.out/'EPISODES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,max(3,len(rows)*.38)));y=np.arange(len(rows));ax.barh(y,[r['final_cube_base_distance_m'] for r in rows],color=['#397d76' if r['success'] else '#ba7848' for r in rows]);ax.axvline(.6,color='black',linestyle='--',label='Native success threshold');ax.set_yticks(y,[r['run'] for r in rows]);ax.invert_yaxis();ax.set_xlabel('Final cube-to-base XY distance (m)');ax.set_title('Native100 PickTool → UseTool | actual physics, no RL claim');ax.legend();fig.tight_layout();fig.savefig(a.out/'COMPLETE_TASK.png',dpi=200);fig.savefig(a.out/'COMPLETE_TASK.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4))
 for seed in sorted({r['seed'] for r in rows}):
  rs=[r for r in rows if r['seed']==seed and r['boundary_step'] is not None];ax.scatter([r['attachment_x_m']*1000 for r in rs],[100-r['boundary_step'] for r in rs],label=str(seed))
 ax.set_xlabel('Measured TCP-relative tool X after physical lift (mm)');ax.set_ylabel('Remaining physical steps at second decision');ax.set_title('Predecessor parameter changes attachment and successor budget');ax.legend();fig.tight_layout();fig.savefig(a.out/'SKILLRESULT_BUDGET.png',dpi=200);fig.savefig(a.out/'SKILLRESULT_BUDGET.svg');plt.close(fig)
 print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
