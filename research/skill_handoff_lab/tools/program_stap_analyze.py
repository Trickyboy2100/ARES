"""Recompute all scores from archived trajectories, not printed summaries."""
import argparse,gzip,json,hashlib,collections,csv,math
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];traces={}
 for path in sorted(a.root.rglob('TRAJECTORY.json.gz')):
  d=json.load(gzip.open(path,'rt'));s=d['summary'];last=d['steps'][-1];native=bool(last['native_success']);target=s.get('target_y');distance=None if target is None else float(np.linalg.norm(np.array(last['cube'][:2])-[-.045,target]));success=native and (distance is None or distance<=.04);assert success==s.get('task_success',s['success']);assert len(d['steps'])==100
  group=path.relative_to(a.root).parts[0];r=dict(group=group,seed=s['seed'],mass=s['density_scale'],target=target,method=s.get('method','intervention'),grasp=d['decisions'][0]['parameters']['grasp_x_m'],hook=d['decisions'][1]['parameters']['hook_y_m'] if len(d['decisions'])>1 else None,native=native,success=success,distance=distance,steps=100,initial_sha=s['initial_sha256'],wall_s=s['wall_s'],selector_wall_s=s.get('selector_wall_s',0),queries=s.get('model_queries',0),error=s['error'],failure='SUCCESS' if success else 'PREFIX' if len(d['decisions'])<2 else 'BUDGET_'+last['phase'] if s['error'] and 'HORIZON' in s['error'] else 'SUCCESSOR_CONTACT_OR_GOAL',path=str(path.relative_to(a.root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  r['geometry_candidates']=sum(len(z.get('candidates',[])) for z in d['plans']);r['planner_wall_s']=sum(z.get('planning_wall_s',0) for z in d['plans'])
  logs=d.get('selector_logs',[]);r['critic_queries']=sum(sum(z['population'] for z in l['iterations'])*(2-l['stage']) if 'iterations' in l else l.get('skill_model_queries',0) for l in logs);r['learned_dynamics_queries']=sum(1+sum(z['population'] for z in l['iterations']) for l in logs if l['stage']==0 and 'iterations' in l);r['actor_queries']=sum(2-l['stage'] for l in logs if 'iterations' in l)
  rows.append(r);traces[r['path']]=d
 groups={g:dict(episodes=sum(x['group']==g for x in rows),success=sum(x['success'] for x in rows if x['group']==g),native=sum(x['native'] for x in rows if x['group']==g),failures=dict(collections.Counter(x['failure'] for x in rows if x['group']==g))) for g in sorted({x['group'] for x in rows})}
 evaluation=[x for x in rows if x['group']=='eval32'];scores={};pairs={};dynamics=[]
 key=lambda x:(x['seed'],x['mass'],x['target'])
 base={key(x):x for x in evaluation if x['method']=='fixed'}
 for m in sorted({x['method'] for x in evaluation}):
  rs=[x for x in evaluation if x['method']==m];scores[m]=dict(success=sum(x['success'] for x in rs),native=sum(x['native'] for x in rs),n=len(rs),queries=sum(x['queries'] for x in rs),selector_wall_s=sum(x['selector_wall_s'] for x in rs),wall_s=sum(x['wall_s'] for x in rs))
  assert all(x['initial_sha']==base[key(x)]['initial_sha'] for x in rs)
  rescue=sum(x['success'] and not base[key(x)]['success'] for x in rs);regress=sum(not x['success'] and base[key(x)]['success'] for x in rs);n=rescue+regress;pval=min(1.,2*sum(math.comb(n,k) for k in range(min(rescue,regress)+1))/2**n) if n else 1.;pairs[m]=dict(rescue=rescue,regression=regress,exact_mcnemar_p=pval)
  for x in rs:
   if m!='stap':continue
   d=traces[x['path']];logs=d['selector_logs'];post=d['skill_results'][0]['measurement'];pred=np.array(logs[0]['predicted_states'][1]);dynamics.append(dict(case=key(x),relative_translation_error_mm=float(np.linalg.norm(pred[28:31]-post['tcp_T_tool'][:3])*1000),tool_translation_error_mm=float(np.linalg.norm(pred[14:17]-post['tool'][:3])*1000),budget_error_steps=float(abs(pred[41]*100-d['skill_results'][0]['remaining_steps'])),predicted_product=logs[0]['p_success'],actual_success=x['success'],replanned_hook=logs[1]['normalized_action'],openloop_hook=logs[0]['planned_actions'][1][0]))
 ablations={}
 for group in ['cem_budget16','residual_paired16','absolute_paired16']:
  gr=[x for x in rows if x['group']==group];ablations[group]={m:dict(success=sum(x['success'] for x in gr if x['method']==m),n=sum(x['method']==m for x in gr),critic_queries=sum(x['critic_queries'] for x in gr if x['method']==m)) for m in sorted({x['method'] for x in gr})}
 result=dict(ablations=ablations,episodes=len(rows),control_steps=sum(x['steps'] for x in rows),groups=groups,evaluation=scores,paired_vs_fixed=pairs,stap_model_errors=dynamics,scope='Development exploration; one training seed; two training and two new scene seeds; no independent method claim; startup failure has zero recorded control steps')
 (a.out/'SUMMARY.json').write_text(json.dumps(result,indent=2));(a.out/'EPISODES.json').write_text(json.dumps(rows,indent=2))
 with (a.out/'EPISODES.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
 fig,ax=plt.subplots(figsize=(7,4));names=['fixed','fq','foresight','stap'];vals=[scores[n]['success'] for n in names];ax.bar(names,vals,color=['#4e657b','#b57c38','#668d68','#327fa4']);ax.set(ylim=(0,8.7),ylabel='Full-task successes / 8',title='New initial states: same two-skill action interface');[ax.text(i,v+.12,f'{v}/8',ha='center') for i,v in enumerate(vals)];fig.text(.02,.01,'DATA_DERIVED_PLOT | one training seed; two new scenes; no new RL claim',fontsize=9);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(a.out/'METHOD_RESULTS.png',dpi=300);fig.savefig(a.out/'METHOD_RESULTS.svg');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));ax.scatter([x['predicted_product'] for x in dynamics],[int(x['actual_success']) for x in dynamics]);ax.set(xlim=(0,1),ylim=(-.1,1.1),xlabel='STAP predicted Q product',ylabel='Actual terminal success',title='Model confidence is not physical success (8 cases)');fig.tight_layout();fig.savefig(a.out/'STAP_CALIBRATION.png',dpi=300);plt.close(fig)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
