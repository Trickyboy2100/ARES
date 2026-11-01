"""CPU full-task outcome and controller tracking diagnostics, not RL gains."""
import argparse,gzip,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);rows=[];initial={};raw={}
    for p in sorted(a.root.glob('*/TRAJECTORY.json.gz')):
        with gzip.open(p,'rt') as f:d=json.load(f)
        raw[p.parent.name]=d
        m=json.loads((p.parent/'MANIFEST.json').read_text());r=d['summary'];tr=d['steps'];initial.setdefault(r['seed'],set()).add(r['initial_sha256'])
        tracking={}
        for arm in ['0','1']:
            err=[np.abs(np.array(x['measurement'][arm]['q'][:7])-np.array(x['action']['panda_wristcam-'+arm][:7])) for x in tr]
            tracking[arm]=dict(max_abs_joint_tracking_error_rad=float(np.max(err)),median_abs_joint_tracking_error_rad=float(np.median(err)),max_abs_qvel_rad_s=float(np.max([np.abs(x['measurement'][arm]['qvel'][:7]) for x in tr])))
        phases={}
        for x in tr:
            v=phases.setdefault(x['phase'],dict(steps=0,peak_link_cube_force_N=0.));v['steps']+=1
            v['peak_link_cube_force_N']=max(v['peak_link_cube_force_N'],*[max(x['measurement'][i]['robot_cube_force_N'].values(),default=0) for i in ['0','1']])
        second=d['decisions'][1] if len(d['decisions'])>1 else None
        selected_cost=next((v for v in second.get('candidate_costs',[]) if v['axis']==second['parameters']['grasp_axis']),None) if second and second.get('selection_objective')=='full-suffix' else None
        first_success=next((x['step'] for x in tr if x['measurement']['native_success']),None)
        boundary=second['boundary_step'] if second else None
        last_motion=max([x['step'] for x in tr if x['phase']!='TERMINAL_HOLD'],default=0)
        prediction=dict(selection_objective=second.get('selection_objective') if second else None,predicted_suffix_steps=selected_cost['objective_steps'] if selected_cost else None,remaining_budget_at_boundary=100-boundary if boundary is not None else None,actual_suffix_steps_to_first_native_success=first_success-boundary if first_success is not None and boundary is not None else None,actual_executed_suffix_motion_steps=last_motion-boundary if boundary is not None else None,observed_suffix_steps=100-boundary if boundary is not None else None,success_time_censored=first_success is None)
        if first_success is not None and boundary is not None and selected_cost:prediction['predicted_minus_actual_success_steps']=selected_cost['objective_steps']-(first_success-boundary)
        row=dict(prediction=prediction,case=p.parent.name,seed=r['seed'],parameters=m['parameters'],resource_context=m.get('resource_context'),success=r['success'],error=r['error'],steps=len(tr),boundaries=[x['boundary_step'] for x in d['decisions']],skill_results=d['skill_results'],initial_sha256=r['initial_sha256'],final_goal_distance_m=r['final']['cube_goal_distance_m'],first_native_success_step=next((x['step'] for x in tr if x['measurement']['native_success']),None),tracking=tracking,phases=phases,wall_s=r['wall_s'],planning_wall_s=r['planning_wall_s'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        rows.append(row)
    pairs=[]
    paired_names=[(k,k+'_other') for k in raw if k.endswith('_full_p020') and k+'_other' in raw]
    for left,right in paired_names+[('742000_continuous_p020','742000_continuous_p080'),('742001_continuous_p020','742001_continuous_p080'),('742001_p020_aligned','742001_p020_aligned_y')]:
        if left not in raw or right not in raw:continue
        x,y=raw[left],raw[right];shared=0;states_exact=True
        for u,v in zip(x['steps'],y['steps']):
            if u['action']!=v['action']:break
            shared+=1;states_exact=states_exact and u['state']==v['state']
        boundary=x['decisions'][1]['boundary_step'] if len(x['decisions'])>1 else None
        pairs.append(dict(left=left,right=right,initial_exact=x['initial']==y['initial'],shared_action_prefix_steps=shared,shared_prefix_states_exact=states_exact,second_decision_boundary=boundary,same_second_boundary_state=boundary is not None and len(y['decisions'])>1 and y['decisions'][1]['boundary_step']==boundary and x['steps'][boundary-1]['state']==y['steps'][boundary-1]['state'],left_success=x['summary']['success'],right_success=y['summary']['success']))
    (a.out/'MODEL_DURATION_AUDIT.json').write_text(json.dumps([dict(case=r['case'],success=r['success'],error=r['error'],**r['prediction']) for r in rows],indent=2))
    (a.out/'PAIRED_PREFIX_AUDIT.json').write_text(json.dumps(pairs,indent=2))
    report=dict(episodes=len(rows),actual_physical_steps=sum(r['steps'] for r in rows),native_successes=sum(r['success'] for r in rows),optimizer_updates=0,same_seed_initial_states_exact={str(k):len(v)==1 for k,v in initial.items()},resource_context='SHARED_GPU_CONTENTION; timing not exclusive performance',rows=rows)
    (a.out/'ANALYSIS.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    lines=['# Native two-robot push → pick: physical controller scout','', 'All original failures are retained. Native100 horizon, initial distribution and25mm-plus-static success are unchanged. No RL training. Both robots act in the same continuing episode. Timing is SHARED_GPU_CONTENTION, not an exclusive benchmark.','', '| Case | Controller / stride | Decisions | Native success | Final error mm | Failure |','|---|---|---|---|---:|---|']
    for r in rows:lines.append(f"|{r['case']}|{r['parameters'].get('controller','original')} / {r['parameters']['path_stride']}|{r['boundaries']}|{r['success']}|{1000*r['final_goal_distance_m']:.3f}|{r['error']}|")
    lines+=['','The original full-TOPP controller exhausts its budget while pushing: repeated short segments pay repeated timing overhead. Explicit stride3 reaches a real push boundary but still leaves insufficient suffix budget or poor grasp tracking. The continuous-clear correction uses a longer push segment with actual lateral feedback and withdraws the left hand to y=-.20m/z=.15m before right grasp. It changes controller geometry and timing, not a learned policy. These coupled fixes are not a causal isolation of arm-arm collision; arm-arm contact was not logged.','', 'Full per-step states, both q/qvel, dictionary commands, TCP/cube transforms, link/cube forces, measured SkillResult and decision observations are stored. Tracking maxima in ANALYSIS include transient commanded-path errors; they are not hardware certification. There is no registered inter-arm/world planning obstacle, so this remains a development geometric controller, not a fully collision-aware strong baseline.','', 'A successful two-skill chain establishes feasibility. It does not establish task-dependent second-action ranking or an independent RL contribution. Same-state parameter interventions are needed before learning.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in rows if 'continuous_p' in r['case'] or 'p020_aligned' in r['case'] or '_full_' in r['case']]
    fig,ax=plt.subplots(figsize=(11,max(5,len(chosen)*.55+1.8)))
    for i,r in enumerate(chosen):
        prefix=r['boundaries'][1] if len(r['boundaries'])>1 else 100
        ax.barh(i,prefix,color='#4d809a',label='Actual push + release prefix' if i==0 else None)
        ax.barh(i,100-prefix,left=prefix,color='#62a88a' if r['success'] else '#cf8062',label='Actual suffix to native100' if i==0 else None)
        ax.text(102,i,f"{'PASS' if r['success'] else 'FAIL'} | {r['final_goal_distance_m']*1000:.1f}mm",va='center',fontsize=9)
    ax.set_yticks(range(len(chosen)),[r['case'].replace('continuous_','') for r in chosen]);ax.invert_yaxis();ax.set_xlim(0,129);ax.set_xlabel('Actual control steps (20Hz; native horizon100)');ax.set_title('Two-robot physical parameter probes: full-task outcome\nPaired native scenes; deterministic controller, no RL gain');fig.text(.5,.012,'Blue: actual push/release prefix | Green/red: remaining suffix, terminal pass/fail',ha='center',fontsize=9);ax.spines[['top','right']].set_visible(False);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(a.out/'TWO_SKILL_BUDGET.png',dpi=180);fig.savefig(a.out/'TWO_SKILL_BUDGET.svg');plt.close(fig)
    lines+=['','Earlier controller-development panel (742000/742001; separate raw archive): the orthogonal aligned-axis pair at742001/endpoint.02 shares the complete physical prefix to step55. Axis_x reaches10.16mm but fails the native static requirement at100; axis_y reaches82.04mm and fails at horizon. These are legitimate distinct second actions, but neither yields full success and no conditional ranking reversal is demonstrated. The next strong baseline should budget complete successor execution plus settling rather than minimize pregrasp motion alone. Native multi-agent execution permits simultaneous commands, but this controller serializes left release then right approach; budget failure can therefore reflect a shared scheduling limitation. Parallel withdrawal/approach requires actual collision and tracking validation before any learning comparison.','', 'The742000 endpoint pair changes complete outcome (20mm target succeeds,80mm fails); current evidence is consistent with a simple smaller-push rule, not yet a task-conditioned RL opportunity. That earlier panel covered only two native seeds; raw failures diagnose controller maturity, not task impossibility. Current table and metrics above identify the present panel separately.']
    predicted=[r for r in rows if r['prediction']['predicted_suffix_steps'] is not None]
    if predicted:
        confusion={'true_positive':0,'false_positive':0,'false_negative':0,'true_negative':0}
        for r in predicted:
            feasible=r['prediction']['predicted_suffix_steps']<=r['prediction']['remaining_budget_at_boundary']
            name=('true_positive' if r['success'] else 'false_positive') if feasible else ('false_negative' if r['success'] else 'true_negative')
            confusion[name]+=1
        (a.out/'MODEL_BUDGET_CONFUSION.json').write_text(json.dumps(dict(scope='descriptive nominal complete-suffix budget feasibility versus native terminal outcome, not learned or calibrated probability',cases=len(predicted),**confusion),indent=2))
        fig,ax=plt.subplots(figsize=(7,5));maximum=max(max(r['prediction']['predicted_suffix_steps'],r['prediction']['remaining_budget_at_boundary']) for r in predicted)+5
        ax.plot([0,maximum],[0,maximum],ls='--',color='#777',label='Predicted duration = remaining native budget')
        for r in predicted:
            d=r['prediction'];ax.scatter(d['remaining_budget_at_boundary'],d['predicted_suffix_steps'],c='#247a62' if r['success'] else '#b85c43',marker='x' if r['case'].endswith('_other') else 'o',s=75)
        minimum=min(min(r['prediction']['predicted_suffix_steps'],r['prediction']['remaining_budget_at_boundary']) for r in predicted)-4
        ax.set_xlim(minimum,maximum);ax.set_ylim(minimum,maximum);ax.set_xlabel('Actual remaining control steps at second boundary');ax.set_ylabel('Predicted complete suffix + settling reserve (steps)');ax.set_title('Nominal budget contract versus native terminal outcome');ax.legend(loc='upper left',fontsize=8);fig.text(.5,.015,'Green: terminal PASS | Red: terminal FAIL | x: matched orthogonal-axis intervention',ha='center',fontsize=9);ax.spines[['top','right']].set_visible(False);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(a.out/'MODEL_BUDGET.png',dpi=180);fig.savefig(a.out/'MODEL_BUDGET.svg');plt.close(fig)
    (a.out/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
