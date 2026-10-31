"""CPU-only provenance-separated EXP-002 evidence audit and figures."""
import argparse, collections, csv, gzip, hashlib, importlib.util, itertools, json
from pathlib import Path
import numpy as np


def read(path, default=None):
    if not path.exists():return default
    return json.loads(path.read_text())

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def scalar(x):return float(np.asarray(x).reshape(-1)[0])

def stop_rule(run, manifest):
    p=run/'program_adaptive_contact.py'
    if p.exists():
        text=p.read_text()
        if 'measured=self.goal.inv()' in text and ('self.done=success' in text or 'self.done=bool(success' in text):return 'OBSERVED_GOAL_STOP'
        if "if bool(u.evaluate()['success'][0]):break" in text:return 'PRIVILEGED_NATIVE_GOAL_STOP'
        return 'UNKNOWN_STOP_RULE'
    if manifest.get('variant','').endswith('_stop'):return 'NATIVE_TRUE_GOAL_STOP'
    return 'UNKNOWN_STOP_RULE'

def failure(row, steps):
    if row.get('truncated_training_tail'):return 'TRAINING_TAIL_NOT_POLICY_OUTCOME'
    if row['success']:return 'SUCCESS'
    err=row.get('error') or ''
    if 'GRASP' in err or 'LIFT' in err:return 'PREFIX_GRASP_OR_LIFT'
    if 'PLAN_' in err:return 'PREFIX_PLAN'
    if 'IK' in err:return 'SUFFIX_IK'
    force=max([float(s.get('tool_cube_force_N',0)) for s in steps]+[0])
    return 'CONTACT_WITHOUT_TERMINAL_GOAL' if force>.01 else 'NO_RECORDED_SOCKET_CONTACT'

def readback(row):
    rb=row.get('physical_readback');config=row.get('physical_config',{})
    if not rb:return dict(status='NOT_RECORDED')
    issues=[];scale=config.get('charger_density_scale',1.);friction=config.get('socket_friction');friction=.3 if friction is None else friction
    for b in rb['charger']:
        if not np.isclose(b['mass_kg'],.0291072*scale,rtol=1e-5):issues.append('mass')
        if not all(x>0 for x in b['inertia_kg_m2']):issues.append('inertia')
        for s in b['collisions']:
            if not np.isclose(s['density_kg_m3'],1000*scale):issues.append('density')
    for b in rb['socket']:
        for s in b['collisions']:
            for key in ['static_friction','dynamic_friction']:
                if not np.isclose(s[key],friction):issues.append(key)
            if not np.isclose(s['restitution'],0.):issues.append('restitution')
    return dict(status='PASS' if not issues else 'FAIL',issues=issues,readback=rb)

def effect_stats(run, transitions, bias_mm=None):
    path=run/'program_contract_effect.py'
    if not path.exists() or not transitions:return dict(status='NOT_APPLICABLE')
    if any(len(t['obs'])!=21 for t in transitions):return dict(status='SKIP_NOT_21_DIMENSIONAL')
    spec=importlib.util.spec_from_file_location('evidence_effect_'+sha(path)[:8],path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    effect=mod.ContractEffect();prior=effect.reset(np.zeros(21));features=[]
    for i,t in enumerate(transitions):
        f=effect.reset(t['obs']) if i==0 else effect.update(t['obs'])
        features.append(np.array(f))
    f=np.asarray(features);used=f[:,-1]>0;nonprior=np.any(np.abs(f-prior)>1e-6,axis=1)
    diagnosis={}
    if bias_mm is not None and np.any(used):
        truth=-np.asarray(bias_mm,dtype=float);pred=f[used,:2]*1.5;spread=f[used,2:4]*1.5
        error=pred-truth;prior_error=np.zeros_like(pred)-truth
        sign_eligible=np.abs(truth)>1e-9
        diagnosis=dict(label_use='OFFLINE_DIAGNOSTIC_ONLY_NEVER_POLICY_INPUT',true_hole_offset_mm=truth.tolist(),
            activated_decisions=int(used.sum()),squared_vector_error_mm2=float(np.sum(error**2)),
            squared_prior_vector_error_mm2=float(np.sum(prior_error**2)),
            mean_estimated_offset_mm=pred.mean(axis=0).tolist(),
            rmse_vector_mm=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            prior_rmse_vector_mm=float(np.sqrt(np.mean(np.sum(prior_error**2,axis=1)))),
            sign_matches=int(np.sum(np.sign(pred[:,sign_eligible])==np.sign(truth[sign_eligible]))),
            sign_eligible_components=int(len(pred)*sign_eligible.sum()),
            heuristic_196std_contains_truth=int(np.sum(np.all(np.abs(error)<=1.96*spread,axis=1))),
            interpretation='Compatibility weights, not calibrated probabilities; heuristic spread coverage only.')
    return dict(status='REPLAYED_EXACT_SNAPSHOT',source_sha256=sha(path),decisions=len(f),
                inference_diagnostic=diagnosis,
                used_nonzero=int(used.sum()),posterior_nonprior=int(nonprior.sum()),
                prior_decisions=int((~nonprior).sum()),max_used_fraction=float(f[:,-1].max()),
                unique_feature_vectors=len(np.unique(np.round(f,6),axis=0)),
                features_constant=bool(np.allclose(f,f[0],atol=1e-6)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    episodes=[];runs=[];qa=[];boundaries={};seen=set();effect_totals=collections.defaultdict(collections.Counter);recoveries=[]
    for run in sorted(a.root.iterdir()):
        if not run.is_dir():continue
        manifest=read(run/'MANIFEST.json') or read(run/'manifest.json') or {}
        rows=read(run/'EPISODES.json');result=read(run/'RESULTS.json')
        if rows is None:rows=result if isinstance(result,list) else []
        accounting=read(run/'CRASH_ACCOUNTING.json')
        for recovered in sorted(run.glob('RECOVERED*.json.gz')):
            recovered_raw=json.load(gzip.open(recovered,'rt'));summary=recovered_raw['summary']
            matching=[r for r in rows if r['seed']==summary['seed'] and r.get('label')==summary.get('label')]
            duplicate=False
            for r in matching:
                original=Path(r.get('path',run/r.get('trajectory','MISSING')))
                try:duplicate=json.load(gzip.open(original,'rt'))['steps']==recovered_raw['steps']
                except (OSError,ValueError):continue
                if duplicate:break
            if not duplicate:
                rows=[r for r in rows if r not in matching]
                rows.append(dict(summary,path=str(recovered),sha256=sha(recovered),completion_reason='recovered_serialization_failure'))
            recoveries.append(dict(run=run.name,accounting=accounting,recovered_path=str(recovered),
                recovered_sha256=sha(recovered),duplicate_of_indexed_episode=duplicate,
                disposition='DO_NOT_DOUBLE_COUNT' if duplicate else 'INCLUDED_RECOVERED_EPISODE',
                valid_indexed_episode_count=len(rows)))
        if not rows:
            if manifest.get('mode')=='train':
                runs.append(dict(run=run.name,path=str(run),manifest=manifest,
                    stop_rule=stop_rule(run,manifest),source=manifest.get('sources',{}).get('program_adaptive_contact.py','UNKNOWN'),
                    result=result if isinstance(result,dict) else None))
            continue
        source=manifest.get('sources',{}).get('program_adaptive_contact.py',manifest.get('source_sha256','UNKNOWN'))
        rule=stop_rule(run,manifest);mode=manifest.get('mode','eval' if run.name.startswith('eval_') else 'diagnostic')
        training_manifest=manifest if mode=='train' else {}
        checkpoint=manifest.get('args',{}).get('model')
        if checkpoint:
            training_manifest=read(Path(checkpoint).parent/'MANIFEST.json',{})
        training_alpha=training_manifest.get('args',{}).get('alpha')
        training_method=training_manifest.get('method')
        runs.append(dict(run=run.name,path=str(run),manifest=manifest,stop_rule=rule,source=source,
                         result=result if isinstance(result,dict) else None))
        for i,row in enumerate(rows):
            path=Path(row.get('path',run/row.get('trajectory','MISSING')))
            if str(path) in seen:continue
            seen.add(str(path));raw=json.load(gzip.open(path,'rt'));steps=raw['steps'];trans=raw.get('transitions',[])
            checks=[]
            if sha(path)!=row['sha256']:checks.append('SHA_MISMATCH')
            if len(steps)!=row['steps'] or len(steps)!=200:checks.append('HORIZON')
            if bool(steps[-1]['native_success'])!=bool(row['success']):checks.append('TERMINAL_SUCCESS_MISMATCH')
            max_clip=0.;max_joint=0.;skill_count=0
            for j,s in enumerate(steps):
                if 'skill_action' not in s:continue
                ac=s['skill_action'];skill_count+=1
                max_clip=max(max_clip,float(np.max(np.abs(ac['clipped']))))
                if not np.allclose(ac['clipped'],np.clip(ac['requested'],-1,1)):checks.append('SKILL_CLIP')
                if abs(ac['advance_m'])>.004001 or ac['advance_m']<-.002001:checks.append('ADVANCE_BOUND')
                if np.max(np.abs(ac['applied_goal_offset_m']))>.0015001:checks.append('LATERAL_BOUND')
                if j:
                    delta=np.max(np.abs(np.array(s['action'])[:7]-np.array(steps[j-1]['q'])[:7]));max_joint=max(max_joint,float(delta))
                    if delta>.050001:checks.append('JOINT_BOUND')
            rb=readback(row);es=effect_stats(run,trans,row.get('bias_mm'))
            if rb['status']=='FAIL':checks.append('PHYSICAL_READBACK')
            policy=manifest.get('method',row.get('label',manifest.get('variant','unknown')))
            tail=bool(row.get('truncated_training_tail',False));label=row.get('label',str(i))
            comparison_id=('controls/'+policy if run.name.startswith('eval_controls_') else run.name)
            first_native=next((j+1 for j,s in enumerate(steps) if s['native_success']),None)
            last_native=next((len(steps)-j for j,s in enumerate(reversed(steps)) if s['native_success']),None)
            first_hold=next((j+1 for j,s in enumerate(steps) if s['phase']=='TERMINAL_HOLD'),None)
            er=dict(run=run.name,policy=policy,comparison_id=comparison_id,mode=mode,stop_rule=rule,source=source,
                    training_alpha=training_alpha,training_method=training_method,checkpoint=checkpoint,
                    seed=row['seed'],label=label,bias_mm=row.get('bias_mm',[0,0]),
                    density=row.get('physical_config',{}).get('charger_density_scale',1),
                    friction=row.get('physical_config',{}).get('socket_friction'),
                    success=bool(row['success']),tail=tail,steps=len(steps),prefix_steps=row.get('prefix_steps',sum(s['phase'] not in ['CONTACT_SERVO','TERMINAL_HOLD'] for s in steps)),
                    failure=failure(row,steps),decisions=len(trans),wall_s=row['wall_s'],
                    first_native_criteria_physics_step=first_native,
                    last_native_criteria_physics_step=last_native,
                    terminal_hold_first_physics_step=first_hold,
                    final_native_position_error_m=steps[-1].get('goal_position_error_m'),
                    final_native_orientation_error_rad=steps[-1].get('goal_angle_error_rad'),
                    final_force_N=float(steps[-1].get('tool_cube_force_N',0)),
                    terminal_native_criteria=bool(steps[-1]['native_success']),
                    native_success_lost_after_first=first_native is not None and not bool(steps[-1]['native_success']),
                    planning_wall_s=row.get('planning_wall_s'),peak_force_N=max(float(s.get('tool_cube_force_N',0)) for s in steps),
                    path=str(path),sha256=row['sha256'],initial_sha256=row['initial_sha256'])
            episodes.append(er);qa.append(dict(run=run.name,label=label,checks=sorted(set(checks)),
                skill_steps=skill_count,max_skill_action=max_clip,max_joint_delta=max_joint,physical_readback=rb,effect=es))
            if es['status']=='REPLAYED_EXACT_SNAPSHOT':
                features_used_by_policy=manifest.get('method') in {'effect_sac','effect_ln_sac'} and not manifest.get('args',{}).get('prior_only',False)
                es['feature_use']='POLICY_INPUT' if features_used_by_policy else 'COUNTERFACTUAL_REPLAY_NOT_POLICY_INPUT'
                effect_totals[(run.name,policy)]['policy_nonprior_decisions']+=es['posterior_nonprior'] if features_used_by_policy else 0
                for key in ['decisions','used_nonzero','posterior_nonprior','prior_decisions']:
                    effect_totals[(run.name,policy)][key]+=es[key]
                effect_totals[(run.name,policy)]['constant_episodes']+=es['features_constant']
                effect_totals[(run.name,policy)]['episodes']+=1
            boundary=raw.get('prefix_boundary')
            if boundary:
                key=(rule,row['seed'],tuple(er['bias_mm']),er['density'],er['friction'])
                boundaries.setdefault(key,[]).append((er,boundary))
    pairs=[]
    fields=['q','qvel','tcp','tool','cube','tcp_T_tool','tool_linear_velocity','tool_angular_velocity','cube_linear_velocity']
    for key,items in boundaries.items():
        for (left,lb),(right,rb) in itertools.combinations(items,2):
            if left['policy']==right['policy'] and left['run']==right['run']:continue
            diffs={f:float(np.max(np.abs(np.asarray(lb[f])-np.asarray(rb[f])))) for f in fields if f in lb and f in rb}
            pairs.append(dict(left=[left['run'],left['label']],right=[right['run'],right['label']],seed=key[1],
                              left_comparison=left['comparison_id'],right_comparison=right['comparison_id'],
                              both_eval=left['mode']=='eval' and right['mode']=='eval',
                              same_initial_hash=left['initial_sha256']==right['initial_sha256'],
                              left_source=left['source'],right_source=right['source'],
                              left_success=left['success'],right_success=right['success'],
                              right_rescue=not left['success'] and right['success'],
                              right_regression=left['success'] and not right['success'],
                              bias_mm=list(key[2]),
                              same_source_and_condition=left['source']==right['source'],prefix_steps_equal=left['prefix_steps']==right['prefix_steps'],
                              recorded_state_max_abs=max(diffs.values(),default=None),per_field_max_abs=diffs,
                              exact_recorded_boundary=all(v==0 for v in diffs.values()) and left['prefix_steps']==right['prefix_steps']))
    groups=[]
    for key in sorted(set((e['run'],e['policy']) for e in episodes)):
        es=[e for e in episodes if (e['run'],e['policy'])==key];valid=[e for e in es if not e['tail']]
        groups.append(dict(run=key[0],policy=key[1],mode=es[0]['mode'],stop_rule=es[0]['stop_rule'],source=es[0]['source'],
                           episodes=len(es),scored_episodes=len(valid),success=sum(e['success'] for e in valid),
                           physical_steps=sum(e['steps'] for e in es),prefix_steps=sum(e['prefix_steps'] for e in es),
                           failures=dict(collections.Counter(e['failure'] for e in es))))
    effect_summary=[dict(run=k[0],policy=k[1],**dict(v),used_fraction=v['used_nonzero']/max(1,v['decisions']),prior_fraction=v['prior_decisions']/max(1,v['decisions'])) for k,v in effect_totals.items()]
    inference=[]
    for name in sorted({q['run'] for q in qa}):
        ds=[q['effect'].get('inference_diagnostic',{}) for q in qa if q['run']==name]
        ds=[d for d in ds if d];n=sum(d['activated_decisions'] for d in ds)
        if not n:continue
        sign_n=sum(d['sign_eligible_components'] for d in ds)
        inference.append(dict(run=name,activated_decisions=n,activated_episodes=len(ds),
            rmse_vector_mm=float(np.sqrt(sum(d['squared_vector_error_mm2'] for d in ds)/n)),
            prior_rmse_vector_mm=float(np.sqrt(sum(d['squared_prior_vector_error_mm2'] for d in ds)/n)),
            sign_matches=sum(d['sign_matches'] for d in ds),sign_eligible_components=sign_n,
            heuristic_196std_coverage=sum(d['heuristic_196std_contains_truth'] for d in ds)/n,
            label_use='OFFLINE_ONLY',interpretation='Decision-weighted compatibility diagnostics; not calibrated probability claims.'))
    eval_scores=[]
    for name in sorted({e['comparison_id'] for e in episodes if e['mode']=='eval'}):
        es=[e for e in episodes if e['mode']=='eval' and e['comparison_id']==name and not e['tail']]
        conditions={(e['seed'],tuple(e['bias_mm']),e['density'],e['friction']) for e in es}
        first=[e['first_native_criteria_physics_step'] for e in es if e['success'] and e['first_native_criteria_physics_step'] is not None]
        eval_scores.append(dict(comparison=name,episodes=len(es),unique_conditions=len(conditions),success=sum(e['success'] for e in es),
            training_alphas=sorted({e['training_alpha'] for e in es if e['training_alpha'] is not None}),
            training_methods=sorted({e['training_method'] for e in es if e['training_method'] is not None}),
            checkpoints=sorted({e['checkpoint'] for e in es if e['checkpoint'] is not None}),
            prefix_failures=sum(e['failure'].startswith('PREFIX_') for e in es),
            first_criteria_mean_step_success_only=float(np.mean(first)) if first else None,
            transient_success_terminal_failure=sum(e['native_success_lost_after_first'] for e in es),
            source_versions=sorted({e['source'] for e in es}),stop_rules=sorted({e['stop_rule'] for e in es}),
            duplicate_condition_warning=len(es)!=len(conditions)))
    eval_pairs=[]
    for names in sorted({(p['left_comparison'],p['right_comparison']) for p in pairs if p['both_eval']}):
        ps=[p for p in pairs if p['both_eval'] and (p['left_comparison'],p['right_comparison'])==names]
        n=len(ps);rescues=sum(p['right_rescue'] for p in ps);regressions=sum(p['right_regression'] for p in ps)
        eval_pairs.append(dict(left=names[0],right=names[1],paired_conditions=n,
            left_success=sum(p['left_success'] for p in ps),right_success=sum(p['right_success'] for p in ps),
            right_rescues=rescues,right_regressions=regressions,right_minus_left_success_fraction=(rescues-regressions)/n,
            exact_recorded_prefix=sum(p['exact_recorded_boundary'] for p in ps),
            same_initial_hash=sum(p['same_initial_hash'] for p in ps),
            differing_source_pairs=sum(not p['same_source_and_condition'] for p in ps),
            max_prefix_numeric_difference=max(p['recorded_state_max_abs'] or 0 for p in ps),
            stop_rule_matching_required=True))
    repeats=[]
    for repeat in [e for e in episodes if e['run'].startswith('repeat_controls_')]:
        originals=[e for e in episodes if e['run'].startswith('eval_controls_') and
                   (e['policy'],e['seed'],e['bias_mm'],e['density'],e['friction'])==
                   (repeat['policy'],repeat['seed'],repeat['bias_mm'],repeat['density'],repeat['friction'])]
        for original in originals:
            left=json.load(gzip.open(original['path'],'rt'))['steps']
            right=json.load(gzip.open(repeat['path'],'rt'))['steps']
            deltas={}
            if len(left)==len(right):
                for field in fields+['action','tool_cube_force_N','native_success']:
                    if all(field in s for s in left+right):
                        deltas[field]=float(np.max(np.abs(np.asarray([s[field] for s in left],dtype=float)-np.asarray([s[field] for s in right],dtype=float))))
            repeats.append(dict(original=original['path'],repeat=repeat['path'],policy=repeat['policy'],
                seed=repeat['seed'],bias_mm=repeat['bias_mm'],same_success=original['success']==repeat['success'],
                same_initial_hash=original['initial_sha256']==repeat['initial_sha256'],
                same_stop_rule=original['stop_rule']==repeat['stop_rule'],
                original_success=original['success'],repeat_success=repeat['success'],
                step_count_equal=len(left)==len(right),trajectory_field_max_abs=deltas,
                exact_recorded_trajectory=len(left)==len(right) and bool(deltas) and all(v==0 for v in deltas.values())))
    unfinished=[r['run'] for r in runs if r['manifest'].get('mode')=='train' and r['result'] is None]
    finalized_updates=sum((r['result'] or {}).get('optimizer_updates',0) for r in runs)
    budget=dict(saved_complete_episodes=len(episodes),saved_physics_steps=sum(e['steps'] for e in episodes),
                saved_prefix_steps=sum(e['prefix_steps'] for e in episodes),saved_skill_decisions=sum(e['decisions'] for e in episodes),
                optimizer_updates=None if unfinished else finalized_updates,
                finalized_run_optimizer_updates=finalized_updates,unfinalized_training_runs=unfinished,
                excludes_live_unfinished_episodes=True,training_tail_episodes=sum(e['tail'] for e in episodes))
    payload=dict(budget=budget,groups=groups,effect=effect_summary,qa_failures=sum(bool(q['checks']) for q in qa),
                 paired_boundaries=len(pairs),exact_paired_boundaries=sum(x['exact_recorded_boundary'] for x in pairs),
                 interpretation='Scores grouped by exact run/source/stop rule. Diagnostics and training are not held-out evaluation.')
    for name,obj in [('SUMMARY',payload),('EPISODES',episodes),('RUN_INDEX',runs),('QA',qa),('PREFIX_PAIRS',pairs),('RECOVERY_ACCOUNTING',recoveries),('EVAL_COMPARISON',dict(scores=eval_scores,pairs=eval_pairs)),('REPEATABILITY',repeats),('EFFECT_INFERENCE',inference)]:
        (a.out/f'{name}.json').write_text(json.dumps(obj,indent=2))
    if episodes:
        with (a.out/'EPISODES.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=episodes[0]);w.writeheader();w.writerows(episodes)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    if eval_scores:
        fig,ax=plt.subplots(figsize=(9,max(3,len(eval_scores)*.5)))
        y=np.arange(len(eval_scores));values=[s['success']/max(1,s['episodes']) for s in eval_scores]
        colors=['#c28138' if 'zero_z' in s['comparison'] else '#397d76' for s in eval_scores]
        labels={'controls/geometry':'Geometry feedback','controls/rls':'Online RLS','controls/search':'Progress search','eval_sac':'Residual SAC (alpha .02)','eval_effect_sac':'SAC + effect features','eval_effect_prior_only':'Effect SAC: prior only','eval_ln_sac':'LN-SAC (alpha .10)','eval_ln_sac_alpha002':'LN-SAC (alpha .02)','eval_effect_ln_sac':'LN-SAC + effect features','eval_effect_ln_prior_only':'LN effect: prior only','eval_sac_zero_z':'SAC, Z=0 (post-hoc)'}
        ax.barh(y,values,color=colors);ax.set_yticks(y,[labels.get(s['comparison'],s['comparison']) for s in eval_scores]);ax.invert_yaxis();ax.set_xlim(0,1.15)
        for j,s in enumerate(eval_scores):ax.text(values[j]+.015,j,f"{s['success']}/{s['episodes']}",va='center')
        ax.set_title('Development: 4 scene seeds × 3 biases | Full-task terminal outcome')
        ax.set_xlabel('Success fraction; single-seed bounded training pilot, no novelty claim')
        fig.tight_layout();fig.savefig(a.out/'EVAL_MAIN.png',dpi=200);fig.savefig(a.out/'EVAL_MAIN.svg');plt.close(fig)
    for mode,name in [('diagnostic','DIAGNOSTIC_OUTCOMES'),('eval','EVALUATION_OUTCOMES')]:
        gs=[g for g in groups if g['mode']==mode]
        if not gs:continue
        fig,ax=plt.subplots(figsize=(11,max(3,len(gs)*.36)))
        y=np.arange(len(gs));colors=['#3b7a57' if 'OBSERVED' in g['stop_rule'] else '#999999' for g in gs]
        ax.barh(y,[g['success']/max(1,g['scored_episodes']) for g in gs],color=colors)
        ax.set_yticks(y,[f"{g['run']} / {g['policy']} / {g['source'][:6]}" for g in gs]);ax.invert_yaxis();ax.set_xlim(0,1.15)
        for j,g in enumerate(gs):ax.text(g['success']/max(1,g['scored_episodes'])+.015,j,f"{g['success']}/{g['scored_episodes']}",va='center')
        ax.set_xlabel('Native terminal success fraction (each row separate; no pooled score)')
        ax.set_title('Observed-stop: green; true-goal/oracle-stop: grey | simulator physics only')
        fig.tight_layout();fig.savefig(a.out/f'{name}.png',dpi=200);fig.savefig(a.out/f'{name}.svg');plt.close(fig)
    if effect_summary:
        fig,ax=plt.subplots(figsize=(9,max(3,len(effect_summary)*.5)));y=np.arange(len(effect_summary))
        ax.barh(y,[e['used_fraction'] for e in effect_summary]);ax.set_yticks(y,[e['run'] for e in effect_summary]);ax.set_xlim(0,1)
        ax.set_xlabel('Decision fraction with at least one eligible effect observation');ax.set_title('Exact snapshot feature replay; activation is not evidence of benefit')
        fig.tight_layout();fig.savefig(a.out/'EFFECT_ACTIVATION.png',dpi=200);fig.savefig(a.out/'EFFECT_ACTIVATION.svg');plt.close(fig)
    text=['# EXP-002 automated evidence audit','',json.dumps(budget,indent=2),'',
          f"QA issues: {payload['qa_failures']}; exact recorded prefix pairs {payload['exact_paired_boundaries']}/{len(pairs)}.",
          'Per-run scores retain snapshot and stop-rule identity. Prefix comparisons may cross source hashes, but require the same seed, bias, physics and stop rule; source changes are explicitly recorded. A matching initial state hash does not imply matching grasp/contact boundary.',
          'Source RGB/video replay adds no physics. Physical readback verifies constructed component values, not system-identification estimates.',
          'Training tails are counted in cost but excluded from policy terminal success denominators. Live unfinished runs are not included until saved.',
          'Effect activation/constant-feature diagnostics use saved source snapshots and only 21-dimensional observations. Older 19-dimensional records are skipped.',
          '', '## Per-run scores','']
    text += [f"- {g['run']} / {g['policy']}: {g['success']}/{g['scored_episodes']}; {g['stop_rule']}; source {g['source'][:12]}; {g['mode']}." for g in groups]
    (a.out/'README.md').write_text('\n'.join(text)+'\n');print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
