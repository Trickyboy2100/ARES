"""Reconcile native outcomes, paired risk, real actor updates and interaction costs."""
import argparse,csv,hashlib,json,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def csvout(p,rows):
    with p.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
def group_ci(delta):
    # Preserve shared 16-env reset groups; never claim seeds x cases independent.
    groups=np.array(delta,dtype=float).reshape(-1,16).mean(1);rng=np.random.default_rng(20261028);boot=groups[rng.integers(0,len(groups),(5000,len(groups)))].mean(1)
    return np.quantile(boot,[.025,.975]).tolist()

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=True,parents=True);base=a.runs
    raw=json.loads((base/'validation64_r1/RESULTS.json').read_text());cem=json.loads((base/'validation64_cem/RESULTS.json').read_text());rows=raw['rows']+cem['rows'];by=defaultdict(dict);hashes={}
    for r in rows:
        group=r['seeds'][0]
        if group not in hashes:hashes[group]=r['initial_sha256']
        assert hashes[group]==r['initial_sha256']
        for case,y in zip(r['seeds'],r['success']):by[r['policy']][case]=bool(y)
    cases=sorted(by['nominal']);nominal=np.array([by['nominal'][c] for c in cases],int);tables=[];paired=[]
    for policy in by:
        ys=np.array([by[policy][c] for c in cases],int);delta=ys-nominal;lo,hi=group_ci(delta)
        tables.append(dict(task='StackCube-v1',split='DEVELOPMENT_NOT_SEALED_TEST',policy=policy,model_seed=policy.rsplit('_',1)[-1] if '_ppo_' in policy else '',cases=len(cases),successes=int(ys.sum()),success_rate=float(ys.mean()),rescues=int((delta==1).sum()),regressions=int((delta==-1).sum()),gain_pp=float(delta.mean()*100),gain_group_ci95_low_pp=lo*100,gain_group_ci95_high_pp=hi*100,native_horizon=50))
        for case,y,n in zip(cases,ys,nominal):paired.append(dict(task='StackCube-v1',case=case,reset_group=(case-cases[0])//16,policy=policy,nominal_success=int(n),policy_success=int(y),class_pair=('RESCUE' if y>n else 'REGRESSION' if y<n else 'BOTH_SUCCESS' if y else 'BOTH_FAIL'),sealed_test=False))
    csvout(out/'EXPERIMENT_RESULTS.csv',tables);csvout(out/'CASE_PAIRED_CONFUSION.csv',paired)
    curves=[];models=[];training_eval_cost=0
    for job in json.loads((base/'training/BATCH.json').read_text()):
        assert job['complete'] and job['returncode']==0
        folder=base/'training/runs'/Path(job['checkpoint']).parent.name;first=folder/'ckpt_1.pt';last=folder/'final_ckpt.pt';initial=torch.load(first,map_location='cpu',weights_only=True);final=torch.load(last,map_location='cpu',weights_only=True)
        actor_l2=float(torch.sqrt(sum(((final[k]-initial[k])**2).sum() for k in final if k.startswith('actor_mean.'))));assert actor_l2>0
        event=EventAccumulator(str(next(folder.glob('events.*'))),size_guidance={'scalars':0});event.Reload();tags=event.Tags()['scalars']
        for tag in ['train/success_at_end','eval/success_at_end','losses/policy_loss','losses/value_loss']:
            curves.extend(dict(policy=job['mode'],seed=job['seed'],tag=tag,step=x.step,value=x.value) for x in event.Scalars(tag))
        evaluation_count=len(event.Scalars('eval/success_at_end'));training_eval_cost+=evaluation_count*16*50
        models.append(dict(relative_path=str(last.relative_to(base)),mode=job['mode'],seed=job['seed'],path=str(last),sha256=sha(last),bytes=last.stat().st_size,initial_checkpoint_sha256=sha(first),actor_delta_l2=actor_l2,online_interactions=102400,evaluation_interactions=evaluation_count*16*50,load_check='PASS: torch weights_only plus actually executed policy in paired native evaluation'))
    save(out/'LEARNING_CURVES.json',curves);save(out/'MODEL_INDEX.json',dict(models=models,selector_models=[dict(path=str(p),relative_path=str(p.relative_to(base)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((base/'selectors_r1').glob('*.pt')) if p.name!='paired_dataset.pt'],source_freeze={p.name:sha(p) for p in (base/'training/source').glob('*.py')},raw_models_public_git=False,loader='method_outcome_selector.py eval --models [training/runs] --selector [selectors_r1] --upstream [pinned PPO]'))
    save(out/'ACTOR_UPDATE_AUDIT.json',models)
    # Preserve exact frozen controller code used to generate the model checkpoint hashes.
    for pth in (base/'training/source').glob('*.py'):
        target=out/'repro/frozen_r1'/pth.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(pth.read_bytes())
    cost=[];sources={}
    folders=['stack_authority_pilot','stack_paired_train512','peg_authority_pilot','peg_tracking_r2','peg_feedback_r3','cem_height_equal_budget','validation64_r1','validation64_cem','pyramid_authority_r1','pyramid_native_planner','pyramid_paired_parameters','stack_video_batch0','stack_video_batch1','peg_video_round1','peg_video_round2','pyramid_video_rescue','pyramid_video_regression','pyramid_release_control']
    physical={}
    for name in folders:
        folder=base/name;result=next((folder/n for n in ['RESULTS.json','results.json'] if (folder/n).exists()),None)
        if result:
            value=json.loads(result.read_text());interactions=value.get('physical_interactions',0);cost.append(dict(run=name,control_interactions=interactions,kind='paired_probe_or_evaluation',source_sha256=sha(result)));sources[name]=dict(path=str(result),relative_path=str(result.relative_to(base)),sha256=sha(result),bytes=result.stat().st_size);physical[name]=value
    cost.extend([dict(run='six_PPO_training',control_interactions=614400,kind='online_training',source_sha256=sha(base/'training/BATCH.json')),dict(run='upstream_training_monitor_eval',control_interactions=training_eval_cost,kind='monitor_evaluation',source_sha256=sha(out/'LEARNING_CURVES.json'))]);csvout(out/'ACTUAL_INTERACTION_COST.csv',cost)
    save(out/'RAW_SOURCE_INDEX.json',sources)
    # Design probes: actual terminal change, not request magnitude alone.
    design=physical['stack_authority_pilot'];design_by={(seed,r['variant']):bool(y) for r in design['rows'] for seed,y in zip(r['seeds'],r['success'])};authority=[]
    for policy in ['x_minus','x_plus','y_minus','y_plus','height_minus','height_plus']:
        ids=sorted(s for s,v in design_by if v=='nominal');d=np.array([int(design_by[s,policy])-int(design_by[s,'nominal']) for s in ids]);axis='height' if policy.startswith('height') else policy[0]
        authority.append(dict(task='StackCube-v1',parameter=policy,unit='m',bound=.012 if axis=='height' else .015,active_phases='5/6 for height; 4+ for XY',cases=len(ids),rescues=int((d==1).sum()),regressions=int((d==-1).sum()),native_budget=50,source='stack_authority_pilot; same initial hash for all seven branches',qualification='Geometry/state baseline; not an RL gain'))
    for name in ['peg_authority_pilot','peg_tracking_r2','peg_feedback_r3','pyramid_authority_r1','pyramid_native_planner','pyramid_paired_parameters','pyramid_release_control']:
        if name not in physical:continue
        value=physical[name];by_policy=defaultdict(list)
        for r in value['rows']:by_policy[r.get('policy',r.get('variant'))].extend(r['success'] if isinstance(r['success'],list) else [r['success']])
        for label,ys in by_policy.items():authority.append(dict(task=value['task'],parameter=label,unit='native controller',bound='',active_phases='see raw source/controller contract',cases=len(ys),rescues='',regressions='',native_budget=value['native_horizon'],source=name,qualification=f'Native completion {sum(ys)}/{len(ys)}; no learner increment claim'))
    pyr=physical['pyramid_paired_parameters'];pby=defaultdict(dict)
    for r in pyr['rows']:pby[r['policy']][r['seed']]=r
    pyr_pairs=[]
    for policy,rr in pby.items():
        for seed,r in rr.items():
            n=pby['tracked_nominal'][seed];assert n['initial_sha256']==r['initial_sha256'];d=int(r['success'])-int(n['success'])
            pyr_pairs.append(dict(task='StackPyramid-v1',seed=seed,policy=policy,nominal_success=int(n['success']),policy_success=int(r['success']),class_pair='RESCUE' if d==1 else 'REGRESSION' if d==-1 else 'BOTH_SUCCESS' if r['success'] else 'BOTH_FAIL',initial_sha256=r['initial_sha256'],request_m=r['first_parameter_requested_m'],applied_m=r['first_parameter_applied_m'],steps=r['steps']))
    csvout(out/'PYRAMID_PAIRED_CASES.csv',pyr_pairs)
    # Measured handoff drift and a same-case strong-control explanation.
    drift=[];release_pairs=[]
    for folder in ['pyramid_paired_parameters','pyramid_release_control']:
        if folder not in physical:continue
        for i,r in enumerate(physical[folder]['rows']):
            ordinal=(r['seed']-122000);trace_file=base/folder/f'{ordinal:03d}_{r["policy"]}.json';trace=json.loads(trace_file.read_text());states=trace['transitions']
            opening=next((j for j,x in enumerate(states) if x['action'][-1]>0),None)
            distances=np.array([np.linalg.norm(np.array(x['state']['actors']['cubeA'][0][:2])-np.array(x['state']['actors']['cubeB'][0][:2])) for x in states])*1000
            actors=states[-1]['state']['actors'];ca=np.array(actors['cubeC'][0][:3])-np.array(actors['cubeA'][0][:3]);cb=np.array(actors['cubeC'][0][:3])-np.array(actors['cubeB'][0][:3])
            drift.append(dict(folder=folder,seed=r['seed'],policy=r['policy'],success=r['success'],first_open_step=opening+1 if opening is not None else '',AB_min_before_open_mm=float(distances[:opening+1].min()) if opening is not None else '',AB_at_open_mm=float(distances[opening]) if opening is not None else '',AB_terminal_mm=float(distances[-1]),C_minus_A_height_mm=float(ca[2]*1000),C_minus_B_height_mm=float(cb[2]*1000),native_success_with_C_above_both=bool(r['success'] and ca[2]>.02 and cb[2]>.02),trajectory_sha256=sha(trace_file)))
            if folder=='pyramid_release_control':
                n=pby['tracked_nominal'][r['seed']];assert n['initial_sha256']==r['initial_sha256'];d=int(r['success'])-int(n['success'])
                release_pairs.append(dict(seed=r['seed'],policy=r['policy'],nominal_success=n['success'],policy_success=r['success'],class_pair='RESCUE' if d>0 else 'REGRESSION' if d<0 else 'BOTH_SUCCESS' if r['success'] else 'BOTH_FAIL',steps=r['steps'],initial_sha256=r['initial_sha256']))
    csvout(out/'PYRAMID_HANDOFF_DRIFT.csv',drift)
    if release_pairs:csvout(out/'PYRAMID_RELEASE_CASES.csv',release_pairs)
    failure=[]
    for row in raw['rows']:
        ordinal=(row['seeds'][0]-200000)//16;trace=json.loads((base/'validation64_r1'/f'{row["policy"]}_{ordinal:03d}.json').read_text());last,prev=trace['transitions'][-2:]
        for j,seed in enumerate(row['seeds']):
            if row['success'][j]:continue
            state=prev;before=last;phase=state['authority']['phase_before'][j];delta=np.array(state['cubeA'][j][:3])-np.array(state['cubeB'][j][:3]);velocity=(np.array(state['cubeA'][j][:3])-np.array(before['cubeA'][j][:3]))/.05
            failure.append(dict(task='StackCube-v1',seed=seed,policy=row['policy'],phase_before_final_step=phase,diagnostic='PRE_PLACE_NOT_REACHED' if phase<4 else 'LOWERING_OR_PREPLACE_INCOMPLETE' if phase<6 else 'RELEASE_DWELL_INCOMPLETE' if phase==6 else 'NATIVE_FAILURE_AFTER_RELEASE',AB_xy_mm=float(np.linalg.norm(delta[:2])*1000),AB_height_mm=float(delta[2]*1000),finger_opening_mm=float(sum(state['q'][j][-2:])*1000),last_step_velocity_proxy_mm_s=float(np.linalg.norm(velocity)*1000),scope='Observed phase/state diagnostic, not a reconstructed native contact predicate'))
    csvout(out/'PHYSICAL_FAILURE_DIAGNOSTICS.csv',failure)
    csvout(out/'CONTROL_AUTHORITY_MAP.csv',authority)
    method_differences=np.array([[by[f'parameter_ppo_{s}'][c]-by[f'residual_ppo_{s}'][c] for c in cases] for s in [31,37,43]],float)
    rng=np.random.default_rng(20261028);boot=[]
    for _ in range(5000):
        selected_seeds=rng.integers(0,3,3);selected_groups=rng.integers(0,4,4);selected_cases=np.concatenate([np.arange(g*16,(g+1)*16) for g in selected_groups]);boot.append(method_differences[selected_seeds][:,selected_cases].mean())
    summary=dict(source_kind='NEW_NATIVE_PHYSICS_AND_TRAINING',task_scores=tables,training_control_interactions=614400,total_control_interactions=sum(r['control_interactions'] for r in cost),trained_PPO_models=6,ppo_actor_updates_verified=True,sealed_test_used=False,independent_method_increment='NOT_ESTABLISHED',parameter_minus_residual_mean_pp=float(method_differences.mean()*100),paired_hierarchical_ci95_pp=(np.quantile(boot,[.025,.975])*100).tolist(),statistical_unit='Shared case groups of16 + training seeds; not3x64 independent scenes',physical_folders={k:dict(task=v.get('task'),rows=len(v['rows']),interactions=v.get('physical_interactions')) for k,v in physical.items()},interpretation='Parameter PPO does not consistently beat ordinary Residual PPO; paired abstention does not improve nominal on current development data; strong fixed/CEM controls matter.')
    save(out/'RESULTS.json',summary)
    with (out/'RESULTS_TABLE.tex').open('w') as f:
        f.write('\\begin{tabular}{lrrr}\\toprule\nPolicy & Success & Rescue & Regress \\\\\n\\midrule\n')
        for r in tables:f.write(r['policy'].replace('_','\\_')+f" & {r['successes']}/{r['cases']} & {r['rescues']} & {r['regressions']} \\\\\n")
        f.write('\\bottomrule\\end{tabular}\n')
    print(json.dumps({k:summary[k] for k in ['total_control_interactions','training_control_interactions','parameter_minus_residual_mean_pp','paired_hierarchical_ci95_pp','independent_method_increment']}))

if __name__=='__main__':main()
