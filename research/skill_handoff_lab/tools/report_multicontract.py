"""Plots only measured episodes; pending methods are absent, never imputed."""
import argparse,json,hashlib,gzip
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--causal',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(path.read_text())
causal=[]
for directory in sorted(a.causal.glob('causal_*')):
    file=directory/'frozen_evaluation.json'
    if not file.exists():continue
    rows=read(file)['episodes'];audit=read(directory/'action_audit.json') if (directory/'action_audit.json').exists() else []
    effect=[];requested=[]
    for step in audit:
        nominal=np.asarray(step['nominal_delta_m']);actual=np.asarray(step['actual_command_delta_m'])
        effect.extend((actual-nominal).tolist());requested.extend(step['residual_before_clip_m'])
    causal.append({'experiment':directory.name,'successes':sum(x['success'] for x in rows),'n':len(rows),
        'mean_episode_s':np.mean([x['episode_control_steps']/60 for x in rows]),
        'mean_abs_effect_mm':(np.abs(effect).mean(0)*1000).tolist() if effect else None,
        'mean_abs_requested_residual_mm':(np.abs(requested).mean(0)*1000).tolist() if requested else None})
(a.output/'causal_summary.json').write_text(json.dumps(causal,indent=2))
def case_traces(directory):
    traces={}
    for step in read(directory/'action_audit.json'):
        if step['phase']!='evaluation':continue
        for case,q in zip(step['case_ids'],step['q_before']):
            if case>=0:traces.setdefault(case,[]).append(q)
    return traces
paired=[]
for family,version in [('residual','v10'),('contract','v17')]:
    zero=case_traces(a.causal/f'causal_{family}_zero_{version}')
    for mode in ['learned','random']:
        candidate=case_traces(a.causal/f'causal_{family}_{mode}_{version}');differences=[]
        for case in sorted(zero.keys()&candidate.keys()):
            count=min(len(zero[case]),len(candidate[case]));differences.extend(np.array(candidate[case][:count])-np.array(zero[case][:count]))
        diff=np.array(differences)
        paired.append({'family':family,'mode':mode,'comparison':'same case and within-episode control-step occurrence vs zero residual',
            'paired_physical_states':len(diff),'mean_joint_vector_deviation_rad':float(np.linalg.norm(diff,axis=1).mean()),
            'max_joint_vector_deviation_rad':float(np.linalg.norm(diff,axis=1).max()),
            'per_joint_rms_deviation_rad':np.sqrt((diff**2).mean(0)).tolist()})
(a.output/'causal_physical_state_comparison.json').write_text(json.dumps(paired,indent=2))
fig,axes=plt.subplots(1,2,figsize=(13,5));labels=[r['experiment'].replace('causal_','').replace('_v10','').replace('_v17','') for r in causal]
axes[0].bar(labels,[r['successes']/r['n'] for r in causal]);axes[0].set_ylim(0,1.1);axes[0].set_ylabel('Measured success fraction (same 64 cases)')
axes[1].bar(labels,[r['mean_episode_s'] for r in causal]);axes[1].set_ylabel('Measured mean episode duration (s)')
for ax in axes:ax.tick_params(axis='x',rotation=65)
fig.suptitle('Legacy wide-cabinet causal audit | REAL PHYSX | no performance gain inferred');fig.tight_layout();fig.savefig(a.output/'fig_causal_ablation.png',dpi=150);plt.close(fig)
grid=[]
for directory in sorted(a.grid.glob('multicontract_v*')):
    if not directory.is_dir() or not directory.name.endswith(('_val','_test','_validation')):continue
    file=directory/'frozen_evaluation.json'
    if not file.exists():continue
    meta=read(directory/'manifest.json');rows=read(file)['episodes']
    groups={g: [x for x in rows if x['task_group']==g] for g in sorted({x['task_group'] for x in rows})}
    grid.append({'experiment':directory.name,'split':meta['task_split'],'method':meta['controller'],'seed':meta['seed'] if 'seed' in meta else meta['industrial_geometry']['training_seed'],
        'successes':sum(x['success'] for x in rows),'n':len(rows),'success_fraction':np.mean([x['success'] for x in rows]),
        'mean_episode_s':np.mean([x['episode_control_steps']/60 for x in rows]),
        'mean_success_s':np.mean([x['episode_control_steps']/60 for x in rows if x['success']]) if any(x['success'] for x in rows) else None,
        'timeouts':sum(x['timeout'] for x in rows),'failed':sum(x['failed'] for x in rows),
        'max_peak_net_force_N':max(x['peak_net_contact_force_N'] for x in rows),
        'groups':{g:{'n':len(xs),'successes':sum(x['success'] for x in xs),'mean_episode_s':np.mean([x['episode_control_steps']/60 for x in xs])} for g,xs in groups.items()},
        'source_sha256':meta['source_sha256'],'suite_sha256':meta['task_suite_sha256']})
(a.output/'validation_summary.json').write_text(json.dumps([r for r in grid if r['split']=='validation'],indent=2))
(a.output/'test_summary.json').write_text(json.dumps([r for r in grid if r['split']=='test'],indent=2))
if grid:
    plot_rows=[r for r in grid if r['split']=='test'] or grid
    methods=sorted({r['method'] for r in plot_rows});fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,ylabel in zip(axes,['success_fraction','mean_episode_s'],['Success fraction','Episode duration (s)']):
        values=[[r[key] for r in plot_rows if r['method']==m] for m in methods]
        ax.bar(methods,[np.mean(v) for v in values],yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in values],capsize=4)
        for i,v in enumerate(values):ax.scatter([i]*len(v),v,c='black',s=15)
        ax.set_ylabel(ylabel);ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Frozen geometry-group '+('TEST' if any(r['split']=='test' for r in grid) else 'validation')+' | REAL PHYSX | seeds, mean +/- SD')
    fig.tight_layout();fig.savefig(a.output/'fig_multicontract_validation.png',dpi=150);plt.close(fig)
table=['# Frozen measured results','', '|Split|Method|Seed outcomes (success/64)|Mean success fraction ± seed SD|Mean episode seconds ± seed SD|','|---|---|---|---|---|']
for split in ['validation','test']:
    for method in sorted({r['method'] for r in grid if r['split']==split}):
        rs=[r for r in grid if r['split']==split and r['method']==method]
        success=[r['success_fraction'] for r in rs];duration=[r['mean_episode_s'] for r in rs]
        fmt=lambda v:f'{np.mean(v):.6f} ± {np.std(v,ddof=1):.6f}' if len(v)>1 else f'{v[0]:.6f} (one deterministic baseline)'
        outcomes=', '.join(f"{r['seed']}: {r['successes']}/{r['n']}" for r in rs)
        table.append(f'|{split}|{method}|{outcomes}|{fmt(success)}|{fmt(duration)}|')
table+=['','Each training seed reuses the same 64 scene cases; do not treat three seeds as 192 independent scenes. A 64/64 Wilson 95% interval is approximately [94.3%, 100%], not proof of population success=100%.','',
    'V19 is a pilot BEFORE measured-workspace termination was added in v21: command projection did not prevent 15–16mm workspace excursions in some FAILED PPO trajectories. Successful cases independently pass. Keep original failures/time; do not present this as the corrected formal-training comparison.','',
    'All methods use the predeclared 16,384 interaction checkpoint; no validation checkpoint selection. Standard PPO at this small pilot budget is not a converged-algorithm verdict. Episode time includes failures; success-only time is stored separately in JSON.','',
    'Raw endpoint success is cross-checked by independent per-process convex geometry, nonadjacent right-arm self collision, soft limits, measured pose and workspace audit; inspect those reports before calling a trajectory qualified. No physical contact grasp or full robot safety claim.']
(a.output/'FROZEN_RESULTS.md').write_text('\n'.join(table)+'\n')
print(json.dumps({'causal_experiments':len(causal),'frozen_evaluations':len(grid)}))
