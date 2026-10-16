"""Observed M0 / paired-mechanism / online RL results. Unknown != failure."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def load(p):return json.loads(p.read_text())
def yaw_error(post,spec):
    q=post['object_quaternion_wxyz'];yaw=Rotation.from_quat([*q[1:],q[0]]).as_euler('xyz')[2]-spec.get('direction',0.)
    return abs(np.arctan2(np.sin(4*yaw),np.cos(4*yaw))/4)

def main(root):
    rows=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/'frozen_evaluation.json').exists():continue
        m=load(directory/'manifest.json');cases=load(directory/'frozen_evaluation.json');train=load(directory/'training_episodes.json')
        audit=load(directory/'INDEPENDENT_TASK_AND_CONTACT.json') if (directory/'INDEPENDENT_TASK_AND_CONTACT.json').exists() else None
        row={'run':directory.name,'seed':m['seed'],'role':m.get('evaluation_role'),'task':m.get('task_version'),
            'training_interactions':m['online_training_interactions'],'training_episodes':len(train),'training_successes':sum(x['success'] for x in train),
            'n':len(cases),'raw_successes':sum(x['success'] for x in cases),'independent_successes':audit['independent_successes'] if audit else None,
            'families':{f:{'n':sum(x['family']==f for x in cases),'success':sum(x['success'] and x['family']==f for x in cases)} for f in ['push','slot']},
            'handoffs':sum(x.get('handoff_state') is not None for x in cases),'constraint_failures':sum(x['failed'] for x in cases),
            'mean_episode_time_s':float(np.mean([x['steps']/60 for x in cases])),
            'success_time_s':float(np.mean([x['steps']/60 for x in cases if x['success']])) if any(x['success'] for x in cases) else None,
            'failures_reached_position':sum(not x['success'] and x.get('minimum_final_distance_m',1)<.008 for x in cases),
            'mean_peak_contact_N':float(np.mean([x['peak_force_N'] for x in cases])),
            'mean_terminal_speed_m_s':float(np.mean([np.linalg.norm(x['post_step_state']['object_linear_velocity']) for x in cases])),
            'source_sha256':m['task_source_sha256'],'method_source_sha256':m.get('credit_source_sha256')}
        if (directory/'action_log.json').exists():
            actions=load(directory/'action_log.json');ev=[x for x in actions if x['mode']=='evaluation']
            raw=np.array([x['raw'] for x in ev]);applied=np.array([x['applied_xy'] for x in ev])
            row['evaluation_action_raw_xy_rms']=float(np.sqrt(np.mean(raw[:,:,:2]**2)))
            row['evaluation_raw_xy_clip_fraction']=float(np.mean(abs(raw[:,:,:2])>1))
            row['applied_xy_rms_m']=float(np.sqrt(np.mean(applied**2)))
            if ev and 'phase_before' in ev[0]:
                phase=np.array([x['phase_before'] for x in ev]);mask=phase==1
                row['phase1_negative_switch_fraction']=float(np.mean(raw[:,:,2][mask]<0)) if mask.any() else None
        credit=load(directory/'credit_audits.json')
        row['credit_updates']=len(credit)
        if credit:
            row['credit_coverage']=sum(x['samples'] for x in credit)/sum(x['total_samples'] for x in credit)
            row['credit_actor_gradient_max']=max(x['last_minibatch_actor_gradient_l2'] for x in credit)
        rows.append(row)
    mechanism={}
    ds=[root/f'v6b_paired_dwell{n}' for n in [0,12,30]]
    if all((d/'frozen_evaluation.json').exists() for d in ds):
        records=[{x['case_id']:x for x in load(d/'frozen_evaluation.json')} for d in ds]
        common=set.intersection(*[{i for i,x in d.items() if x['handoff_state'] is not None} for d in records])
        mechanism={'paired_cases':len(common),'all_cases_each':64,'common_case_ids':sorted(common),
            'full_successes':[sum(x['success'] for x in d.values()) for d in records],
            'common_suffix_successes':[sum(d[i]['success'] for i in common) for d in records],
            'mean_handoff_linear_speed':[float(np.mean([np.linalg.norm(d[i]['handoff_state']['object_linear_velocity']) for i in common])) for d in records],
            'delayed_rescued_cases':[sorted(i for i in common if not records[0][i]['success'] and d[i]['success']) for d in records[1:]],
            'scope':'Fresh-process full physical replay, same case reset and suffix controller; no partial state teleport or solver-cache restoration claim',
            'interpretation':'Slower handoff alone does not improve this suffix. Multiple state dimensions change together; not isolated velocity causation.'}
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for k,(d,label) in enumerate(zip(records,['0 wait','12 controls','30 controls'])):
            speed=[1000*np.linalg.norm(d[i]['handoff_state']['object_linear_velocity']) for i in sorted(common)]
            axes[0].scatter(np.full(len(speed),k),speed,c=[d[i]['success'] for i in sorted(common)],cmap='RdYlGn',s=12,alpha=.8,vmin=0,vmax=1)
        axes[0].set(xticks=range(3),xticklabels=['0','12','30'],xlabel='Pre-handoff wait (control steps)',ylabel='Observed handoff speed (mm/s)',title='Same 46 valid-handoff cases; green=actual suffix success')
        axes[1].bar(range(3),mechanism['common_suffix_successes']);axes[1].set(xticks=range(3),xticklabels=['0','12','30'],ylim=(0,len(common)),ylabel='Same downstream controller success / common cases',xlabel='Wait controls')
        fig.savefig(root/'fig_handoff_mechanism.png',dpi=160);plt.close(fig)
    (root/'RESULTS_V6.json').write_text(json.dumps({'rows':rows,'mechanism':mechanism},indent=2))
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    for label,color in [('ppo','#777777'),('residual','#377eb8'),('recover','#e68613'),('bootstrap_all','#54a24b')]:
        for seed in [31,41,51]:
            path=root/f'v6b_{label}_seed{seed}'/'training_episodes.json'
            if not path.exists():continue
            ep=load(path);xs=sorted(set(e['env_interactions'] for e in ep))
            yy=[100*np.mean([e['success'] for e in ep if x-16384<e['env_interactions']<=x]) for x in xs]
            ax.plot(xs,yy,label=f'{label} seed{seed}',color=color,alpha=.7)
    ax.set(title='Actual online physics; completed episodes, trailing 16384 interactions',xlabel='Real online interactions',ylabel='Training success (%)',ylim=(-2,102));ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.2)
    fig.savefig(root/'fig_online_learning_v6.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for x,label in enumerate(['ppo','residual','recover']):
        vals=[100*r['raw_successes']/r['n'] for r in rows if r['run'] in [f'v6b_{label}_seed{s}_validation' for s in [31,41,51]]]
        if vals:ax.scatter(x+np.linspace(-.08,.08,len(vals)),vals,label=label);ax.bar(x,np.mean(vals),alpha=.2)
    base=next((r for r in rows if r['run']=='v6b_damped_validation'),None)
    if base:ax.axhline(100*base['raw_successes']/base['n'],color='black',linestyle='--',label='damped feedback')
    ax.set(xticks=range(3),xticklabels=['PPO','residual PPO','learned proceed/hold'],ylim=(-2,102),ylabel='Raw validation success (%)',title='Development validation ONLY; no final-test tuning');ax.legend(fontsize=8)
    fig.savefig(root/'fig_validation_v6.png',dpi=160);plt.close(fig)
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='ARTIFACT_MANIFEST_V6.json']
    (root/'ARTIFACT_MANIFEST_V6.json').write_text(json.dumps(files,indent=2))
    print(json.dumps([{'run':r['run'],'success':r['raw_successes'],'n':r['n'],'verified':r['independent_successes']} for r in rows]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
