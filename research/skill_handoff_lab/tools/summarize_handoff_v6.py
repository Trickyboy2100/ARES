"""Observed M0 / paired-mechanism / online RL results. Unknown != failure."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.stats import t as student_t

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
        row={'run':directory.name,'run_complete':(directory/'RUN_COMPLETE.json').exists(),'software_exception':(directory/'exception.txt').exists(),
            'seed':m['seed'],'role':m.get('evaluation_role'),'task':m.get('task_version'),
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
            if ev and 'phase_before' in ev[0] and raw.shape[2]>2:
                phase=np.array([x['phase_before'] for x in ev]);mask=phase==1
                row['phase1_negative_switch_fraction']=float(np.mean(raw[:,:,2][mask]<0)) if mask.any() else None
        credit=load(directory/'credit_audits.json')
        row['credit_updates']=len(credit)
        if credit and 'samples' in credit[0]:
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
    frozen={}
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,split in zip(axes,['test_id','test_geometry','test_dynamics']):
        frozen[split]={}
        for i,method in enumerate(['ppo','residual','recover']):
            names=[f'v6b_{split}_v6b_{method}_seed{s}' for s in [31,41,51]]
            rs=[next((r for r in rows if r['run']==n),None) for n in names]
            values=[100*r['independent_successes']/r['n'] if r and r['independent_successes'] is not None else None for r in rs]
            frozen[split][method]={'seeds':[31,41,51],'independent_percent':values,
                'raw_successes':[r['raw_successes'] if r else None for r in rs]}
            if all(v is not None for v in values):
                frozen[split][method].update(mean_percent=float(np.mean(values)),sample_sd_percent=float(np.std(values,ddof=1)))
                ax.bar(i,np.mean(values),alpha=.25);ax.scatter(i+np.linspace(-.08,.08,3),values,c=['#377eb8','#e68613','#54a24b'],s=25)
        base=next((r for r in rows if r['run']==f'v6b_{split}_damped'),None)
        if base and base['independent_successes'] is not None:
            frozen[split]['damped_percent']=100*base['independent_successes']/base['n'];ax.axhline(frozen[split]['damped_percent'],ls='--',c='black',label='damped control')
        ax.set(title=split,xticks=range(3),xticklabels=['PPO','residual','hold/proceed'],ylim=(-2,102),ylabel='Independent success (%)')
        ax.tick_params(axis='x',rotation=20);ax.grid(alpha=.2)
    fig.suptitle('Fixed 64k checkpoints / same 32 cases per split / three independent training seeds')
    fig.savefig(root/'fig_frozen_test_v6.png',dpi=160);plt.close(fig)
    paired={}
    for split in ['test_id','test_geometry','test_dynamics']:
        base={x['case_id']:x for x in load(root/f'v6b_{split}_damped'/'frozen_evaluation.json')}
        paired[split]={}
        for method in ['residual','recover']:
            stats=[]
            for seed in [31,41,51]:
                es=load(root/f'v6b_{split}_v6b_{method}_seed{seed}'/'frozen_evaluation.json')
                common=[e for e in es if e['success'] and base[e['case_id']]['success']]
                stats.append({'seed':seed,'matched_success_cases':len(common),
                    'matched_completion_time_delta_s':float(np.mean([(e['steps']-base[e['case_id']]['steps'])/60 for e in common])) if common else None,
                    'rescued_cases':[e['case_id'] for e in es if e['success'] and not base[e['case_id']]['success']],
                    'lost_cases':[e['case_id'] for e in es if not e['success'] and base[e['case_id']]['success']],
                    'failure_counts':{f:sum(e['failure_reason']==f for e in es) for f in sorted(set(e['failure_reason'] for e in es))}})
            paired[split][method]=stats
    aggregate={}
    for method in ['residual','recover']:
        values=np.array([np.mean([frozen[s][method]['independent_percent'][i] for s in frozen]) for i in range(3)])
        mean=float(values.mean());sd=float(values.std(ddof=1));half=float(student_t.ppf(.975,2)*sd/np.sqrt(3))
        aggregate[method]={'seed_macro_percent':values.tolist(),'mean_percent':mean,'sample_sd_percent':sd,
            't95_interval_percent_unclipped':[mean-half,mean+half],
            'warning':'Only three training seeds, same frozen cases; not 288 independent task trials or a significance claim'}
    (root/'RESULTS_V6.json').write_text(json.dumps({'rows':rows,'mechanism':mechanism,'frozen':frozen,'paired_vs_feedback':paired,'aggregate':aggregate},indent=2))
    diagnostic=root/'v6b_contact_history_diagnostic'/'post_step_contact_history.npz'
    if diagnostic.exists():
        data=np.load(diagnostic)['states'];fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
        for col,case in enumerate([1,3]):
            x=data[data[:,:,1]==case];t=(x[:,0]-x[0,0])/120
            axes[0,col].plot(t,x[:,3],label='robot net-contact proxy (N)');axes[0,col].legend(fontsize=8)
            axes[1,col].plot(t,np.linalg.norm(x[:,7:10],axis=-1)*1000,label='linear speed mm/s')
            twin=axes[1,col].twinx();twin.plot(t,np.linalg.norm(x[:,10:13],axis=-1),color='orange',label='angular speed rad/s');twin.set_ylabel('rad/s',color='orange')
            axes[0,col].set_title(f'Actual diagnostic case {case} / '+('failure' if case==1 else 'success'))
            axes[1,col].set(xlabel='Actual episode physics time (s)',ylabel='mm/s');axes[1,col].legend(fontsize=8)
        fig.suptitle('Post-control-step measured simulation contact proxy; not calibrated hardware force')
        fig.savefig(root/'fig_contact_failure_diagnostic.png',dpi=160);plt.close(fig)
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
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');ax.set(xlim=(0,11),ylim=(0,5));ax.axis('off')
    boxes=[(1.7,3.8,'Current exact simulation state\nq/qd, object pose/v/w, tool\ngeometry and given dynamics'),
        (5.5,3.8,'Same 56D observation / PPO\nfeedback + XY residual\noptional proceed / hold gate'),
        (9.3,3.8,'Shared DLS / servo / limits\nactual PhysX contacts\n120 Hz physics, 60 Hz control'),
        (9.3,1.45,'Exact post-step, before reset\n30-control stable completion\nsampled independent FCL audit'),
        (5.5,1.45,'Actual reward -> PPO update\nGAE main baseline\nno future label to Actor'),
        (1.7,1.45,'Frozen scene groups / seeds\npaired mechanism tests\nraw data + source hashes')]
    for x,y,label in boxes:ax.text(x,y,label,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.65',fc='#eef3f7',ec='#37566c'))
    for p,q in [((3.1,3.8),(4.05,3.8)),((6.95,3.8),(7.8,3.8)),((9.3,3.15),(9.3,2.15)),((7.8,1.45),(7,1.45)),((5.5,2.15),(5.5,3.1)),((4,1.45),(3.2,1.45))]:
        ax.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='->',color='#37566c',lw=1.5))
    ax.text(5.5,.25,'Implemented research interfaces; hold / active-likelihood increment NOT validated superior to residual PPO',ha='center',fontsize=9,color='#a43328')
    fig.suptitle('ARES contact RL / implemented mechanism and evidence boundary')
    fig.savefig(root/'fig_method_v6.png',dpi=160);plt.close(fig)
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted(root.rglob('*')) if p.is_file() and p.name!='ARTIFACT_MANIFEST_V6.json']
    (root/'ARTIFACT_MANIFEST_V6.json').write_text(json.dumps(files,indent=2))
    print(json.dumps([{'run':r['run'],'success':r['raw_successes'],'n':r['n'],'verified':r['independent_successes']} for r in rows]))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
