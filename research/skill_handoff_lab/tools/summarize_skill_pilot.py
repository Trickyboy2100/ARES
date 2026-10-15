"""Private, measured-only skill pilot figures and source manifests."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from controlled_paths import require_private_location


def main(a):
    require_private_location(a.root);os.umask(0o077)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=a.root;dataset=json.loads((root/'SKILL_RL_DATASET.json').read_text())
    fig,axis=plt.subplots(figsize=(13,5));axis.set(xlim=(0,13),ylim=(0,5));axis.axis('off')
    def node(x,y,label):
        axis.text(x,y,label,ha='center',va='center',fontsize=10,
                  bbox={'boxstyle':'round,pad=.6','facecolor':'#e9f2f7','edgecolor':'#416881'})
    for x,label in [(1.5,'Observed q / goal\nremaining budget'),(4.8,'MaskablePPO\npermuted candidates'),
                    (8.1,'Exact-start real\ncuRobo transition cache'),(11.3,'Qualified next q\ncost / task outcome')]:node(x,3.6,label)
    for x,y in [(2.7,4),(6.1,7.1),(9.2,10.3)]:axis.annotate('',xy=(y,3.6),xytext=(x,3.6),arrowprops={'arrowstyle':'->'})
    axis.text(.2,4.65,'IMPLEMENTED TRAINING: planning-only, 3 seeds; no PhysX claim',fontsize=12)
    axis.annotate('',xy=(1.5,3.05),xytext=(11.3,3.05),arrowprops={'arrowstyle':'->','connectionstyle':'arc3,rad=-.12'})
    for x,label in [(1.5,'Articulation drives\nactual PhysX q'),(4.8,'Observed terminal q\nno state rewrite'),
                    (8.1,'Live cuRobo +\nindependent qualification'),(11.3,'Next physical Option\nactual trajectory checks')]:node(x,1.1,label)
    for x,y in [(2.7,3.7),(6.1,7),(9.2,10.1)]:axis.annotate('',xy=(y,1.1),xytext=(x,1.1),arrowprops={'arrowstyle':'->'})
    axis.text(.2,2.15,'INDEPENDENT INTEGRATION: scripted 2-Option execution; policy-to-PhysX wiring NOT TESTED',fontsize=11)
    axis.text(.2,.1,'CONTROLLED | conservative attached-object / scene proxies; NOT tray CAD, field execution or industrial safety certification',fontsize=9)
    fig.tight_layout();destination=root/'mentor_package/01_METHOD';destination.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination/'METHOD_IMPLEMENTED.png',dpi=160);plt.close(fig)
    train=root/a.training;ev=json.loads((train/'FROZEN_EVALUATION.json').read_text())
    summary=[]
    for method in sorted({r['method'] for r in ev}):
        rows=[r for r in ev if r['method']==method]
        groups=sorted({r.get('seed',0) for r in rows})
        rates=[np.mean([r['success'] for r in rows if r.get('seed',0)==s]) for s in groups]
        summary.append({'method':method,'episodes':len(rows),'independent_test_geometry_groups':len({r['task_id'] for r in rows}),
            'success_rate':float(np.mean(rates)),'seed_success_std':float(np.std(rates)),
            'logical_request_calls_mean':float(np.mean([r['calls'] for r in rows])),
            'internal_attempt_cap_sum_mean':float(np.mean([r['attempt_cap_sum'] for r in rows])),
            'measured_solve_plus_planned_execution_s_mean':float(np.mean([r['planned_option_time_s'] for r in rows])),
            'successful_two_stage_episodes':sum(r['success'] and len(r['decisions'])>=2 for r in rows),
            'new_online_planning_calls':0,'evidence':'planning-only real cuRobo cache'})
    (train/'STATISTICAL_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    labels=['Direct FAST','Direct FALLBACK','Transit FAST','Lift FAST',
            'Transit -> Preland FAST','Transit -> Preland FALLBACK','Lift -> Preland FAST','Lift -> Preland FALLBACK']
    matrix=[];solve=[]
    for task in dataset['tasks']:
        name=task['task_id']
        ids=[name+'__CENTER__PRELAND__FAST',name+'__CENTER__PRELAND__FALLBACK',
             name+'__CENTER__TRANSIT__FAST',name+'__CENTER__LIFT__FAST',
             name+'__CENTER__TRANSIT__FAST__NEXT_FAST',name+'__CENTER__TRANSIT__FAST__NEXT_FALLBACK',
             name+'__CENTER__LIFT__FAST__NEXT_FAST',name+'__CENTER__LIFT__FAST__NEXT_FALLBACK']
        matrix.append([int(dataset['edges'][e]['qualified']) for e in ids])
        solve.append([dataset['edges'][e]['planning_wall_s'] for e in ids])
    fig,axes=plt.subplots(1,2,figsize=(16,5))
    axes[0].imshow(matrix,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
    im=axes[1].imshow(solve,cmap='viridis',aspect='auto');fig.colorbar(im,ax=axes[1],label='measured solve s; excludes warmup/qualification')
    for axis in axes:
        axis.set_xticks(range(8),labels,rotation=65,ha='right')
        axis.set_yticks(range(6),[t['task_id']+' ['+t['split']+']' for t in dataset['tasks']])
    for i,row in enumerate(matrix):
        for j,v in enumerate(row):axes[0].text(j,i,'PASS' if v else 'REJECT',ha='center',va='center',fontsize=7)
    axes[0].set_title('Independent execution-model qualification (NOT PhysX)')
    axes[1].set_title('Actual per-candidate cuRobo planning solve time')
    fig.suptitle('CONTROLLED | six field-derived station offsets, not six independent scans | all 48 branches retained')
    fig.tight_layout();fig.savefig(root/'CANDIDATE_COMPARISON.png',dpi=160);plt.close(fig)
    physical=root/a.physical
    if (physical/'PHYSX_STATES.npz').exists():
        data=np.load(physical/'PHYSX_STATES.npz');rows=data['rows'];phase=data['phase'];dt=1/120
        vel=np.diff(rows[:,1:7],axis=0)/dt;acc=np.diff(vel,axis=0)/dt
        measured={'physics_steps':len(rows),'phase_counts':{str(i):int(np.sum(phase==i)) for i in np.unique(phase)},
            'max_actual_velocity_rad_s':float(abs(vel).max()),'max_actual_acceleration_rad_s2':float(abs(acc).max()),
            'max_command_velocity_rad_s':float(abs(np.diff(rows[:,7:13],axis=0)/dt).max()),
            'max_command_acceleration_rad_s2':float(abs(np.diff(rows[:,7:13],n=2,axis=0)/dt**2).max()),
            'minimum_soft_margin_rad':float(rows[:,18].min()),'max_level_deg':float(rows[:,16].max()),
            'max_net_force_N':float(rows[:,17].max()),'scope':'actual PhysX states, research proxy model; not contact certification'}
        (physical/'MEASURED_DYNAMICS.json').write_text(json.dumps(measured,indent=2))
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for i in range(6):
            axes[0,0].plot(rows[:,0],rows[:,i+1],label=f'J{i+1}')
            axes[0,1].plot(rows[2:,0],acc[:,i])
        axes[0,0].legend(ncol=3);axes[0,0].set(ylabel='actual q (rad)',xlabel='physics time s')
        axes[0,1].axhline(.2,color='r',ls='--');axes[0,1].axhline(-.2,color='r',ls='--')
        axes[0,1].set(ylabel='actual acceleration rad/s^2',xlabel='physics time s')
        axes[1,0].plot(rows[:,0],rows[:,16]);axes[1,0].set(ylabel='actual TCP level error deg',xlabel='physics time s')
        for i in np.unique(phase):
            part=rows[phase==i];axes[1,1].plot(part[:,13],part[:,15],label=f'physical Option {i}')
        axes[1,1].set(xlabel='BODY TCP X m',ylabel='BODY TCP Z m');axes[1,1].legend()
        fig.suptitle('CONTROLLED | actual PhysX, live observed-state Oracle boundary; NOT online RL / field execution')
        fig.tight_layout();fig.savefig(root/'PHYSX_TWO_OPTION_EXECUTION.png',dpi=160);plt.close(fig)
    records=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='PILOT_ARTIFACT_MANIFEST.json':
            records.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (root/'PILOT_ARTIFACT_MANIFEST.json').write_text(json.dumps({'controlled_not_public':True,'artifacts':records},indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--training',default='training_v2');p.add_argument('--physical',default='physx_live_chain_v2');main(p.parse_args())
