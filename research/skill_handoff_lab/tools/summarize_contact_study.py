"""Render only observed research results; missing runs never become zeros."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(root):
    rows=[];files=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/'manifest.json').exists():continue
        m=json.loads((directory/'manifest.json').read_text())
        if not (directory/'frozen_evaluation.json').exists():continue
        cases=json.loads((directory/'frozen_evaluation.json').read_text())
        audit=json.loads((directory/'INDEPENDENT_TASK_AND_CONTACT.json').read_text()) if (directory/'INDEPENDENT_TASK_AND_CONTACT.json').exists() else None
        credit=json.loads((directory/'credit_audits.json').read_text()) if (directory/'credit_audits.json').exists() else []
        row={'run':directory.name,'role':'heldout' if directory.name.startswith('heldout_') else 'development',
             'seed':m['seed'],'controller':m['controller'],'credit_mode':m.get('credit_mode','standard'),
             'interactions':m.get('online_training_interactions',0),'episodes':len(cases),
             'successes':sum(c['success'] for c in cases),
             'independent_successes':audit['independent_successes'] if audit else None,
             'success_time_s':float(np.mean([c['steps']/60 for c in cases if c['success']])) if any(c['success'] for c in cases) else None,
             'timeout_count':sum(c['timeout'] for c in cases),'physical_failure_count':sum(c['failed'] for c in cases),
             'success_by_family':{f:{'successes':sum(c['success'] for c in cases if c['family']==f),'n':sum(c['family']==f for c in cases)} for f in ['push','slot']},
             'credit_mixed_samples':sum(c['mixed_actor_samples'] for c in credit),
             'credit_rollouts_with_effect':sum(c['mixed_actor_samples']>0 for c in credit),
             'credit_rollouts_total':len(credit),'asset_sha256':m.get('asset_sha256'),'task_source_sha256':m.get('task_source_sha256')}
        if (directory/'actual_physics_trace.npz').exists():
            trace=np.load(directory/'actual_physics_trace.npz')['states']
            row['actor_clamped_rms']=float(np.sqrt(np.mean(trace[:,:,27:29]**2))) if trace.shape[-1]>=31 else None
            row['actual_object_max_displacement_m']=float(np.max(np.linalg.norm(trace[:,:,6:9]-trace[0,:,6:9],axis=-1)))
        rows.append(row)
        for path in directory.rglob('*'):
            if path.is_file():files.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (root/'RESULTS.json').write_text(json.dumps({'rows':rows,'limitations':['development cases used during debugging; not final test',
        'heldout uses only two parameter groups, not industrial/general contract transfer',
        'sampled contact audit is not full robot mesh swept-collision validation']},indent=2))
    (root/'ARTIFACT_MANIFEST.json').write_text(json.dumps(files,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,role in zip(axes,['development','heldout']):
        series=[('nominal',['contact_v5_nominal_eval'] if role=='development' else ['heldout_nominal']),
                ('fixed +X',['contact_v5_fixed_push_eval'] if role=='development' else ['heldout_fixed_push']),
                ('random',['contact_v5_random_eval'] if role=='development' else ['heldout_random']),
                ('PPO',[('contact_v5_standard_seed' if role=='development' else 'heldout_contact_v5_standard_seed')+str(s) for s in [31,41,51]]),
                ('boundary MC',[('contact_v5_boundary_seed' if role=='development' else 'heldout_contact_v5_boundary_seed')+str(s) for s in [31,41,51]]),
                ('all-phase MC',['contact_v5_all_mc_seed31'] if role=='development' else ['heldout_contact_v5_all_mc_seed31']),
                ('goal feedback',['contact_goal_feedback_eval'] if role=='development' else ['heldout_goal_feedback'])]
        for i,(label,names) in enumerate(series):
            vals=[100*r['successes']/r['episodes'] for r in rows if r['run'] in names]
            if not vals:continue
            ax.bar(i,np.mean(vals),color=['#777777','#aaaaaa','#dddddd','#377eb8','#e68613','#54a24b','#9467bd'][i],alpha=.75)
            ax.scatter(i+np.linspace(-.10,.10,len(vals)),vals,c='black',s=20,zorder=3)
            ax.text(i,min(100,max(vals)+3),f'n={len(vals)}',ha='center',fontsize=8)
        ax.set(xticks=range(len(series)),xticklabels=[s[0] for s in series],ylabel='Raw environment success (%)',ylim=(0,112),title=role+' / 64 same cases per run')
        ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Actual Isaac Sim 5.1 / Lab 2.3 physics; no synthetic success labels')
    fig.savefig(root/'fig_method_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for mode,color in [('standard','#377eb8'),('boundary','#e68613'),('all_mc','#54a24b')]:
        for seed in ([31] if mode=='all_mc' else [31,41,51]):
            d=root/f'contact_v5_{mode}_seed{seed}'
            if not (d/'training_episodes.json').exists():continue
            eps=json.loads((d/'training_episodes.json').read_text())
            xs=sorted(set(x['env_interactions'] for x in eps))
            yy=[100*np.mean([e['success'] for e in eps if x-8192<e['env_interactions']<=x]) for x in xs]
            ax.plot(xs,yy,color=color,alpha=.65,label=f'{mode} / seed {seed}')
    ax.set(xlabel='Real online environment interactions',ylabel='Completed-episode success, trailing 8192 (%)',title='Training curves, not frozen evaluation; 3 seeds where available',ylim=(-2,102))
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.savefig(root/'fig_learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    center=np.array(json.loads((root/'asset/asset_manifest.json').read_text())['paddle_center_body_m'])
    for ax,case in zip(axes,[0,1]):
        for name,label in [('contact_v5_nominal_eval','feedback'),('contact_v5_standard_seed31','PPO seed31'),('contact_v5_boundary_seed31','boundary seed31')]:
            p=root/name/'actual_physics_trace.npz'
            if not p.exists():continue
            tr=np.load(p)['states'];ids=tr[:,:,20]==case;points=tr[:,:,6:9][ids]
            if len(points):ax.plot((points[:,0]-center[0])*1000,(points[:,1]-center[1])*1000,label=label)
        if case==1:
            from matplotlib.patches import Rectangle
            for side in [-1,1]:ax.add_patch(Rectangle((100,-25+side*33.5-7.5),50,15,color='gray',alpha=.5))
        ax.set(title=f'Predeclared case {case}: '+('push' if case==0 else 'partial slot feed'),xlabel='Object X relative to tool start (mm)',ylabel='Object Y (mm)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Observed free-workpiece trajectories; same initial cases, not planned interpolation')
    fig.savefig(root/'fig_actual_contact_trajectories.png',dpi=160);plt.close(fig)
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
    boxes=[(0.2,3.1,2.3,1.1,'Observed robot / workpiece\ncurrent + next goal / phase'),
           (3.1,3.1,2.,1.1,'PPO Actor\nbounded XY increments'),
           (5.7,3.1,2.2,1.1,'DLS + joint limits\npose / rate constraints'),
           (8.5,3.1,3.1,1.1,'Real PhysX contact\nfree workpiece + slot rails'),
           (8.5,.5,3.1,1.2,'Observed task result\nstage / success / cost'),
           (3.1,.5,4.8,1.2,'On-policy actual interaction buffer\nGAE + completed-episode return mixing\nChanges Actor advantage; no label fabrication')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',facecolor='#e9f0f5',edgecolor='#446'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    for begin,end in [((2.55,3.65),(3.05,3.65)),((5.15,3.65),(5.65,3.65)),((7.95,3.65),(8.45,3.65)),
                      ((10,3.05),(10,1.76)),((8.45,1.1),(7.95,1.1)),((4.1,1.76),(4.1,3.05))]:
        ax.add_patch(FancyArrowPatch(begin,end,arrowstyle='-|>',mutation_scale=15,color='#345'))
    ax.text(.2,4.65,'Implemented prototype, not yet a validated novel skill-chaining method',fontsize=13)
    ax.text(.2,.1,'Isaac 5.1 / Lab 2.3 | 16 cloned physics environments | No field data or public-benchmark result claim',fontsize=9)
    fig.savefig(root/'fig_implemented_method.png',dpi=160);plt.close(fig)
    print(json.dumps([{'run':r['run'],'success':f"{r['successes']}/{r['episodes']}",'independent':r['independent_successes']} for r in rows],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
