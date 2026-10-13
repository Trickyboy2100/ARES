"""Package measured R1 evidence; never invent missing physical task labels."""
import argparse, hashlib, json, shutil, zipfile, io, gzip
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    metrics=[]
    for name in ['smoke1_v6','smoke4_v1','smoke16_camera_v2']:
        root=a.runs/name;m=json.loads((root/'manifest.json').read_text())
        with zipfile.ZipFile(root/'ppo_physics.zip') as z:
            optimizer=torch.load(io.BytesIO(z.read('policy.optimizer.pth')),map_location='cpu',weights_only=True)
        actual=max(int(v['step']) for v in optimizer['state'].values())
        rows=json.loads((root/'physics_interactions.json').read_text())
        metrics.append({'experiment':name,'num_envs':m['num_envs'],'physics_steps':m['physics_steps'],
            'env_physics_steps':m['env_physics_steps'],'adam_updates':actual,'raw_reported_updates':m['gradient_updates'],
            'policy_changed':m['policy_parameters_changed'],'reset_events':m['episodes'],
            'unique_origins':len(np.unique(m['origins'],axis=0)),
            'unique_targets_first_record':len(np.unique(rows[0]['per_env_target'],axis=0)),
            'max_contact_force_N':max(r['max_contact_force_N'] for r in rows)})
    (out/'parallel_physics_audit.json').write_text(json.dumps(metrics,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for item in metrics:
        rows=json.loads((a.runs/item['experiment']/'physics_interactions.json').read_text())
        x=[r['control_step']*item['num_envs'] for r in rows]
        axes[0].plot(x,[r['mean_reward'] for r in rows],label=f"{item['num_envs']} env")
        axes[1].plot(x,[r['mean_joint_error_rad'] for r in rows],label=f"{item['num_envs']} env")
    axes[0].set_ylabel('Measured joint tracking reward');axes[1].set_ylabel('Joint error norm (rad)')
    for ax in axes:ax.set_xlabel('Real environment interactions');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Isaac 5.1 / Isaac Lab 2.3: real PPO software smoke, seed 31\nNOT industrial task success or baseline superiority')
    fig.tight_layout();fig.savefig(out/'fig_r1_physics_learning.png',dpi=150);plt.close(fig)
    validations=[json.loads((a.runs/'mechanism_a_execution'/str(i)/'e0_independent_validation.json').read_text()) for i in range(5)]
    costs=[validations[0]['joint_path_rad'],sum(v['joint_path_rad'] for v in validations[1:3]),sum(v['joint_path_rad'] for v in validations[3:5])]
    comparison=[{'option':n,'planning_calls':c,'all_segments_model_qualified':all(v['model_validated_path'] for v in s),
        'joint_arc_rad':cost,'physical_task_success':'NOT_TESTED'} for n,c,s,cost in zip(['Nominal','Prepare standoff','Prepare lift'],[1,2,2],[validations[:1],validations[1:3],validations[3:5]],costs)]
    (out/'same_start_comparison.json').write_text(json.dumps(comparison,indent=2))
    physical=[]
    fig,ax=plt.subplots(figsize=(7,5))
    for name,label in [('replay_feedback_v1','Nominal + feedback hold'),('replay_standoff_v1','Prepare standoff + hold'),('replay_lift_v1','Prepare lift + hold')]:
        root=a.runs/name
        if not (root/'independent_physx_validation.json').is_file():continue
        validation=json.loads((root/'independent_physx_validation.json').read_text())
        physical.append({'experiment':name,**validation})
        rows=json.loads((root/'replay.json').read_text());xyz=np.array([r['link6_pos_w'][0] for r in rows])
        ax.plot(xyz[:,0],xyz[:,2],label=label)
    ax.set_xlabel('Measured Link6 BODY X (m)');ax.set_ylabel('Measured Link6 BODY Z (m)')
    ax.set_title('Same-start measured PhysX trajectories\nIsolated arm / welded proxy; NOT complete industrial safety')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'fig_r1_physx_paths.png',dpi=150);plt.close(fig)
    (out/'physical_option_comparison.json').write_text(json.dumps(physical,indent=2))
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([x['option'] for x in comparison],costs)
    ax.set_ylabel('Total planner joint arc (rad)');ax.set_title('Same start / same goal: 5 real cuRobo segments\nResearch proxy model only; no physical task labels')
    for i,(cost,calls) in enumerate(zip(costs,[1,2,2])):ax.text(i,cost,f'{cost:.4f}\n{calls} calls',ha='center',va='bottom')
    ax.set_ylim(0,max(costs)*1.2);fig.tight_layout();fig.savefig(out/'fig_r1_same_start.png',dpi=150);plt.close(fig)
    for name in ['physical_scene.png','online_ppo_physics.gif','video_frame_records.json']:
        shutil.copy2(a.runs/'smoke16_camera_v2'/name,out/name)
    raw=out/'raw';raw.mkdir(exist_ok=True)
    for name in ['smoke1_v6','smoke4_v1','smoke16_camera_v2','smoke16_camera_v1',
                 'replay_nominal_v1','replay_nominal_v2','replay_feedback_v1',
                 'mechanism_a_execution','mechanism_a_requests','prepare_chains',
                 'replay_standoff_v1','replay_lift_v1']:
        if (a.runs/name).is_dir():shutil.copytree(a.runs/name,raw/name,dirs_exist_ok=True)
    for f in a.runs.glob('*'):
        if f.is_file() and (f.suffix in ['.log','.json']):shutil.copy2(f,raw/f.name)
    if not (raw/'smoke1_v6/runtime_source.py').exists():
        shutil.copy2(raw/'smoke4_v1/runtime_source.py',raw/'smoke1_v6/runtime_source.py')
    for name in ['replay_standoff_v1','replay_lift_v1']:
        if (raw/name).is_dir():shutil.copy2(raw/'replay_feedback_v1/runtime_source.py',raw/name/'runtime_source.py')
    asset=raw/'asset_v4';asset.mkdir(exist_ok=True)
    shutil.copy2(a.runs/'asset_v4/asset_manifest.json',asset/'asset_manifest.json')
    with (asset/'jaka_held.usda.gz').open('wb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as compressed:
            compressed.write((a.runs/'asset_v4/jaka_held.usda').read_bytes())
    artifacts=[]
    for f in sorted(a.runs.rglob('*')):
        if f.is_file() and 'jaka_held.usda' not in f.name:
            artifacts.append({'path':str(f.relative_to(a.runs)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    (out/'R1_DATA_MANIFEST.json').write_text(json.dumps({'base_commit':'f93cd24','artifacts':artifacts,
        'scope':'real isolated-arm PhysX PPO smoke and same-start planning; not formal RL advantage'},indent=2))
    bundle=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        for f in sorted(out.rglob('*')) if f.is_file() and f.name!='BUNDLE_MANIFEST.json']
    (out/'BUNDLE_MANIFEST.json').write_text(json.dumps({'base_commit':'f93cd24','artifacts':bundle,
        'source_run_manifest':'R1_DATA_MANIFEST.json describes the larger preserved local run; this manifest describes packaged files'},indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
