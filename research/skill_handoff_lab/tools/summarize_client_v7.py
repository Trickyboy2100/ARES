"""Collect only completed synthetic v7 runs, preserving raw/independent distinction."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(path,rows,fields):
    with path.open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=fields);w.writeheader();w.writerows(rows)

def main(a):
    a.root=a.root.resolve();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True);rows=[];index=[];episodes=[];boundaries=[]
    for d in sorted(a.root.iterdir()):
        if not (d/'RUN_COMPLETE.json').exists() or not (d/'manifest.json').exists():continue
        m=json.loads((d/'manifest.json').read_text());evals=json.loads((d/'frozen_evaluation.json').read_text())
        auditpath=d/'INDEPENDENT_TASK_AND_CONTACT.json';audit=json.loads(auditpath.read_text()) if auditpath.exists() else None
        independent=None if audit is None else audit['independent_successes']
        modelsteps=m['online_training_interactions']
        cfg=json.loads((d/'run_config.json').read_text())
        if cfg.get('load_policy'):
            prior=a.root/Path(cfg['load_policy']).parent.name/'manifest.json'
            if prior.exists():modelsteps=json.loads(prior.read_text())['online_training_interactions']
        method=d.name.split('_seed')[0].replace('_validation','')
        if 'intervention' in d.name:method=d.name
        row=dict(method=method,seed=m['seed'],split=m['evaluation_role'],family='P2',n=len(evals),raw_successes=sum(r['success'] for r in evals),
                 independent_successes='' if independent is None else independent,interactions=modelsteps,run=str(d),
                 source_sha256=m['task_source_sha256'],audit_sha256=sha(auditpath) if audit else '')
        rows.append(row)
        good=[r for r in evals if r['success']]
        artifacts={}
        for name in ['manifest.json','scene.json','run_config.json','policy.zip','task_source.py','frame_source.py','frozen_evaluation.json',
                     'training_episodes.json','action_log.json','actual_physics_trace.npz','INDEPENDENT_TASK_AND_CONTACT.json']:
            path=d/name
            if path.exists():artifacts[name]={'sha256':sha(path),'bytes':path.stat().st_size,'local_path':str(path)}
        trace=np.load(d/'actual_physics_trace.npz');states=trace['states'];columns=list(trace['columns'])
        valid=np.isin(states[:,:,columns.index('case_id')].astype(int),[r['case_id'] for r in evals])
        actual=states[:,:,[columns.index('applied_dx'),columns.index('applied_dy')]][valid]
        requested=np.where(states[:,:,columns.index('stage')]==0,states[:,:,columns.index('switch_signal')].clip(-1,1)*.004,0)
        parameters=requested[valid]
        item=dict(row,online_training_interactions=m['online_training_interactions'],optimizer_updates=m['optimizer_updates'],
                  parameters_changed=m['parameters_changed'],wall_s=m['wall_s'],raw_success_case_ids=[r['case_id'] for r in good],
                  successful_completion_time_mean_s=None if not good else float(np.mean([r['steps']/60 for r in good])),
                  applied_xy_rms_m=float(np.sqrt(np.mean(actual**2))) if actual.size else None,
                  requested_contact_parameter_rms_m=float(np.sqrt(np.mean(parameters**2))) if parameters.size else None,
                  rms_scope='actual named evaluation cases only; invalid padding case -1 excluded',
                  limitations=m['limitations'],artifacts=artifacts)
        if audit:item['independent_success_case_ids']=[r['case_id'] for r in audit['cases'] if r['independent_task_pass']]
        index.append(item)
        for r in json.loads((d/'training_episodes.json').read_text()):
            episodes.append(dict(method=method,seed=m['seed'],run=d.name,interactions=r['env_interactions'],success=int(r['success']),
                                 reward=r['return'],phase=r['phase'],steps=r['steps'],distance_m=r['distance_m']))
        for r in evals:
            h=r.get('handoff_state')
            if not h:continue
            spec=m['scene_specs'][r['env']];phi=spec.get('direction',0);forward=np.array([math.cos(phi),math.sin(phi)]);lateral=np.array([-math.sin(phi),math.cos(phi)])
            pos=np.array(h['object_position']);paddle=np.array(h['paddle_position']);vel=np.array(h['object_linear_velocity']);omega=np.array(h['object_angular_velocity'])
            quat=h['object_quaternion_wxyz'];yaw=math.atan2(2*(quat[0]*quat[3]+quat[1]*quat[2]),1-2*(quat[2]**2+quat[3]**2))
            boundaries.append(dict(run=d.name,method=method,seed=m['seed'],case_id=r['case_id'],scene_group=r['scene_group'],success=int(r['success']),
                handoff_time_s=h['control_step']/60,terminal_time_s=r['steps']/60,
                speed_m_s=float(np.linalg.norm(vel)),angular_speed_rad_s=float(np.linalg.norm(omega)),yaw_error_rad=math.atan2(math.sin(yaw-phi),math.cos(yaw-phi)),
                paddle_object_forward_m=float(np.dot(pos[:2]-paddle[:2],forward)),paddle_object_lateral_m=float(np.dot(pos[:2]-paddle[:2],lateral)),
                object_x_m=pos[0],object_y_m=pos[1],object_z_m=pos[2],vx=vel[0],vy=vel[1],wz=omega[2]))
    fields=['method','seed','split','family','n','raw_successes','independent_successes','interactions','run','source_sha256','audit_sha256']
    write_csv(a.output/'NEW_PILOT_RESULTS.csv',rows,fields)
    if episodes:write_csv(a.output/'ONLINE_TRAINING_EPISODES.csv',episodes,list(episodes[0]))
    if boundaries:write_csv(a.output/'HANDOFF_PHYSICAL_STATES.csv',boundaries,list(boundaries[0]))
    report={'scope':'SYNTHETIC P2; validation-only bounded pilot; no field data','runs':index,
            'online_training_interactions':sum(i['online_training_interactions'] for i in index),
            'final_tests_evaluated':False,'source_sha256':sha(Path(__file__))}
    (a.output/'NEW_RUNS_INDEX.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'runs':len(index),'interactions':report['online_training_interactions']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
