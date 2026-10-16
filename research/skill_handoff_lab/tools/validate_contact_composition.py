"""Independent actual-state task/OBB contact audit. Not full-arm CCD certification."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import fcl
from ares_r.perception.robot_collision import arm_link_transforms

def main(a):
    meta=json.loads((a.asset/'asset_manifest.json').read_text());run=json.loads((a.run/'manifest.json').read_text())
    data=np.load(a.run/'actual_physics_trace.npz');states=data['states'];columns=list(data['columns'])
    if 'case_id' not in columns or 'obj_qw' not in columns:raise ValueError('full observed state trace required; no silent proxy quaternion')
    results={x['case_id']:x for x in json.loads((a.run/'frozen_evaluation.json').read_text())}
    center=np.array(meta['paddle_center_body_m']);base=np.array(meta['body_bases']['right'])
    local_rotation=np.array(meta['level_link6_rotation']).T
    table=fcl.CollisionObject(fcl.Box(.26,.30,.03),fcl.Transform(np.eye(3),center+[.145,0,-.04]))
    specs=run.get('scene_specs',[{'goal_x':.110,'slot_y':-.025,'gap':.052}]*len(run['families']))
    rails_by_env=[]
    for spec in specs:
        rot=Rotation.from_euler('z',spec.get('direction',0.)).as_matrix()
        length=spec.get('rail_length',.05);entry=spec.get('entry_x',spec['goal_x']-.01)
        rails_by_env.append([fcl.CollisionObject(fcl.Box(length,.015,.04),fcl.Transform(rot,
            center+rot@np.array([entry+length/2,spec['slot_y']+s*(spec['gap']/2+.0075),-.005]))) for s in [-1,1]])
    cube=fcl.CollisionObject(fcl.Box(.04,.04,.04));paddle=fcl.CollisionObject(fcl.Box(*meta['paddle_dims_m']))
    request=fcl.CollisionRequest(num_max_contacts=8,enable_contact=True);summary={};last={};control_samples={}
    def depth(left,right):
        result=fcl.CollisionResult();fcl.collide(left,right,request,result)
        return max((float(c.penetration_depth) for c in result.contacts),default=0.) if result.is_collision else 0.
    for step_index,step in enumerate(states):
        for i,row in enumerate(step):
            rails=rails_by_env[i]
            case=int(row[columns.index('case_id')])
            if case<0 or case not in results:continue
            obj=row[6:9];held=row[9:12];q=row[:6]
            r=summary.setdefault(case,{'case_id':case,'family':run['families'][i],'samples':0,'max_unintended_penetration_m':0.,
                'max_paddle_workpiece_penetration_m':0.,'max_cube_table_penetration_m':0.,'max_cube_rail_penetration_m':0.,
                'min_soft_margin_rad':float('inf'),'max_observed_fk_paddle_error_m':0.,'visited_phases':[]})
            r['samples']+=1;phase=int(row[15]);r['visited_phases']=sorted(set(r['visited_phases']+[phase]))
            tick=int(row[columns.index('physics_tick')]) if 'physics_tick' in columns else run.get('training_physics_steps',0)+step_index
            if tick%2==0:
                control_samples.setdefault(case,[]).append(row.copy())
            if phase==1 and case in last and int(last[case][15])==0:
                r['actual_stage0_boundary_error_m']=float(np.linalg.norm(obj[:2]-last[case][24:26]))
            t=base@arm_link_transforms(a.urdf,q)['link6'];fk=t[:3,3]+t[:3,:3]@np.array(meta['held_offset_link6_m'])
            r['max_observed_fk_paddle_error_m']=max(r['max_observed_fk_paddle_error_m'],float(np.linalg.norm(fk-held)))
            quat=row[16:20];rot=Rotation.from_quat([*quat[1:],quat[0]]).as_matrix()
            cube.setTransform(fcl.Transform(rot,obj));paddle.setTransform(fcl.Transform(t[:3,:3]@local_rotation,held))
            dt=depth(cube,table);dp=depth(paddle,cube);dr=max((depth(cube,x) for x in rails),default=0.) if r['family']=='slot' else 0.
            pr=max((depth(paddle,x) for x in rails),default=0.) if r['family']=='slot' else 0.
            r['max_cube_table_penetration_m']=max(r['max_cube_table_penetration_m'],dt)
            r['max_cube_rail_penetration_m']=max(r['max_cube_rail_penetration_m'],dr)
            r['max_paddle_workpiece_penetration_m']=max(r['max_paddle_workpiece_penetration_m'],dp)
            r['max_unintended_penetration_m']=max(r['max_unintended_penetration_m'],depth(paddle,table),pr)
            r['min_soft_margin_rad']=min(r['min_soft_margin_rad'],float(np.minimum(q-meta['joint_lower'],np.array(meta['joint_upper'])-q).min()))
            last[case]=row.copy()
    for case,r in summary.items():
        row=last[case];r['raw_task_success']=bool(results[case]['success'])
        if 'post_step_state' in results[case]:
            post=results[case]['post_step_state'];row=row.copy()
            row[:6]=post['q'];row[6:9]=post['object_position'];row[9:12]=post['paddle_position']
            row[12:15]=post['object_linear_velocity'];row[16:20]=post['object_quaternion_wxyz'];row[24:27]=post['goal'];row[15]=post['phase']
            if len(row)>33:row[31:34]=post['object_angular_velocity']
            control_samples.setdefault(case,[]).append(row)
            h=results[case].get('handoff_state')
            if h:r['actual_stage0_boundary_error_m']=float(np.linalg.norm(np.array(h['object_position'])[:2]-np.array(h['goal'])[:2]))
            r['terminal_source']='EXACT_POST_CONTROL_STEP_BEFORE_RESET'
            q=row[:6];t=base@arm_link_transforms(a.urdf,q)['link6'];quat=row[16:20]
            r['min_soft_margin_rad']=min(r['min_soft_margin_rad'],float(np.minimum(q-meta['joint_lower'],np.array(meta['joint_upper'])-q).min()))
            fk=t[:3,3]+t[:3,:3]@np.array(meta['held_offset_link6_m'])
            r['max_observed_fk_paddle_error_m']=max(r['max_observed_fk_paddle_error_m'],float(np.linalg.norm(fk-row[9:12])))
            cube.setTransform(fcl.Transform(Rotation.from_quat([*quat[1:],quat[0]]).as_matrix(),row[6:9]))
            paddle.setTransform(fcl.Transform(t[:3,:3]@local_rotation,row[9:12]))
            r['max_cube_table_penetration_m']=max(r['max_cube_table_penetration_m'],depth(cube,table))
            r['max_paddle_workpiece_penetration_m']=max(r['max_paddle_workpiece_penetration_m'],depth(paddle,cube))
            rails=rails_by_env[results[case]['env']]
            if r['family']=='slot':
                r['max_cube_rail_penetration_m']=max(r['max_cube_rail_penetration_m'],max(depth(cube,x) for x in rails))
                r['max_unintended_penetration_m']=max(r['max_unintended_penetration_m'],max(depth(paddle,x) for x in rails))
            r['max_unintended_penetration_m']=max(r['max_unintended_penetration_m'],depth(paddle,table))
        r['independent_goal_distance_m']=float(np.linalg.norm(row[6:8]-row[24:26]))
        r['independent_speed_m_s']=float(np.linalg.norm(row[12:15]))
        r['geometry_pass']=bool(r['max_unintended_penetration_m']<=.001 and r['max_cube_table_penetration_m']<=.001
            and r['max_cube_rail_penetration_m']<=.001 and r['max_paddle_workpiece_penetration_m']<=.002)
        r['independent_task_pass']=bool(r['raw_task_success'] and r['geometry_pass'] and r['visited_phases']==[0,1]
            and r.get('actual_stage0_boundary_error_m',float('inf'))<=.00901 and r['independent_goal_distance_m']<=.00801
            and r['independent_speed_m_s']<=.0251 and r['min_soft_margin_rad']>=-.001 and r['max_observed_fk_paddle_error_m']<1e-4)
        if 'post_step_state' in results[case]:
            contract=run['success_contract'];hold=contract['hold_control_steps'];samples=control_samples[case][-hold:]
            spec=specs[results[case]['env']]
            def stable(x):
                quat=x[16:20];yaw=Rotation.from_quat([*quat[1:],quat[0]]).as_euler('xyz')[2]
                err=abs(np.arctan2(np.sin(4*(yaw-spec.get('direction',0))),np.cos(4*(yaw-spec.get('direction',0))))/4)
                return (int(x[15])==1 and np.linalg.norm(x[6:8]-x[24:26])<contract['position_m']+1e-6
                    and np.linalg.norm(x[12:15])<contract['linear_speed_m_s']+1e-6
                    and abs(x[8]-(center[2]-.005))<.008+1e-6
                    and (contract['angular_speed_rad_s'] is None or (np.linalg.norm(x[31:34])<contract['angular_speed_rad_s']+1e-6
                        and err<contract['yaw_error_rad']+1e-6 and 1-2*(quat[1]**2+quat[2]**2)>np.cos(contract['upright_rad'])-1e-6)))
            enough_angular=contract['angular_speed_rad_s'] is None or len(row)>33
            r['hold_audit_status']='COMPLETE' if enough_angular else 'UNKNOWN_MISSING_ANGULAR_HISTORY'
            r['independent_stable_hold_pass']=enough_angular and len(samples)==hold and all(stable(x) for x in samples)
            r['independent_task_pass']=bool(r['independent_task_pass'] and r['independent_stable_hold_pass'])
    report={'evidence':'ACTUAL_PHYSX_SAMPLED_CONTACT_AND_TASK_AUDIT','physics_dt_s':float(data['sample_dt_s']),
        'raw_successes':sum(x['success'] for x in results.values()),'independent_successes':sum(x['independent_task_pass'] for x in summary.values()),
        'cases':list(summary.values()),'trace_sha256':hashlib.sha256((a.run/'actual_physics_trace.npz').read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'thresholds_m':{'unintended':.001,'cube_table':.001,'cube_rail':.001,'intentional_paddle_cube':.002},
        'limitations':['sampled OBB contacts, not swept CCD','arm/gripper mesh vs table and full self-collision not independently rechecked here',
            'not a field execution or calibrated force/safety certificate']}
    (a.run/'INDEPENDENT_TASK_AND_CONTACT.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['run','asset','urdf']:p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
