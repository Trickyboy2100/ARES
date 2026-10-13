"""Independent held-envelope validation, restricted to the declared E0 model."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import yaml
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import CollisionBox, obb_overlap, arm_link_transforms
from ares_r.motion.independent_path_validation import validate_dense_world
from ares_r.motion.tcp_orientation import validate_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True)
    p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--robot',type=Path,required=True);a=p.parse_args()
    m=PinnedRobotModel(a.ares,a.ares_r)
    ignore=yaml.safe_load((m.urdf.parent/'jaka_minicobo_curobo.yml').read_text())['kinematics']['self_collision_ignore']
    reports=[]
    for job in sorted(a.execution.iterdir()):
        if not (job/'request.json').exists():continue
        request=json.loads((job/'request.json').read_text());result=json.loads((job/'result.json').read_text())
        points=np.asarray(result.get('trajectory_points_rad') or [])
        report={'candidate_id':request['candidate_id'],'model_scope':'E0_RESEARCH_HELD_CUBOID_HISTORICAL_LOWER_CHASSIS',
            'physical_success':False,'full_chassis_safety_qualified':False,'planning_result':result.get('observed_result'),
            'request_sha256':hashlib.sha256((job/'request.json').read_bytes()).hexdigest(),
            'result_sha256':hashlib.sha256((job/'result.json').read_bytes()).hexdigest(),
            'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if len(points)<2:
            report.update(model_validated_path=False,learning_outcome='UNKNOWN_NO_RETURNED_PATH')
        else:
            request['robot_yaml_urdf']=str(a.robot);world=validate_dense_world(points,request,subdivisions=4)
            dense=np.concatenate([before[None]+np.arange(4)[:,None]/4*(after-before)
                for before,after in zip(points,points[1:])]+[points[-1:]])
            self_collisions=set();held_collisions=set();corners=[]
            for q in dense:
                boxes=m.arm_boxes(q)
                for i,b in enumerate(boxes):
                    one=b.geometry_id.split('/')[-1];one='link6' if one=='gripper' else one
                    for c in boxes[i+1:]:
                        two=c.geometry_id.split('/')[-1];two='link6' if two=='gripper' else two
                        if one==two or two in ignore.get(one,[]) or one in ignore.get(two,[]):continue
                        if obb_overlap(b,c):self_collisions.add(b.geometry_id+' / '+c.geometry_id)
                t=m.body_base@arm_link_transforms(m.urdf,q)['link6']@np.asarray(request['attached_object_collision']['T_link6_object'])
                dims=np.asarray(request['tray_provenance']['observed_world_aabb_m'])
                held=CollisionBox('held','held','cuboid',tuple(t[:3,3]),tuple(map(tuple,t[:3,:3])),tuple(dims/2+.008),False,'E0')
                for b in boxes:
                    # Gripper-object contact is intended; no other held/arm exclusions.
                    if b.geometry_id=='right/gripper':continue
                    if obb_overlap(held,b):held_collisions.add(b.geometry_id)
                corners.append(t[:3,3])
            orientation=validate_path(m.fk,points,subdivisions=4,tolerance_deg=3.,
                target_rotation=np.asarray(request['orientation_lock']['target_R_body_tcp']))
            limits=bool(np.all(dense>=m.lower) and np.all(dense<=m.upper))
            errors=[float(np.max(np.abs(points[0]-request['start_rad']))),float(np.max(np.abs(points[-1]-request['goal_rad'])))]
            binding=result.get('scene_digest')==request['scene_digest']
            valid=world['collision_free'] and not self_collisions and not held_collisions and limits and orientation['passed'] and binding and max(errors)<.001
            tcp=np.array([m.fk(q)[:3,3] for q in points])
            report.update(world=world,self_collision_codes=sorted(self_collisions),held_arm_collision_codes=sorted(held_collisions),
                soft_limits=limits,orientation=orientation,scene_binding=binding,endpoint_errors_rad=errors,
                dense_samples=len(dense),model_validated_path=bool(valid),
                learning_outcome='MODEL_QUALIFIED_PATH' if valid else 'INDEPENDENT_VALIDATION_REJECTED',
                tcp_path_m=float(np.linalg.norm(np.diff(tcp,axis=0),axis=1).sum()),
                joint_path_rad=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                minimum_joint_margin_rad=float(min((dense-m.lower).min(),(m.upper-dense).min())),
                held_center_path_m=float(np.linalg.norm(np.diff(corners,axis=0),axis=1).sum()))
        (job/'e0_independent_validation.json').write_text(json.dumps(report,indent=2));reports.append(report)
    (a.execution/'e0_summary.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps({'jobs':len(reports),'model_validated_paths':sum(r['model_validated_path'] for r in reports)}))

if __name__=='__main__':main()
