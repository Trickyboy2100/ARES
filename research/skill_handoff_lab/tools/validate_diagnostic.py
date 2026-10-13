"""Independent limited-scope trajectory validation; no handoff success label."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import obb_overlap
from ares_r.motion.independent_path_validation import validate_dense_world

def main():
    p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True)
    p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--robot',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.job/'request.json').read_text());result=json.loads((a.job/'result.json').read_text())
    points=np.asarray(result['trajectory_points_rad']);model=PinnedRobotModel(a.ares,a.ares_r)
    r['robot_yaml_urdf']=str(a.robot);world=validate_dense_world(points,r,subdivisions=4)
    cfg=yaml.safe_load((model.urdf.parent/'jaka_minicobo_curobo.yml').read_text())
    ignore=cfg['kinematics']['self_collision_ignore'];collisions=set()
    dense=np.concatenate([before[None]+np.arange(4)[:,None]/4*(after-before)
        for before,after in zip(points,points[1:])]+[points[-1:]])
    for q in dense:
        boxes=model.arm_boxes(q)
        for i,first in enumerate(boxes):
            one=first.geometry_id.split('/')[-1];one='link6' if one=='gripper' else one
            for second in boxes[i+1:]:
                two=second.geometry_id.split('/')[-1];two='link6' if two=='gripper' else two
                if one==two or two in ignore.get(one,[]) or one in ignore.get(two,[]):continue
                if obb_overlap(first,second):collisions.add(first.geometry_id+' / '+second.geometry_id)
    initial=model.fk(points[0])[:3,:3]
    orientation=max(float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(
        initial.T@model.fk(q)[:3,:3]).as_rotvec()))) for q in dense)
    limits=bool(np.all(dense>=model.lower) and np.all(dense<=model.upper))
    errors=[float(np.max(np.abs(points[0]-r['start_rad']))),float(np.max(np.abs(points[-1]-r['goal_rad'])))]
    data={'evidence':'INDEPENDENT_DIAGNOSTIC_VALIDATION_NOT_HANDOFF',
        'world':world,'soft_limits':limits,'self_collision_obb_free':not collisions,
        'self_collision_codes':sorted(collisions),'self_collision_exclusions':'unchanged ARES fixed YAML adjacency policy',
        'orientation_drift_from_start_deg':orientation,'diagnostic_drift_limit_deg':3.,
        'endpoint_joint_errors_rad':errors,'dense_samples':len(dense),
        'diagnostic_scope_pass':world['collision_free'] and limits and not collisions and orientation<=3 and max(errors)<.001,
        'held_object':False,'physical_success':False,'usable_for_handoff_training':False,
        'request_sha256':hashlib.sha256((a.job/'request.json').read_bytes()).hexdigest(),
        'result_sha256':hashlib.sha256((a.job/'result.json').read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.job/'diagnostic_independent_validation.json').write_text(json.dumps(data,indent=2));print(json.dumps(data))

if __name__=='__main__':main()
