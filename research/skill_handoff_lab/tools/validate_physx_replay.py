"""Offline measured-state validation; isolated-arm scope is not full safety."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import arm_link_transforms

def main():
    p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--request',type=Path,required=True)
    a=p.parse_args();model=PinnedRobotModel(a.ares,a.ares_r)
    rows=json.loads((a.replay/'replay.json').read_text());request=json.loads(a.request.read_text())
    pos_errors=[];rot_errors=[];orientation=[];margins=[]
    target=np.array(request['orientation_lock']['target_R_body_tcp'])
    for row in rows:
        q=np.array(row['q_actual'][0]);pred=model.body_base@arm_link_transforms(model.urdf,q)['link6']
        measured=np.eye(4);measured[:3,3]=row['link6_pos_w'][0]
        w,x,y,z=row['link6_quat_w'][0];measured[:3,:3]=Rotation.from_quat([x,y,z,w]).as_matrix()
        pos_errors.append(float(np.linalg.norm(pred[:3,3]-measured[:3,3])))
        rot_errors.append(float(np.degrees(Rotation.from_matrix(pred[:3,:3].T@measured[:3,:3]).magnitude())))
        tcp=measured@model.link6_tool
        orientation.append(float(np.degrees(Rotation.from_matrix(target.T@tcp[:3,:3]).magnitude())))
        margins.append(float(np.minimum(q-model.lower,model.upper-q).min()))
    last=rows[-1];q=np.array(last['q_actual'][0]);goal=model.fk(request['goal_rad']);actual=model.fk(q)
    endpoint=float(np.linalg.norm(actual[:3,3]-goal[:3,3]));force=max(r['max_contact_force_N'] for r in rows)
    result={'evidence':'MEASURED_ISOLATED_ARM_PHYSX_TRAJECTORY','complete_industrial_task_safe':False,
        'raw_knots':sum(isinstance(r['raw_knot'],int) for r in rows),
        'physics_steps':json.loads((a.replay/'manifest.json').read_text())['physics_steps'],
        'max_link6_fk_position_error_m':max(pos_errors),'max_link6_fk_rotation_error_deg':max(rot_errors),
        'endpoint_tcp_error_m':endpoint,'final_joint_error_rad':last['tracking_error_rad'],
        'max_tcp_orientation_error_deg':max(orientation),'minimum_soft_limit_margin_rad':min(margins),
        'max_recorded_contact_force_N':force,'continuous_contacts_checked':False,
        'sampled_scope_pass':max(pos_errors)<1e-4 and max(rot_errors)<.1 and endpoint<.002 and max(orientation)<3 and min(margins)>0 and force<1,
        'tolerances':{'endpoint_m':.002,'orientation_deg':3,'fk_position_m':1e-4,'fk_rotation_deg':.1},
        'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.request,a.replay/'replay.json',Path(__file__)]},
        'limitations':['no full chassis/left arm/cabinet collision scene','convex hull and welded tray proxy',
            'contacts sampled at raw knots, not every physics step','not contact grasp or complete placement']}
    (a.replay/'independent_physx_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
