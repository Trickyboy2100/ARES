"""Independent offline acceptance: terminal reach != full execution validity."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from controlled_paths import require_private_location
from ares_r.perception.robot_collision import arm_link_transforms


def main(a):
    require_private_location(a.physics)
    request=json.loads(a.request.read_text());asset=json.loads((a.asset/'SITE_ASSET_MANIFEST.json').read_text())
    data=np.load(a.physics/'PHYSX_STATES.npz')['rows'];raw=json.loads((a.physics/'PHYSX_REPLAY_RESULT.json').read_text())
    limits=asset['reference_limits'];dt=raw['physics_dt_s']
    velocity=np.max(np.abs(np.diff(data[:,1:7],axis=0)),axis=0)/dt
    acceleration=np.max(np.abs(np.diff(data[:,1:7],n=2,axis=0)),axis=0)/(dt*dt)
    command_accel=np.max(np.abs(np.diff(data[:,7:13],n=2,axis=0)),axis=0)/(dt*dt)
    base=np.array(request['T_body_model']);tool=np.array(request['T_link6_tcp'])
    predicted=np.array([base@arm_link_transforms(a.urdf,q)['link6']@tool for q in data[:,1:7]])
    tcp_difference=float(np.linalg.norm(predicted[:,:3,3]-data[:,13:16],axis=1).max())
    yaw=float(np.arctan2(predicted[-1,1,2],predicted[-1,0,2]));target=request['orientation_lock']['final_yaw_target_rad']
    yaw_error=float(abs(np.degrees(np.arctan2(np.sin(yaw-target),np.cos(yaw-target)))))
    tolerance=request['orientation_lock']['max_error_deg']
    summary={'publication':'CONTROLLED_NOT_PUBLIC','source_request_sha256':hashlib.sha256(a.request.read_bytes()).hexdigest(),
        'states_sha256':hashlib.sha256((a.physics/'PHYSX_STATES.npz').read_bytes()).hexdigest(),
        'pose_reach':bool(raw['final_tcp_error_m']<.003),'pose_tolerance_basis':'3mm explicit research check, not release/contact success',
        'level_constraint_deg':tolerance,'level_pass':bool(data[:,16].max()<=tolerance),
        'final_yaw_error_deg':yaw_error,'yaw_pass':bool(yaw_error<=tolerance),
        'velocity_pass_reference':bool(np.all(velocity<=np.array(limits['max_velocity_rad_s'])+1e-4)),
        'acceleration_pass_reference':bool(np.all(acceleration<=np.array(limits['max_acceleration_rad_s2'])+1e-3)),
        'measured_max_velocity_per_joint_rad_s':velocity.tolist(),'measured_max_acceleration_per_joint_rad_s2':acceleration.tolist(),
        'command_interpolant_max_acceleration_per_joint_rad_s2':command_accel.tolist(),
        'acceleration_estimate':'finite differences at 120Hz; also includes initial dynamic settling, not filtered',
        'reference_config_date':'2026-10-14; not byte-for-byte original October10 limits',
        'soft_limits_pass_reference':bool(data[:,18].min()>=0),
        'FK_vs_actual_Articulation_TCP_max_error_m':tcp_difference,
        'process_collision_penetration_certification':'NOT_TESTED; net-force diagnostic is insufficient',
        'full_execution_conditions_pass':False,'real_robot_execution':False,
        'interpretation':'transport reach/pose demonstrated; acceleration or missing penetration/self-collision certification prevents full acceptance'}
    (a.physics/'INDEPENDENT_PHYSX_ACCEPTANCE.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['request','asset','physics','urdf']:p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
