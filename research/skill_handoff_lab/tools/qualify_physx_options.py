"""Independent model-grade checks on actual PhysX samples, never success from playback."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import yaml
from controlled_paths import require_private_location


def main(a):
    require_private_location(a.run);os.umask(0o077)
    from research.skill_handoff_lab.cash import qualified_planning as qp
    from fast_proxy_collision import independent_validator
    from ares_r.motion.runtime_goal_ik import RuntimeGoalIK
    qp.validate_dense_world=independent_validator()[0]
    data=np.load(a.run/'PHYSX_STATES.npz');rows=data['rows'];phases=data['phase']
    meta=json.loads((a.asset/'SITE_ASSET_MANIFEST.json').read_text());limits=meta['reference_limits']
    ignore=yaml.safe_load(a.yaml.read_text())['kinematics']['self_collision_ignore']
    requests=[a.prefix,a.run/'observed_state_preland_request.json'];checks=[]
    for i in np.unique(phases):
        request=json.loads(requests[int(i)].read_text());actual=rows[phases==i,1:7]
        validation=qp.qualify(actual,request,a.urdf,limits,ignore)
        validation['phase']=int(i);validation['request_sha256']=hashlib.sha256(requests[int(i)].read_bytes()).hexdigest()
        checks.append(validation)
    dt=1/120;velocity=np.diff(rows[:,1:7],axis=0)/dt;acceleration=np.diff(velocity,axis=0)/dt
    command_velocity=np.diff(rows[:,7:13],axis=0)/dt;command_acceleration=np.diff(command_velocity,axis=0)/dt
    dynamics={'measured_velocity_pass':bool(np.all(abs(velocity)<=np.array(limits['max_velocity_rad_s'])+1e-9)),
        'measured_acceleration_pass':bool(np.all(abs(acceleration)<=np.array(limits['max_acceleration_rad_s2'])+1e-9)),
        'command_velocity_pass':bool(np.all(abs(command_velocity)<=np.array(limits['max_velocity_rad_s'])+1e-9)),
        'command_acceleration_pass':bool(np.all(abs(command_acceleration)<=np.array(limits['max_acceleration_rad_s2'])+1e-9)),
        'max_actual_velocity_rad_s':float(abs(velocity).max()),'max_actual_acceleration_rad_s2':float(abs(acceleration).max()),
        'max_command_velocity_rad_s':float(abs(command_velocity).max()),'max_command_acceleration_rad_s2':float(abs(command_acceleration).max())}
    import xml.etree.ElementTree as ET
    model=ET.parse(a.urdf).getroot()
    urdf_limits=[model.find(f"joint[@name='joint{i+1}']/limit") for i in range(6)]
    hard_low=np.array([float(e.attrib['lower']) for e in urdf_limits]);hard_high=np.array([float(e.attrib['upper']) for e in urdf_limits])
    dynamics['source_urdf_hard_bounds_pass']=bool(np.all(rows[:,1:7]>=hard_low) and np.all(rows[:,1:7]<=hard_high))
    final_request=json.loads(requests[int(phases[-1])].read_text())
    solver=RuntimeGoalIK(a.urdf,final_request['T_body_model'],final_request['T_link6_tcp'])
    expected=solver.fk(final_request['goal_rad'])[:3,3]
    nominal=np.array(a.goal) if a.goal else expected
    endpoint_error=float(np.linalg.norm(rows[-1,13:16]-nominal))
    report={'evidence':'PhysX executed, proxy-model independent qualification; NOT real-robot or mesh-contact certification',
        'phases':checks,'dynamics':dynamics,'nominal_goal_body_m':nominal.tolist(),'nominal_endpoint_error_m':endpoint_error,
        'two_actual_options_executed':set(phases)=={0,1},
        'physics_model_contract_pass':bool(set(phases)=={0,1} and all(c['qualified'] for c in checks)
            and all(v for k,v in dynamics.items() if k.endswith('_pass')) and endpoint_error<=.003),
        'industrial_safety_certified':False,'physics_samples_sha256':hashlib.sha256((a.run/'PHYSX_STATES.npz').read_bytes()).hexdigest()}
    (a.run/'ACTUAL_PATH_QUALIFICATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'physics_model_contract_pass':report['physics_model_contract_pass'],**dynamics,'endpoint_error_m':endpoint_error}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['run','asset','prefix','urdf','yaml']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--goal',type=float,nargs=3);main(p.parse_args())
