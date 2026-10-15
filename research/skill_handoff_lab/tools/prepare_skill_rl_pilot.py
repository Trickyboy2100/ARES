"""Freeze bounded original-goal IK search, then declared station-derived family."""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ares_r.motion.runtime_goal_ik import RuntimeGoalIK
from controlled_paths import require_private_location

def main(a):
    require_private_location(a.output);os.umask(0o077);a.output.mkdir(mode=0o700,exist_ok=False)
    request=json.loads((a.root/'transit10.10/preland_plan/planner_request.json').read_text())
    prefix=json.loads((a.root/'transit10.10/transit_plan/planner_request.json').read_text())
    limits=json.loads((a.root/'real_aligned_sources/config_snapshot/site_config_20261014/jaka_mini2_motion.site.json').read_text())
    low=np.array(limits['lower_rad'])+limits['soft_limit_margin_rad']+.001
    high=np.array(limits['upper_rad'])-limits['soft_limit_margin_rad']-.001
    solver=RuntimeGoalIK(a.urdf,request['T_body_model'],request['T_link6_tcp'])
    target=solver.fk(request['goal_rad']);rotation=np.array(request['orientation_lock']['target_R_body_tcp'])
    goal_source={'type':'SPLIT_REQUEST_GOAL_FK_NOT_AUTOMATICALLY_70MM_PRELAND',
                 'request_sha256':hashlib.sha256((a.root/'transit10.10/preland_plan/planner_request.json').read_bytes()).hexdigest()}
    if a.goal_request:
        goal_request=json.loads(a.goal_request.read_text());runtime=goal_request['runtime_motion_goal']
        if runtime['frame']!='BODY':raise ValueError('explicit BODY goal required')
        lock=goal_request['orientation_lock']
        target[:3,3]=runtime['position_m']
        rotation=np.array(lock.get('target_R_body_tcp',lock.get('candidate_target_rotations',[None])[0]))
        request['orientation_lock']=dict(lock,target_R_body_tcp=rotation.tolist())
        goal_source={'type':'EXPLICIT_RUNTIME_BODY_GOAL_BOUND_TO_HELD_SPLIT_WORLD_FOR_RESEARCH',
                     'request_sha256':hashlib.sha256(a.goal_request.read_bytes()).hexdigest(),'runtime_goal':runtime}
    rng=np.random.default_rng(410);trials=[]
    def solve(xyz,seeds,label):
        values=[]
        for i,seed in enumerate(seeds):
            seed=np.clip(seed,low+1e-6,high-1e-6)
            def residual(q):
                t=solver.fk(q)
                return np.r_[t[:3,3]-xyz,.18*Rotation.from_matrix(rotation.T@t[:3,:3]).as_rotvec(),1e-5*(q-seed)]
            fit=least_squares(residual,seed,bounds=(low,high),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            t=solver.fk(fit.x);pe=float(np.linalg.norm(t[:3,3]-xyz));oe=float(np.degrees(Rotation.from_matrix(rotation.T@t[:3,:3]).magnitude()))
            passed=pe<=.003 and oe<=request['orientation_lock']['max_error_deg']
            trials.append({'task':label,'seed_index':i,'q':fit.x.tolist(),'position_error_m':pe,'orientation_error_deg':oe,'endpoint_contract_pass':passed})
            if passed and all(np.linalg.norm(fit.x-x)>0.1 for x in values):values.append(fit.x)
        return sorted(values,key=lambda q:float(np.linalg.norm(q-prefix['start_rad'])))
    original=solve(target[:3,3],[request['start_rad'],request['goal_rad'],*rng.uniform(low,high,(30,6))],'ORIGINAL_FIELD_GOAL')
    tasks=[]
    # A predeclared 6-case pilot: same geometry translated relative to BODY to
    # represent a 4/7/10 cm different station stance, NOT six new field scans.
    for x in [.04,.07,.10]:
        for y in [-.025,.025]:
            shift=np.array([x,y,0]);name=f'station_x{x:.3f}_y{y:.3f}'
            solutions=solve(target[:3,3]+shift,[request['goal_rad'],prefix['goal_rad'],*rng.uniform(low,high,(6,6))],name)
            tasks.append({'task_id':name,'split':'test' if x==.07 else 'train','station_shift_body_m':shift.tolist(),
                          'goal_body_m':(target[:3,3]+shift).tolist(),'ik_candidates':[q.tolist() for q in solutions],
                          'source':'FIELD_DERIVED_PARAMETRIC_NOT_NEW_SCAN'})
    (a.output/'IK_SEARCH.json').write_text(json.dumps({'seed':410,'trials':trials,'fixed_original_attempts':32,
        'fixed_family_attempts':8,'original_candidates':[q.tolist() for q in original],'not_full_path_labels':True},indent=2))
    (a.output/'TASK_FAMILY.json').write_text(json.dumps({'tasks':tasks,'limits':limits,'original_request':request,
        'prefix_request':prefix,'goal_source':goal_source},indent=2))
    print(json.dumps({'original_legal_IK':len(original),'family':[(t['task_id'],len(t['ik_candidates'])) for t in tasks]}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','output','urdf']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--goal-request',type=Path)
    main(p.parse_args())
