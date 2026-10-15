"""Predeclared bounded REAL-model endpoint probe; never a path success label."""
import argparse
import copy
import json
import os
from controlled_paths import require_private_location
from pathlib import Path
import numpy as np
from ares_r.motion.runtime_goal_ik import RuntimeGoalIK
from ares_r.motion.independent_path_validation import validate_dense_world


def main(a):
    require_private_location(a.output)
    os.umask(0o077)
    sources=a.root/'real_aligned_sources'
    limits=json.loads((sources/'config_snapshot/site_config_20261014/jaka_mini2_motion.site.json').read_text())
    low=np.array(limits['lower_rad'])+limits['soft_limit_margin_rad'];high=np.array(limits['upper_rad'])-limits['soft_limit_margin_rad']
    rows=[]
    # Four fixed seeds per request, no extension after seeing outcomes.
    for relative in a.select:
        request=json.loads((a.root/relative/'planner_request.json').read_text())
        solver=RuntimeGoalIK(a.urdf,request['T_body_model'],request['T_link6_tcp'])
        solver.lower=low;solver.upper=high
        original_goal=solver.fk(request['goal_rad'])
        runtime=request.get('runtime_motion_goal')
        xyz=runtime['position_m'] if runtime else original_goal[:3,3]
        rotation=np.array(request['goal_candidate_metadata'][0]['rotation']) if runtime else original_goal[:3,:3]
        rng=np.random.default_rng(31)
        seeds=[request['start_rad'],request['goal_rad'],*rng.uniform(low,high,(2,6)).tolist()]
        for i,seed in enumerate(seeds):
            row={'request':relative,'seed_index':i,'rng_seed':31,'fixed_total_seeds':4,'evidence':'NUMERICAL_IK_ENDPOINT_ONLY',
                 'full_trajectory_success':None,'physical_success':None}
            try:
                result=solver.solve(xyz,rotation,seed);q=np.array(result.joints_rad)
                relocated=copy.deepcopy(request);relocated['robot_yaml_urdf']=str(a.urdf.resolve())
                endpoint=validate_dense_world([q.tolist(),q.tolist()],relocated,subdivisions=2)
                row.update(status='IK_RESIDUAL_PASS',q_rad=q.tolist(),position_error_m=result.position_error_m,
                           orientation_error_rad=result.orientation_error_rad,
                           reference_soft_margin_rad=float(np.minimum(q-low,high-q).min()),
                           modeled_endpoint=endpoint)
            except Exception as error:row.update(status='IK_OR_SOFTWARE_REJECT',error=str(error))
            rows.append(row)
    (a.output/'BOUNDED_REAL_MODEL_IK.json').write_text(json.dumps({'publication':'CONTROLLED_NOT_PUBLIC',
        'reference_limits_date':'2026-10-14 NOT original config freeze','trials':rows,
        'no_random_success_labels':True,'path_or_execution_qualified':False},indent=2))
    print(json.dumps([{'request':r['request'],'seed':r['seed_index'],'status':r['status'],
                      'gap_m':r.get('modeled_endpoint',{}).get('min_clearance_m')} for r in rows],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','urdf','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--select',action='append',required=True);main(p.parse_args())
