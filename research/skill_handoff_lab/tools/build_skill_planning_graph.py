"""Bounded two-stage request graph; all results must come from a real planner."""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ares_r.motion.runtime_goal_ik import RuntimeGoalIK
from controlled_paths import require_private_location

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def main(a):
    require_private_location(a.directory);os.umask(0o077)
    family=json.loads((a.directory/'TASK_FAMILY.json').read_text());source=family['original_request'];prefix=family['prefix_request']
    solver=RuntimeGoalIK(a.urdf,source['T_body_model'],source['T_link6_tcp'])
    solver.lower=np.array(family['limits']['lower_rad'])+family['limits']['soft_limit_margin_rad']+.001
    solver.upper=np.array(family['limits']['upper_rad'])-family['limits']['soft_limit_margin_rad']-.001
    prep=[np.array(prefix['goal_rad'])]
    # A second fixed Prepare point: real Transit TCP lifted 3cm, same yaw.
    t=solver.fk(prep[0]);lift=solver.solve(t[:3,3]+[0,0,.03],t[:3,:3],prep[0]);prep.append(np.array(lift.joints_rad))
    geometry=json.loads(json.dumps(source['compiled_scene']))
    base=np.array(source['T_body_model']);worldshift_rotation=base[:3,:3].T
    jobs=[];tasks=[]
    requests=a.directory/'requests';requests.mkdir(mode=0o700,exist_ok=False)
    for task in family['tasks']:
        if not task['ik_candidates']:continue
        scene=copy.deepcopy(geometry);shift=np.array(task['station_shift_body_m'])
        for name,box in scene['cuboids'].items():
            if name.startswith('observed_') or name.startswith('known_support'):
                box['pose'][:3]=(np.array(box['pose'][:3])+worldshift_rotation@shift).tolist()
        sid='RESEARCH_STATION_'+digest(task)[:16];scene['scene_snapshot_id']=sid
        scene['planning_context_digest']=digest({'source':source['scene_digest'],'shift':shift.tolist()})
        scene.pop('digest',None);scene['digest']=digest(scene)
        goal=np.array(task['ik_candidates'][0]);nodes={'CENTER':prefix['start_rad'],'TRANSIT':prep[0].tolist(),'LIFT':prep[1].tolist()}
        item=dict(task,nodes=nodes,goal_q=goal.tolist(),scene_digest=scene['digest'],edges=[])
        for node,start in nodes.items():
            targets=[('PRELAND',goal)] if node!='CENTER' else [('PRELAND',goal),('TRANSIT',prep[0]),('LIFT',prep[1])]
            for destination,qgoal in targets:
                for budget,attempts in [('FAST',1),('FALLBACK',3)]:
                    request=copy.deepcopy(source);request.update(compiled_scene=scene,scene_snapshot_id=sid,scene_digest=scene['digest'],
                        start_rad=start,goal_rad=qgoal.tolist(),goal_candidates_rad=[qgoal.tolist()],runtime_motion_goal=None,
                        robot_yaml=a.remote_directory+'/robot.yml',robot_yaml_urdf=a.remote_urdf)
                    # Frozen holding geometry/TCP preserved. Only target yaw
                    # changes with the actual requested option endpoint.
                    rotation=solver.fk(qgoal)[:3,:3]
                    yaw=float(np.arctan2(rotation[1,2],rotation[0,2]))
                    request['orientation_lock']=dict(source['orientation_lock'],target_R_body_tcp=rotation.tolist(),final_yaw_target_rad=yaw)
                    if destination=='PRELAND':request['orientation_lock']=copy.deepcopy(source['orientation_lock'])
                    request['planning_parameters']['max_attempts']=attempts
                    # Same frozen .02m active sphere grid and all hard limits.
                    key=f'{task["task_id"]}__{node}__{destination}__{budget}'
                    name=key+'.json';(requests/name).write_text(json.dumps(request,indent=2))
                    edge={'edge_id':key,'from':node,'to':destination,'profile':budget,'attempt_cap':attempts,
                          'request':str(Path('requests')/name),'request_sha256':hashlib.sha256((requests/name).read_bytes()).hexdigest()}
                    item['edges'].append(edge);jobs.append((budget,a.remote_directory+'/requests/'+name))
        tasks.append(item)
    for profile in ['FAST','FALLBACK']:
        (a.directory/f'batch_{profile}.json').write_text(json.dumps([p for b,p in jobs if b==profile],indent=2))
    (a.directory/'PLANNING_GRAPH.json').write_text(json.dumps({'tasks':tasks,'holding_object':'original frozen proxy, not CAD',
        'source_scene':source['scene_snapshot_id'],'family_level':'FIELD_DERIVED_PARAMETRIC',
        'cache_not_PhysX_online':True,'planning_call_cap':4,'edges_requested':len(jobs)},indent=2))
    print('Frozen actual planning requests:',len(jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--remote-directory',required=True);p.add_argument('--remote-urdf',required=True);main(p.parse_args())
