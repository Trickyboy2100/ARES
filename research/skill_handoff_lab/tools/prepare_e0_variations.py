"""Six frozen goal groups x two roll branches. No outcome-based seed selection."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import yaml
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from research.skill_handoff_lab.tools.prepare_oracle_batch import digest
from ares_r.motion.independent_path_validation import validate_dense_world

def main():
    p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True)
    p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--templates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-output',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);m=PinnedRobotModel(a.ares,a.ares_r)
    shutil.copyfile(a.templates/'robot.urdf',a.output/'robot.urdf')
    cfg=yaml.safe_load((a.templates/'robot.yml').read_text());cfg['kinematics']['urdf_path']=a.remote_output+'/robot.urdf'
    cfg['kinematics']['asset_root_path']=a.remote_output;(a.output/'robot.yml').write_text(yaml.safe_dump(cfg))
    design={'goals_body_m':[[.46,-.50,1.45],[.48,-.50,1.45],[.50,-.50,1.45],
        [.46,-.46,1.45],[.48,-.46,1.45],[.50,-.46,1.45]],
        'splits':['train','validation','test','train','train','test'],
        'seed':6200,'ik_attempts':1,'branches':[0,1],'gpu_max_requests':12,
        'scope':'E0 wide held transport cost variation, not industrial aperture benchmark',
        'held_model':'historical lower chassis and USD cuboid, physical safety unqualified'}
    (a.output/'frozen_design.json').write_text(json.dumps(design,indent=2))
    batch=[];preflight=[]
    for group,(goal,split) in enumerate(zip(design['goals_body_m'],design['splits'])):
      for roll,index in [(0,0),(1,2)]:
        r=json.loads((a.templates/f'request_{index}.json').read_text());start=r['start_rad']
        solutions=m.solve(goal,0.,roll,[start],6200,attempts=1)
        row={'group':group,'split':split,'roll':roll,'target_body_m':goal,'ik_found':bool(solutions)}
        if not solutions:preflight.append(row);continue
        r.update(start_rad=start,goal_rad=solutions[0][1].tolist(),robot_yaml=a.remote_output+'/robot.yml',
            robot_yaml_urdf=a.remote_output+'/robot.urdf',scene_group=f'E0-goal-{group}',split=split,
            candidate_id=f'E0-goal-{group}-roll-{roll}',scene_snapshot_id=f'E0-goal-{group}')
        r['candidate_features'].update(goal_body_m=goal,roll_branch=roll)
        r['scene_digest']=digest({'parent':r['scene_digest'],'goal':goal,'group':group})
        r['compiled_scene'].update(digest=r['scene_digest'],scene_snapshot_id=r['scene_snapshot_id'])
        check=dict(r,robot_yaml_urdf=str(a.output/'robot.urdf'))
        v=validate_dense_world([r['goal_rad'],r['goal_rad']],check,subdivisions=2)
        row['world']=v;preflight.append(row)
        # Keep every IK-found request, even when the endpoint check fails.
        name=f'request_{len(batch)}.json';(a.output/name).write_text(json.dumps(r,indent=2));batch.append(a.remote_output+'/'+name)
    (a.output/'batch.json').write_text(json.dumps(batch,indent=2));(a.output/'preflight.json').write_text(json.dumps(preflight,indent=2))
    print(json.dumps({'requests':len(batch),'groups':6,'endpoint_free':sum(x.get('world',{}).get('collision_free',False) for x in preflight)}))
if __name__=='__main__':main()
