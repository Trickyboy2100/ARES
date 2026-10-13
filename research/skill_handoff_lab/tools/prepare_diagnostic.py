"""One fixed 1-degree move from historical clear goal; not a handoff sample."""
import argparse,hashlib,json,math
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-output',required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);r=json.loads(a.request.read_text())
    old_digest=r['scene_digest'];r['start_rad']=r['goal_rad'][:]
    r['goal_rad']=r['start_rad'][:];r['goal_rad'][0]+=math.radians(1.)
    r['experiment_role']='FIXED_ONE_DEGREE_DIAGNOSTIC_NOT_HANDOFF_NOT_TRAINING'
    r['diagnostic_parent_scene_digest']=old_digest
    binding=hashlib.sha256(json.dumps({'parent':old_digest,'start':r['start_rad'],
        'goal':r['goal_rad'],'role':r['experiment_role']},sort_keys=True).encode()).hexdigest()
    r['scene_digest']=binding;r['compiled_scene']['digest']=binding
    r['compiled_scene']['scene_snapshot_id']='DIAGNOSTIC_1DEG_20261013';r['scene_snapshot_id']='DIAGNOSTIC_1DEG_20261013'
    r['compiled_scene']['planning_context_digest']=binding
    r['benchmark_runs']=3
    (a.output/'request.json').write_text(json.dumps(r,indent=2))
    (a.output/'batch.json').write_text(json.dumps([a.remote_output+'/request.json']))

if __name__=='__main__':main()
