"""Pre-frozen same-start Prepare mechanism A probe, planning-level only."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
from research.skill_handoff_lab.bench.model import PinnedRobotModel

def main():
    p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True)
    p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    model=PinnedRobotModel(a.ares,a.ares_r);base=json.loads((a.job/'request.json').read_text())
    q0=np.array(base['start_rad']);qg=np.array(base['goal_rad'])
    options=[('Nominal',None),('Prepare_standoff',[.44,-.50,1.45]),('Prepare_lift',[.42,-.48,1.47])]
    requests=[];design=[]
    for name,xyz in options:
        if xyz is None:segments=[(q0,qg)]
        else:
            answers=model.solve(xyz,0,0,[q0],seed_value=31,attempts=1)
            if not answers:
                design.append({'option':name,'status':'UNKNOWN_IK','planning_calls':0});continue
            qp=answers[0][1];segments=[(q0,qp),(qp,qg)]
        row={'option':name,'waypoint_body_m':xyz,'same_start':q0.tolist(),'same_goal':qg.tolist(),'segments':[]}
        for index,(start,goal) in enumerate(segments):
            r=copy.deepcopy(base);r.update(candidate_id=f'R1-A-{name}-{index}',start_rad=start.tolist(),goal_rad=goal.tolist(),
                experiment_role='SAME_START_OPTION_PROBE_PLANNING_ONLY')
            r['candidate_features']={'option':name,'segment':index,'same_initial_state':True}
            filename=f'request_{len(requests)}.json';(a.output/filename).write_text(json.dumps(r,indent=2))
            row['segments'].append(filename);requests.append(filename)
        design.append(row)
    (a.output/'frozen_design.json').write_text(json.dumps({'seed':31,'scope':'same-start planning-only; no physical labels',
        'scene_revision':base['scene_digest'],'options':design,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    (a.output/'batch.json').write_text(json.dumps(requests));print(json.dumps(design))
if __name__=='__main__':main()
