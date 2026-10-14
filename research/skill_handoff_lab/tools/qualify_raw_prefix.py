"""Independent offline qualification input for real cuRobo raw knots.

This is NOT PhysX data, training data, new planning or a success label. Dense
joint sampling is used only to audit the saved planner path geometry.
"""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--planner',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
meta=json.loads((a.scene/'manifest.json').read_text());meta['real_physics']=False;meta['input_evidence']='OFFLINE_FK_DENSE_VALIDATION_OF_REAL_CUROBO_RAW_KNOTS'
meta['dense_raw_planner_path_sha256']=hashlib.sha256(a.planner.read_bytes()).hexdigest()
raw=json.loads(a.planner.read_text());q=np.array(raw['trajectory_points_rad']);points=[]
for left,right in zip(q[:-1],q[1:]):
    count=max(2,int(np.ceil(np.max(np.abs(right-left))/.002)))
    points.extend(left+(right-left)*u for u in np.arange(count)/count)
points.append(q[-1]);states=[{'physics_step':i,'case_ids':[0],'q':[x.tolist()]} for i,x in enumerate(points)]
(a.output/'manifest.json').write_text(json.dumps(meta,indent=2))
(a.output/'frozen_cases.json').write_text(json.dumps([json.loads((a.scene/'frozen_cases.json').read_text())[0]],indent=2))
(a.output/'scene_specs.json').write_text(json.dumps([x for x in json.loads((a.scene/'scene_specs.json').read_text()) if x['env']==0],indent=2))
with gzip.open(a.output/'process_states.json.gz','wt') as stream:json.dump(states,stream)
