"""Concatenate actual accepted planner knots, no interpolation or relabelling."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
for name,indices in [('standoff',[1,2]),('lift',[3,4])]:
    points=[];sources=[]
    for index in indices:
        root=a.execution/str(index);validation=json.loads((root/'e0_independent_validation.json').read_text())
        if not validation['model_validated_path']:raise ValueError('unqualified source path')
        data=json.loads((root/'result.json').read_text());knots=data['trajectory_points_rad']
        if points and np.linalg.norm(np.array(points[-1])-knots[0])>1e-5:raise ValueError('chain discontinuity')
        points.extend(knots);sources.append({'segment':index,'sha256':hashlib.sha256((root/'result.json').read_bytes()).hexdigest()})
    (a.output/(name+'.json')).write_text(json.dumps({'trajectory_points_rad':points,
        'scope':'COLLATED_REAL_PLANNER_KNOTS_NOT_A_NEW_PLANNER_CALL','source_segments':sources},indent=2))
