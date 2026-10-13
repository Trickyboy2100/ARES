"""Predeclared CPU representation sensitivity, not planner-success labels."""
import argparse,copy,json
from pathlib import Path
from ares_r.motion.independent_path_validation import validate_dense_world

def main():
    p=argparse.ArgumentParser();p.add_argument('--requests',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for path in sorted(a.requests.glob('request_[0-9]*.json')):
        request=json.loads(path.read_text());request['robot_yaml_urdf']=str(a.requests/'robot.urdf')
        for cell in (.035,.020,.015):
            for state in ('start_rad','goal_rad'):
                for scope in ('FULL_PROXY_WORLD','CABINET_ONLY_DIAGNOSTIC'):
                    r=copy.deepcopy(request);r['planning_parameters']['active_sphere_cell_m']=cell
                    if scope=='CABINET_ONLY_DIAGNOSTIC':
                        r['compiled_scene']['cuboids']={k:v for k,v in r['compiled_scene']['cuboids'].items() if k.startswith('cabinet/')}
                    result=validate_dense_world([r[state],r[state]],r,subdivisions=2)
                    rows.append({'candidate_id':r['candidate_id'],'cell_m':cell,'state':state,'scope':scope,
                        'min_clearance_m':result['min_clearance_m'],'limiting_component':result['limiting_robot_component_id'],
                        'limiting_obstacle':result['limiting_object_id'],'evidence':'CPU_REPRESENTATION_SENSITIVITY_NOT_PLANNING',
                        'training_label':None})
    a.output.write_text(json.dumps(rows,indent=2));print(json.dumps(rows))

if __name__=='__main__':main()
