"""Independent URDF/FK and all-tray-corners check of measured test terminals."""
import argparse,json,hashlib,itertools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import arm_link_transforms
p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True)
p.add_argument('--experiment',type=Path,required=True);p.add_argument('--request',type=Path,required=True);a=p.parse_args()
m=PinnedRobotModel(a.ares,a.ares_r);r=json.loads(a.request.read_text());manifest=json.loads((a.experiment/'manifest.json').read_text())
g=manifest['industrial_geometry'];h=np.array(r['attached_object_collision']['T_link6_object'])
corners=np.array(list(itertools.product([-1,1],repeat=3)))*np.array(r['tray_provenance']['observed_world_aabb_m'])/2
rows=[]
for episode in json.loads((a.experiment/'frozen_evaluation.json').read_text())['episodes']:
    q=np.array(episode['terminal_q']);pose=m.body_base@arm_link_transforms(m.urdf,q)['link6']@h
    world=corners@pose[:3,:3].T+pose[:3,3];mouth=g['initial_held_center_body_m'][0]+g['mouth_offset_m']
    fit=bool(world[:,0].min()>mouth and world[:,0].max()<mouth+g['cabinet_depth_m'] and
        abs(world[:,1]-g['initial_held_center_body_m'][1]).max()<g['opening_width_m']/2 and
        abs(world[:,2]-g['initial_held_center_body_m'][2]).max()<g['opening_height_m']/2)
    depth=float(pose[0,3]-mouth);align=float(np.linalg.norm(pose[1:3,3]-np.array(episode['goal_body_m'])[1:3]))
    tilt=float(Rotation.from_matrix(np.array(r['orientation_lock']['target_R_body_tcp']).T@m.fk(q)[:3,:3]).magnitude())
    limits=bool((q>=m.lower).all() and (q<=m.upper).all())
    valid=fit and limits and depth>g['success_depth_m'] and align<g['alignment_tolerance_m'] and tilt<g['orientation_tolerance_rad']
    rows.append({'case_id':episode['case_id'],'reported_success':episode['success'],'endpoint_geometry_valid':valid,
        'all_tray_corners_inside':fit,'soft_limits':limits,'depth_m':depth,'alignment_m':align,'tilt_rad':tilt})
result={'reported_successes':sum(x['reported_success'] for x in rows),
    'successes_rejected_by_independent_geometry':sum(x['reported_success'] and not x['endpoint_geometry_valid'] for x in rows),
    'episodes':rows,'scope':'endpoint geometry only, not continuous collision/contact or independent hold-time verification',
    'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.experiment/'independent_endpoints.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='episodes'}))
