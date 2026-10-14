"""Independent continuous-in-time sampled convex geometry collision audit.

Extracts the actual USD collision hulls (including instance proxies), not a
new arm proxy. Checks EVERY saved 1/120s state against authored local walls.
No claim of continuous-time CCD or calibrated real hardware safety.
"""
import argparse,gzip,json,hashlib,itertools,re,time
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
import fcl
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import arm_link_transforms
p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True)
p.add_argument('--asset',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True)
p.add_argument('--self-check',action='store_true');a=p.parse_args()
model=PinnedRobotModel(a.ares,a.ares_r);stage=Usd.Stage.Open(str(a.asset/'jaka_held.usda'));cache=UsdGeom.XformCache()
metadata=json.loads((a.experiment/'manifest.json').read_text())
cases={x['case_id']:x for x in json.loads((a.experiment/'frozen_cases.json').read_text())}
asset_metadata=json.loads((a.asset/'asset_manifest.json').read_text())
request_metadata=json.loads((a.asset/'industrial_request.json').read_text())
held_transform=np.array(request_metadata['attached_object_collision']['T_link6_object'])
reference_link_rotation=np.array(request_metadata['orientation_lock']['target_R_body_tcp'])@np.array(asset_metadata['tool'])[:3,:3].T
groups={g['id']:g for g in metadata.get('task_groups',[]) or []}
pieces=[]
for prim in Usd.PrimRange(stage.GetDefaultPrim(),Usd.TraverseInstanceProxies()):
    path=str(prim.GetPath());match=re.search('/Link_(\\d+)/',path)
    if not match:continue
    if prim.IsA(UsdGeom.Mesh) and ('/collisions/' in path or prim.HasAPI(UsdPhysics.CollisionAPI)):
        points=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
    elif prim.IsA(UsdGeom.Cube) and prim.HasAPI(UsdPhysics.CollisionAPI):
        points=np.array(list(itertools.product([-1,1],repeat=3)))*float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
    else:continue
    index=int(match.group(1));link=stage.GetPrimAtPath(f'/Robot/Link_{index}')
    transform=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(link)).T)@np.array(cache.GetLocalToWorldTransform(prim)).T
    points=points@transform[:3,:3].T+transform[:3,3];hull=ConvexHull(points)
    vertices=points[hull.vertices];tri=ConvexHull(vertices).simplices.copy()
    # FCL Convex requires outward-oriented polygon winding. QHull's simplices
    # are not guaranteed to have it; otherwise EPA depths are meaningless.
    for face in tri:
        v0,v1,v2=vertices[face];normal=np.cross(v1-v0,v2-v0)
        if np.dot(normal,(v0+v1+v2)/3-vertices.mean(0))<0:face[1],face[2]=face[2],face[1]
    faces=np.column_stack([np.full(len(tri),3),tri]).astype(np.int32).flatten()
    geometry=fcl.Convex(vertices,len(tri),faces)
    pieces.append({'path':path,'link':index,'vertices':vertices,'object':fcl.CollisionObject(geometry)})
if len(pieces)<15:raise ValueError(f'Incomplete collision extraction: {len(pieces)} pieces')
specs=json.loads((a.experiment/'scene_specs.json').read_text());walls={}
for item in specs:
    wall=fcl.CollisionObject(fcl.Box(*item['size_m']),fcl.Transform(np.array(item['rotation_body']),np.array(item['center_body_m'])))
    half=np.abs(np.array(item['rotation_body']))@np.array(item['size_m'])/2
    walls.setdefault(item['env'],[]).append((item,wall,np.array(item['center_body_m'])-half,np.array(item['center_body_m'])+half))
with gzip.open(a.experiment/'process_states.json.gz','rt') as stream:states=json.load(stream)
summary={};checked=set();started=time.monotonic();samples=0
request=fcl.CollisionRequest(num_max_contacts=32,enable_contact=True)
for state in states:
    for env,(case,q) in enumerate(zip(state['case_ids'],state['q'])):
        if case<0 or (state['physics_step'],case) in checked:continue
        checked.add((state['physics_step'],case));samples+=1
        row=summary.setdefault(case,{'case_id':case,'samples':0,'collision_samples':0,'max_penetration_m':0.,'pairs':{}});row['samples']+=1
        transforms=arm_link_transforms(model.urdf,q);collided=False;bounds=[]
        row['soft_limit_violating_samples']=row.get('soft_limit_violating_samples',0)+int((np.array(q)<model.lower).any() or (np.array(q)>model.upper).any())
        for piece in pieces:
            name='base_link' if piece['link']==0 else f"link{piece['link']}"
            t=model.body_base@transforms[name]
            if 'body_pos_body' in state:
                if 'body_names' in metadata:body_index=metadata['body_names'].index(f"Link_{piece['link']}")
                else:
                    # Legacy logs have an extra Robot root rigid body before
                    # Link_0..6. Verified against FK; never assume seven bodies.
                    body_index=piece['link']+(len(state['body_pos_body'][env])==8)
                observed=np.eye(4);observed[:3,3]=state['body_pos_body'][env][body_index]
                w,x,y,z=state['body_quat_w'][env][body_index];observed[:3,:3]=Rotation.from_quat([x,y,z,w]).as_matrix()
                row['max_link_fk_position_error_m']=max(row.get('max_link_fk_position_error_m',0.),float(np.linalg.norm(t[:3,3]-observed[:3,3])))
                t=observed
            v=piece['vertices']@t[:3,:3].T+t[:3,3];lo=v.min(0);hi=v.max(0);bounds.append((lo,hi))
            if piece['link']==6 and piece['path'].endswith('/WeldedTray'):
                held_center=(t@held_transform)[:3,3]
                if case in cases and 'contract' in cases[case]:
                    contract=np.array(cases[case]['contract']);group=groups[cases[case]['task_group']]
                    relative=held_center-np.array(group.get('center_body_m',[.45475,-.48,1.45]))
                    excess=float(max(0.,*(contract[:3]-relative),*(relative-contract[3:6])))
                    row['max_workspace_excess_m']=max(row.get('max_workspace_excess_m',0.),excess)
                    desired=Rotation.from_euler('z',group.get('tool_yaw_deg',0),degrees=True).as_matrix()@reference_link_rotation
                    pose=float(Rotation.from_matrix(desired@t[:3,:3].T).magnitude())
                    row['max_pose_error_rad']=max(row.get('max_pose_error_rad',0.),pose)
                    row['pose_violating_samples']=row.get('pose_violating_samples',0)+int(pose>contract[11]+1e-5)
            piece['object'].setTransform(fcl.Transform(t[:3,:3],t[:3,3]))
            for spec,wall,wlo,whi in walls[env]:
                if (hi<wlo).any() or (whi<lo).any():continue
                result=fcl.CollisionResult();fcl.collide(piece['object'],wall,request,result)
                if not result.is_collision:continue
                collided=True;depth=max((x.penetration_depth for x in result.contacts),default=0.)
                row['max_penetration_m']=max(row['max_penetration_m'],float(depth))
                key=piece['path']+'::'+spec['wall'];row['pairs'][key]=row['pairs'].get(key,0)+1
        row['collision_samples']+=int(collided)
        if a.self_check:
            self_collided=False
            for i,left in enumerate(pieces):
                for j in range(i):
                    right=pieces[j]
                    # Same rigid body and joints' directly adjacent link pairs
                    # are excluded, matching the physical articulation convention.
                    if abs(left['link']-right['link'])<=1:continue
                    if (bounds[i][1]<bounds[j][0]).any() or (bounds[j][1]<bounds[i][0]).any():continue
                    result=fcl.CollisionResult();fcl.collide(left['object'],right['object'],request,result)
                    if result.is_collision:
                        self_collided=True;depth=max((x.penetration_depth for x in result.contacts),default=0.)
                        row['max_self_penetration_m']=max(row.get('max_self_penetration_m',0.),float(depth))
            row['self_collision_samples']=row.get('self_collision_samples',0)+int(self_collided)
result={'geometry':'actual USD convex hulls including fixed gripper and welded tray','collision_pieces':len(pieces),
    'pose_source':'OBSERVED_PHYSX_BODY_POSES' if any('body_pos_body' in s for s in states) else
        ('URDF_FK_FROM_RECORDED_PHYSX_Q' if metadata.get('real_physics') else 'OFFLINE_FK_RAW_PLANNER_POINTS'),
    'sample_dt_s':1/120 if any('body_pos_body' in s for s in states) else None,
    'dense_raw_planner_path_sha256':metadata.get('dense_raw_planner_path_sha256'),
    'checked_states':samples,'wall_source':'scene_specs.json','cases':list(summary.values()),
    'wall_s':time.monotonic()-started,'penetration_warning_threshold_m':.001,'nonadjacent_right_arm_self_checked':a.self_check,
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'asset_sha256':hashlib.sha256((a.asset/'jaka_held.usda').read_bytes()).hexdigest(),
    'limitations':['sampled process, not swept continuous collision','USD convex hull approximation, not original concave CAD',
        'no calibrated contact-force/guide-contact certification','welded attachment and fixed fingers only']}
(a.experiment/'independent_process_fcl.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='cases'}))
