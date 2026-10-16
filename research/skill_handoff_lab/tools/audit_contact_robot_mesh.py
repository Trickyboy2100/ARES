"""Actual USD robot-hull/world diagnostic, endpoints + declared 10Hz cases.
Not a swept/full-frequency safety certificate. Reuses the existing hull audit
mechanism, including actual fixed-finger meshes, without removing collisions.
"""
import argparse,json,re,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
import fcl
from ares_r.perception.robot_collision import arm_link_transforms
p=argparse.ArgumentParser()
for k in ['run','asset','urdf']:p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();meta=json.loads((a.asset/'asset_manifest.json').read_text());m=json.loads((a.run/'manifest.json').read_text())
cases=json.loads((a.run/'frozen_evaluation.json').read_text());stage=Usd.Stage.Open(str(a.asset/'jaka_paddle.usda'));cache=UsdGeom.XformCache();pieces=[]
for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
    path=str(prim.GetPath());match=re.search(r'/Link_(\d+)/',path)
    if not match or not prim.HasAPI(UsdPhysics.CollisionAPI):continue
    if prim.IsA(UsdGeom.Mesh):points=np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
    elif prim.IsA(UsdGeom.Cube):points=np.array(list(itertools.product([-1,1],repeat=3)))*float(UsdGeom.Cube(prim).GetSizeAttr().Get())/2
    else:continue
    link=int(match.group(1));local=np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f'/Robot/Link_{link}'))).T)@np.array(cache.GetLocalToWorldTransform(prim)).T
    points=points@local[:3,:3].T+local[:3,3];vertices=points[ConvexHull(points).vertices];tri=ConvexHull(vertices).simplices.copy()
    for face in tri:
        v0,v1,v2=vertices[face]
        if np.dot(np.cross(v1-v0,v2-v0),(v0+v1+v2)/3-vertices.mean(0))<0:face[1],face[2]=face[2],face[1]
    faces=np.column_stack([np.full(len(tri),3),tri]).astype(np.int32).flatten()
    pieces.append((path,link,vertices,fcl.CollisionObject(fcl.Convex(vertices,len(tri),faces))))
if len(pieces)<15:raise ValueError(f'incomplete robot collision extraction: {len(pieces)}')
center=np.array(meta['paddle_center_body_m']);base=np.array(meta['body_bases']['right']);worlds=[]
for i,spec in enumerate(m['scene_specs']):
    walls=[('table',fcl.CollisionObject(fcl.Box(.26,.30,.03),fcl.Transform(np.eye(3),center+[.145,0,-.04])),center+[.015,-.15,-.055],center+[.275,.15,-.025])]
    if m['families'][i]=='slot':
        rot=Rotation.from_euler('z',spec.get('direction',0)).as_matrix();length=spec.get('rail_length',.05)
        for side in [-1,1]:
            pos=center+rot@np.array([spec.get('entry_x',spec['goal_x']-.01)+length/2,spec['slot_y']+side*(spec['gap']/2+.0075),-.005])
            size=np.array([length,.015,.04]);half=abs(rot)@size/2
            walls.append((f'rail{side}',fcl.CollisionObject(fcl.Box(*size),fcl.Transform(rot,pos)),pos-half,pos+half))
    worlds.append(walls)
requests=fcl.CollisionRequest(num_max_contacts=8,enable_contact=True);out={}
def check(case,env,q,scope):
    r=out.setdefault(case,{'case_id':case,'checks':0,'pairs':{},'max_penetration_m':0.,'scopes':[]});r['checks']+=1
    if scope not in r['scopes']:r['scopes'].append(scope)
    transforms=arm_link_transforms(a.urdf,q)
    for path,link,vertices,obj in pieces:
        t=base@transforms['base_link' if link==0 else f'link{link}'];points=vertices@t[:3,:3].T+t[:3,3];lo=points.min(0);hi=points.max(0)
        obj.setTransform(fcl.Transform(t[:3,:3],t[:3,3]))
        for name,wall,wlo,whi in worlds[env]:
            if (hi<wlo).any() or (whi<lo).any():continue
            result=fcl.CollisionResult();fcl.collide(obj,wall,requests,result)
            if result.is_collision:
                key=path+'::'+name;r['pairs'][key]=r['pairs'].get(key,0)+1
                r['max_penetration_m']=max(r['max_penetration_m'],max((float(c.penetration_depth) for c in result.contacts),default=0))
for c in cases:check(c['case_id'],c['env'],c['post_step_state']['q'],'EXACT_TERMINAL')
trace=np.load(a.run/'actual_physics_trace.npz')['states']
for step in trace[::12]:
    for i,row in enumerate(step):
        if int(row[20]) in [1,3]:check(int(row[20]),i,row[:6],'CASE1_AND_CASE3_PROCESS_10HZ_DIAGNOSTIC')
report={'geometry':'actual USD CollisionAPI convex hulls, fixed gripper + research paddle','pieces':len(pieces),'cases':list(out.values()),
    'limitations':['endpoints for all cases; only predeclared example cases1/3 at10Hz','not full120Hz robot/self-collision or CCD safety validation'],
    'asset_sha256':meta['asset_sha256'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.run/'ROBOT_MESH_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'pieces':len(pieces),'cases_with_world_intersection':sum(bool(x['pairs']) for x in out.values()),'max_depth':max(x['max_penetration_m'] for x in out.values())}))
