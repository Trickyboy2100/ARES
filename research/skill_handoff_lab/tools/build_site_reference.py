"""Independent STATIC_RENDER site-reference layer, not a physics benchmark.

Historical BODY installation, measured September q, original robot/metal tray
geometry, wireframe scan proxies, and UNVERIFIED reported October targets.
No invented cabinet walls/free-space, no robot commands, no training.
"""
import argparse,json,itertools,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
from ares_r.perception.robot_collision import arm_link_transforms
from research.skill_handoff_lab.bench.model import PinnedRobotModel
p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True)
p.add_argument('--asset',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
def matrix(prim,t):
    xf=UsdGeom.Xformable(prim);xf.ClearXformOpOrder();xf.AddTransformOp().Set(Gf.Matrix4d(t.T.tolist()))
def sphere(stage,path,pos,color,radius=.014):
    g=UsdGeom.Sphere.Define(stage,path);g.CreateRadiusAttr(radius);g.CreateDisplayColorAttr([color]);g.AddTranslateOp().Set(Gf.Vec3d(*pos))
stage=Usd.Stage.CreateNew(str(a.audit/'SITE_REFERENCE_STATIC.usda'));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
root=UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(root.GetPrim())
source=Usd.Stage.Open(str(a.asset/'jaka_held.usda'));flat=source.Flatten();m=PinnedRobotModel(a.ares,a.ares_r)
field=json.loads((a.audit/'FIELD_REFERENCE.json').read_text());state=field['snapshot']['robot_state']
for spec in flat.rootPrims:
    if spec.name!='Robot':Sdf.CopySpec(flat,spec.path,stage.GetRootLayer(),spec.path)
for side in ['left','right']:
    name='/World/'+side.capitalize()+'Arm';Sdf.CopySpec(flat,'/Robot',stage.GetRootLayer(),name)
    matrix(stage.GetPrimAtPath(name),m.body_bases[side]);fk=arm_link_transforms(m.urdf,state[side+'_joints_rad'])
    for prim in Usd.PrimRange(stage.GetPrimAtPath(name)):
        for rel in prim.GetRelationships():
            rel.SetTargets([Sdf.Path(str(t).replace('/Robot',name,1)) if str(t).startswith('/Robot') else t for t in rel.GetTargets()])
    for i in range(7):
        body=stage.GetPrimAtPath(f'{name}/Link_{i}');matrix(body,fk['base_link' if i==0 else f'link{i}'])
        # Rendering-only template: retain geometry and CollisionAPI, disable
        # dynamics rather than pretend these are commissioned physics states.
        if body.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(body).GetRigidBodyEnabledAttr().Set(False)
    UsdGeom.Imageable(stage.GetPrimAtPath(name+'/Link_6/WeldedTray')).MakeInvisible()
    tcp=np.array(json.loads((a.audit/'FRAME_ALIGNMENT.json').read_text())['arms'][side]['T_link6_tcp_historical'])
    sphere(stage,f'/World/{side}_historical_tcp',(m.body_bases[side]@fk['link6']@tcp)[:3,3],(.1,.8,.9),.01)

# A legacy chassis visual placed by arm BASE POSITIONS ONLY. Full old WORLD
# frame is demonstrably inconsistent: this is not measured chassis alignment.
old=Usd.Stage.Open(str(a.ares/'isaac_sim/simforge/scenes/main.usd'));oldcache=UsdGeom.XformCache()
path='/World/robot/tn__robotstriped_WI'
position_fit=np.eye(4);position_fit[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();position_fit[:3,3]=[1.2,.15,0]
UsdGeom.Xform.Define(stage,'/World/LegacyChassisVisual');count=0
for prim in Usd.PrimRange(old.GetPrimAtPath(path),Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Mesh):continue
    original=UsdGeom.Mesh(prim);vertices=np.array(original.GetPointsAttr().Get())
    t=position_fit@np.array(oldcache.GetLocalToWorldTransform(prim)).T
    target=UsdGeom.Mesh.Define(stage,f'/World/LegacyChassisVisual/Piece_{count}');count+=1
    target.CreatePointsAttr((vertices@t[:3,:3].T+t[:3,3]).tolist())
    target.CreateFaceVertexCountsAttr(original.GetFaceVertexCountsAttr().Get());target.CreateFaceVertexIndicesAttr(original.GetFaceVertexIndicesAttr().Get())
    target.CreateSubdivisionSchemeAttr('none');target.CreateDisplayColorAttr([(.28,.30,.34)])

# Actual old metal mesh centered for comparison display. No field attachment
# transform is fabricated: this tray is NOT claimed held by the right gripper.
data=np.load(a.audit/'old_tray_geometry.npz');points=data['vertices_world_m'];points=points-(points.min(0)+points.max(0))/2+np.array([.38,-.65,.82])
tray=UsdGeom.Mesh.Define(stage,'/World/OldMetalTray_CAD_DisplayOnly');tray.CreatePointsAttr(points.tolist())
tray.CreateFaceVertexCountsAttr(data['face_counts'].tolist());tray.CreateFaceVertexIndicesAttr(data['face_indices'].tolist())
tray.CreateSubdivisionSchemeAttr('none');tray.CreateDisplayColorAttr([(.65,.68,.7)])
sphere(stage,'/World/ReportedOctoberTarget',field['october_target_reported_body_m'],(1.,.35,.02))
sphere(stage,'/World/ReportedOctoberPreland',field['october_preland_reported_body_m'],(.55,.1,.8))
corners=np.array(list(itertools.product([-1,1],repeat=3)));edges=[(i,j) for i in range(8) for j in range(i) if np.count_nonzero(corners[i]!=corners[j])==1]
for index,obstacle in enumerate(field['snapshot']['environment']['observation']['obstacles']):
    if obstacle['object_id'].startswith('inactive_left'):continue
    quat=obstacle['pose']['quaternion_wxyz'];r=Rotation.from_quat([*quat[1:],quat[0]]).as_matrix()
    vertices=corners*np.array(obstacle['dimensions_m'])/2@r.T+np.array(obstacle['pose']['xyz_m'])
    line=UsdGeom.BasisCurves.Define(stage,f'/World/HistoricalProxyWire_{index}')
    line.CreateTypeAttr('linear');line.CreateCurveVertexCountsAttr([2]*len(edges));line.CreatePointsAttr(np.concatenate([vertices[[i,j]] for i,j in edges]).tolist())
    line.CreateWidthsAttr([.002]);line.CreateDisplayColorAttr([(.1,.6,.35) if 'table' in obstacle['object_id'] else (.45,.4,.2)])
stage.GetRootLayer().Save()
(a.audit/'STATIC_SCENE_MANIFEST.json').write_text(json.dumps({'evidence':'STATIC_RENDER_HISTORICAL_CAD_PLUS_SCAN_PROXY_AND_UNVERIFIED_OCTOBER_TARGET',
    'physics_task_completed':False,'physics_training':False,'left_right_q_source':'September21 clear_final measured snapshot, NOT October HOLD_CURRENT',
    'chassis_placement':'BASE_POSITION_FIT_ONLY_UNVERIFIED_ROTATION_AND_MEASUREMENTS','chassis_source_meshes':count,
    'metal_tray':'actual old USD vertices recentered for comparison display, NOT field held-object pose',
    'obstacles':'September wireframe proxies; no October cavity/free-space reconstruction',
    'scene_sha256':hashlib.sha256((a.audit/'SITE_REFERENCE_STATIC.usda').read_bytes()).hexdigest()},indent=2))
