"""Research-only mesh-complete arm asset; preserve original USD and site limits.

Uses original USD articulation frames/inertia, verified against pinned URDF.
Retains arm instance-prototype meshes and adds fixed EG2 gripper hulls.
Fixed-finger, welded-tray scope is not contact grasp or whole-chassis safety.
"""
import argparse,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import arm_link_transforms,transform_xyz_rpy

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def matrix(xform,t):
    xform.ClearXformOpOrder();xform.AddTransformOp().Set(Gf.Matrix4d(t.T.tolist()))
def mesh(stage,path,source,t=None):
    m=trimesh.load(source,force='mesh',process=False)
    prim=UsdGeom.Mesh.Define(stage,path)
    prim.CreatePointsAttr(m.vertices.tolist());prim.CreateFaceVertexCountsAttr([3]*len(m.faces))
    prim.CreateFaceVertexIndicesAttr(m.faces.flatten().tolist())
    prim.CreateSubdivisionSchemeAttr('none');prim.CreateDisplayColorAttr([(0.65,.7,.75)])
    if t is not None:matrix(UsdGeom.Xformable(prim),t)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr('convexHull')

def main():
    p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True)
    p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    model=PinnedRobotModel(a.ares,a.ares_r)
    source=a.ares/'isaac_sim/simforge/assets/jaka_minicobo/jaka_minicobo.usd'
    original=Usd.Stage.Open(str(source));flat=original.Flatten()
    stage=Usd.Stage.CreateNew(str(a.output/'jaka_held.usda'))
    # Flattened instance prototypes are root specs: preserve their real meshes.
    for spec in flat.rootPrims:
        if spec.name!='jaka_minicobo':Sdf.CopySpec(flat,spec.path,stage.GetRootLayer(),spec.path)
    Sdf.CopySpec(flat,'/jaka_minicobo',stage.GetRootLayer(),'/Robot')
    for prim in stage.Traverse():
        for rel in prim.GetRelationships():
            paths=rel.GetTargets()
            if any(str(x).startswith('/jaka_minicobo') for x in paths):
                rel.SetTargets([Sdf.Path(str(x).replace('/jaka_minicobo','/Robot',1)) for x in paths])
    stage.SetDefaultPrim(stage.GetPrimAtPath('/Robot'));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1)
    cache=UsdGeom.XformCache();urdf_frames=arm_link_transforms(model.urdf,[0]*6)
    frame_errors={};hashes={str(source):sha(source),str(model.urdf):sha(model.urdf)}
    for layer in original.GetUsedLayers():
        path=Path(layer.realPath)
        if layer.realPath and path.is_file():hashes[str(path)]=sha(path)
    for i in range(7):
        name='base_link' if i==0 else f'link{i}'
        t=np.array(cache.GetLocalToWorldTransform(original.GetPrimAtPath(f'/jaka_minicobo/Link_{i}'))).T
        frame_errors[name]=float(np.max(np.abs(t-urdf_frames[name])))
        if frame_errors[name]>1e-4:raise ValueError('USD/URDF zero-state frame mismatch')
        source_mesh=model.urdf.parent/f'jaka_minicobo_meshes/Link{i}.STL'
        hashes[str(source_mesh)]=sha(source_mesh)
    root=ET.parse(model.urdf).getroot();transforms={'link6':np.eye(4)}
    pending=[j for j in root.findall('joint') if j.find('child').attrib['link'].startswith('4C2')]
    while pending:
        ready=[j for j in pending if j.find('parent').attrib['link'] in transforms]
        if not ready:raise ValueError('broken gripper chain')
        for j in ready:
            o=j.find('origin');t=transform_xyz_rpy([float(x) for x in o.attrib.get('xyz','0 0 0').split()],
                [float(x) for x in o.attrib.get('rpy','0 0 0').split()])
            transforms[j.find('child').attrib['link']]=transforms[j.find('parent').attrib['link']]@t;pending.remove(j)
    for name,t in transforms.items():
        if name=='link6':continue
        source_mesh=model.urdf.parent/root.find(f"link[@name='{name}']/collision/geometry/mesh").attrib['filename']
        mesh(stage,f'/Robot/Link_6/FixedGripper_{name}',source_mesh,t);hashes[str(source_mesh)]=sha(source_mesh)
    r=json.loads((a.job/'request.json').read_text());v=json.loads((a.job/'e0_independent_validation.json').read_text())
    if not v.get('model_validated_path'):raise ValueError('replay requires qualified planning path')
    held=UsdGeom.Cube.Define(stage,'/Robot/Link_6/WeldedTray');held.CreateSizeAttr(1)
    t=np.array(r['attached_object_collision']['T_link6_object']);scaled=t.copy()
    scaled[:3,:3]=t[:3,:3]@np.diag(r['tray_provenance']['observed_world_aabb_m'])
    matrix(UsdGeom.Xformable(held),scaled);held.CreateDisplayColorAttr([(.9,.6,.08)])
    UsdPhysics.CollisionAPI.Apply(held.GetPrim())
    # Merge payload mass into link6; attachment is deliberately rigid, no fictitious grasp.
    mass=UsdPhysics.MassAPI(stage.GetPrimAtPath('/Robot/Link_6'))
    old=float(mass.GetMassAttr().Get() or 0);mass.GetMassAttr().Set(old+.3)
    for i in range(6):
        j=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f'/Robot/joints/joint_{i+1}'))
        j.GetLowerLimitAttr().Set(float(np.degrees(model.lower[i])));j.GetUpperLimitAttr().Set(float(np.degrees(model.upper[i])))
        drive=UsdPhysics.DriveAPI(j.GetPrim(),'angular');drive.GetStiffnessAttr().Set(1500)
        drive.GetDampingAttr().Set(80);drive.GetMaxForceAttr().Set(100)
    rootprim=stage.GetPrimAtPath('/Robot/root_joint')
    rootprim.GetAttribute('physxArticulation:enabledSelfCollisions').Set(True)
    stage.GetRootLayer().Save()
    result={'scope':'ISOLATED_JAKA_FIXED_FINGER_WELDED_TRAY_PHYSICS_SMOKE',
        'whole_chassis_safe':False,'contact_grasp_tested':False,'collision_approximation':'convexHull of every source STL; tray USD AABB',
        'payload_mass_kg':.3,'payload_inertia':'merged approximate link6 inertia, not measured payload dynamics',
        'body_bases':{k:t.tolist() for k,t in model.body_bases.items()},'tool':model.link6_tool.tolist(),
        'joint_lower':model.lower.tolist(),'joint_upper':model.upper.tolist(),
        'start_q':r['start_rad'],'goal_q':r['goal_rad'],
        'zero_frame_max_abs_error':frame_errors,'source_sha256':hashes,'scene_sha256':sha(a.output/'jaka_held.usda'),
        'planner_request_sha256':sha(a.job/'request.json'),'planner_result_sha256':sha(a.job/'result.json')}
    (a.output/'asset_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
