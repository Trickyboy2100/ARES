"""One real-epoch Site-Reference: preserved mesh robot + declared field proxies.

Not a tray CAD reconstruction or contact grasp. Only a qualified transport
segment may be replayed. Private generated USD and states must never enter Git.
"""
import argparse
import hashlib
import json
import os
from controlled_paths import require_private_location
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
from ares_r.perception.robot_collision import arm_link_transforms, _world_arm_transform
from ares_r.motion.native_execution_package import package_native_preview


def read(p): return json.loads(p.read_text())
def matrix(prim, t):
    x = UsdGeom.Xformable(prim); x.ClearXformOpOrder(); x.AddTransformOp().Set(Gf.Matrix4d(t.T.tolist()))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(a):
    require_private_location(a.output)
    os.umask(0o077)
    a.output.mkdir(mode=0o700, exist_ok=False)
    request = read(a.request / 'planner_request.json'); result = read(a.request / 'planning.json')
    if a.qualification:
        assert read(a.qualification)['qualified'], 'independent full proxy contract required'
    else:
        audit = read(a.analysis / 'FIELD_BINDINGS_AND_REPLAY.json')
        replay = next(r for r in audit['independent_replays'] if r['request_sha256'] == sha(a.request/'planner_request.json'))
        assert replay['world']['collision_free'] and replay['orientation']['passed']
        assert replay['reference_native']['status'] == 'PASS_REFERENCE_OFFLINE_PACKAGING'
    assert request['attached_object_collision'] is not None, 'this replay specifically tests attached-object proxy transport'
    source = a.root / 'real_aligned_sources'; sid = request['scene_snapshot_id']
    scene = source / 'scenes' / (a.source_scene_id or sid); snapshot = read(scene / 'scene/snapshot.json')
    cfg = source / 'config_snapshot/site_config_20261014'
    limits = read(cfg/'jaka_mini2_motion.site.json')
    content, native = package_native_preview(result['trajectory_points_rad'], request['planning_parameters']['interpolation_dt'],
        limits, tool_id=1, controller_tool_pose_mm_rad=request['controller_tool_pose_mm_rad'], captured_at_unix=0)
    native_q = np.array([[float(v) for v in row.split()] for row in content.splitlines()[4:]])
    np.savez_compressed(a.output/'qualified_transport.npz', joints_rad=native_q,
                        lower_rad=np.array(limits['lower_rad'])+limits['soft_limit_margin_rad'],
                        upper_rad=np.array(limits['upper_rad'])-limits['soft_limit_margin_rad'])
    parent = Usd.Stage.Open(str(a.asset/'jaka_held.usda')); flat = parent.Flatten()
    robot = Usd.Stage.CreateNew(str(a.output/'right_robot.usda')); robot.GetRootLayer().TransferContent(flat)
    # The inherited pilot asset baked a historical 0.15-rad SOFT margin into
    # PhysX HARD stops. Restore the source model's declared hardware bounds;
    # execution is separately rejected at the frozen reference soft bounds.
    # This changes no robot/production limit or collision shape.
    limit_corrections=[]
    for i in range(6):
        joint=UsdPhysics.RevoluteJoint(robot.GetPrimAtPath(f'/Robot/joints/joint_{i+1}'))
        before=[joint.GetLowerLimitAttr().Get(),joint.GetUpperLimitAttr().Get()]
        after=[float(np.degrees(limits['lower_rad'][i])),float(np.degrees(limits['upper_rad'][i]))]
        joint.GetLowerLimitAttr().Set(after[0]);joint.GetUpperLimitAttr().Set(after[1])
        limit_corrections.append({'joint':i+1,'inherited_physics_stop_deg':before,'source_hardware_stop_deg':after})
    robot.RemovePrim('/Robot/Link_6/WeldedTray')  # Replace synthetic E0 OBJECT only, not robot collisions.
    attached = request['attached_object_collision']; box = attached['link6_aabb']
    cube = UsdGeom.Cube.Define(robot, '/Robot/Link_6/FieldAttachedProxy');cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*box['center_m']));cube.AddScaleOp().Set(Gf.Vec3f(*(2*np.array(box['half_extents_m']))))
    cube.CreateDisplayColorAttr([(.65,.26,.55)]);UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    robot.GetRootLayer().Save()
    site = Usd.Stage.CreateNew(str(a.output/'site.usda'));root = UsdGeom.Xform.Define(site, '/Site');site.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(site,1.);UsdGeom.SetStageUpAxis(site,'Z')
    # Left appearance is at recorded HOLD state; collision is represented by
    # the exact inactive-arm cuboids already present in the compiled world.
    for spec in flat.rootPrims:
        if spec.name != 'Robot': Sdf.CopySpec(flat,spec.path,site.GetRootLayer(),spec.path)
    Sdf.CopySpec(flat,'/Robot',site.GetRootLayer(),'/Site/LeftArm')
    left = site.GetPrimAtPath('/Site/LeftArm')
    left.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    left_audit = read(scene/'left_fk_audit.json')
    left_base = _world_arm_transform(read(cfg/'robot_world.json'),'left') @ np.array(left_audit['T_controller_model'])
    matrix(left,left_base)
    for prim in Usd.PrimRange(left):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI): UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI): UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if prim.IsA(UsdPhysics.Joint): UsdPhysics.Joint(prim).GetJointEnabledAttr().Set(False)
        for rel in prim.GetRelationships():
            rel.SetTargets([Sdf.Path(str(t).replace('/Robot','/Site/LeftArm',1)) if str(t).startswith('/Robot') else t for t in rel.GetTargets()])
    fk = arm_link_transforms(a.urdf,snapshot['robot_state']['left_joints_rad'])
    for i in range(7):matrix(site.GetPrimAtPath(f'/Site/LeftArm/Link_{i}'),fk['base_link' if i==0 else f'link{i}'])
    site.RemovePrim('/Site/LeftArm/Link_6/WeldedTray')
    body = np.array(request['T_body_model'])
    for i,(name,box) in enumerate(request['compiled_scene']['cuboids'].items()):
        pose = box['pose']; transform=np.eye(4)
        transform[:3,:3]=Rotation.from_quat([*pose[4:7],pose[3]]).as_matrix();transform[:3,3]=pose[:3]
        transform=body@transform
        cube=UsdGeom.Cube.Define(site,f'/Site/PlannerWorld/Box_{i}');cube.CreateSizeAttr(1.)
        matrix(cube.GetPrim(),transform);cube.AddScaleOp().Set(Gf.Vec3f(*box['dims']))
        cube.CreateDisplayColorAttr([(.46,.50,.52)]);UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    # Measured points are visual surfaces only, never collision-filled AABBs.
    cloud = np.load(source/'body_cloud_artifacts'/snapshot['environment']['observation']['pointcloud']['pointcloud_id']/'body_cloud.npz')['points_body_m']
    if a.source_scene_id:
        original=read(scene/'scene/compiled_scene.json')['cuboids']
        key=next(k for k in original if k.startswith('observed_'))
        shift=body[:3,:3]@(np.array(request['compiled_scene']['cuboids'][key]['pose'][:3])-original[key]['pose'][:3])
        cloud=np.load(scene/'scene/clean_residual.npz')['points_body_m']+shift
    sampled = cloud[::max(1,len(cloud)//18000)]
    points=UsdGeom.Points.Define(site,'/Site/MeasuredSurfaces');points.CreatePointsAttr(sampled.tolist())
    points.CreateWidthsAttr([.003]);points.CreateDisplayColorAttr([(.10,.63,.61)])
    goal = (body@arm_link_transforms(a.urdf,request['goal_rad'])['link6']@np.array(request['T_link6_tcp']))[:3,3]
    marker=UsdGeom.Sphere.Define(site,'/Site/TransitGoal');marker.CreateRadiusAttr(.012);marker.AddTranslateOp().Set(Gf.Vec3d(*goal));marker.CreateDisplayColorAttr([(.85,.2,.7)])
    site.GetRootLayer().Save()
    manifest={'publication':'CONTROLLED_NOT_PUBLIC','scope':'OCTOBER_TRANSPORT_SEGMENT_WITH_ATTACHED_AABB_PROXY',
              'scene_id':sid,'source_request_sha256':sha(a.request/'planner_request.json'),'source_result_sha256':sha(a.request/'planning.json'),
              'body_base':request['T_body_model'],'tool':request['T_link6_tcp'],'start_q':request['start_rad'],
              'left_hold_q':snapshot['robot_state']['left_joints_rad'],'reference_limits':limits,'native_timing':native,
              'goal_tcp_body_m':goal.tolist(),'collision_world_boxes':len(request['compiled_scene']['cuboids']),
              'robot_sha256':sha(a.output/'right_robot.usda'),'site_sha256':sha(a.output/'site.usda'),
              'original_robot_mesh_colliders':'retained from v5, not full CAD/PhysX certification',
              'tray':'Link6 conservative field envelope, NOT identified metal tray CAD',
              'mass':'inherited research link mass/inertia, not measured field payload',
              'physics_hard_stop_corrections':limit_corrections,
              'soft_limits':'unchanged reference lower+margin/upper-margin; independently checked throughout',
              'left_collision':'archived inactive-arm cuboids; left mesh visual only',
              'target_scope':'Center->Transit, NOT successful Preland or placement',
              'derived_station_scene':bool(a.source_scene_id),
              'scan_surfaces':'visual only; exact compiled planner boxes used as CONSERVATIVE physics proxies'}
    (a.output/'SITE_ASSET_MANIFEST.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','analysis','request','asset','urdf','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--source-scene-id')
    main(p.parse_args())
