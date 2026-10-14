"""Bounded read-only asset/kinematics audit; no hardware or training.

October coordinates are SECONDARY_REPORTED until original archive is supplied.
USD joint-frame FK is computed independently of URDF FK. Historical scan
cuboids stay wireframe proxies and never become inferred free-space labels.
"""
import argparse,hashlib,json,itertools,datetime,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.perception.robot_collision import arm_link_transforms,transform_xyz_rpy

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def world(cache,prim):return np.array(cache.GetLocalToWorldTransform(prim)).T
def rigid(pos,quat):
    t=np.eye(4);t[:3,3]=pos;t[:3,:3]=Rotation.from_quat([*quat.GetImaginary(),quat.GetReal()]).as_matrix();return t
def usd_fk(stage,q):
    frames={'Link_0':np.eye(4)}
    for i,angle in enumerate(q,1):
        j=UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(f'/jaka_minicobo/joints/joint_{i}'))
        axis={'X':[1,0,0],'Y':[0,1,0],'Z':[0,0,1]}[j.GetAxisAttr().Get()]
        rotation=np.eye(4);rotation[:3,:3]=Rotation.from_rotvec(np.array(axis)*angle).as_matrix()
        parent=str(j.GetBody0Rel().GetTargets()[0]).split('/')[-1];child=str(j.GetBody1Rel().GetTargets()[0]).split('/')[-1]
        frames[child]=frames[parent]@rigid(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get())@rotation@np.linalg.inv(rigid(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get()))
    return frames
def bounds(points):
    return {'min_m':points.min(0).tolist(),'max_m':points.max(0).tolist(),'dimensions_m':np.ptp(points,axis=0).tolist(),'center_m':((points.min(0)+points.max(0))/2).tolist()}
def box_vertices(center,dims,rotation=np.eye(3)):
    return np.array(list(itertools.product([-1,1],repeat=3)))*np.array(dims)/2@rotation.T+np.array(center)

p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True);p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=PinnedRobotModel(a.ares,a.ares_r);main=a.ares/'isaac_sim/simforge/scenes/main.usd';bare=a.ares/'isaac_sim/simforge/assets/jaka_minicobo/jaka_minicobo.usd'
scene=Usd.Stage.Open(str(main));arm=Usd.Stage.Open(str(bare));cache=UsdGeom.XformCache()
frame={'evidence':'HISTORICAL + REPLAY_VERIFIED_OFFLINE_MODEL; NOT_OCTOBER_LIVE',
    'requested_ares_base_commit':'3c20dcd','actual_ares_head':subprocess.check_output(['git','-C',str(a.ares),'rev-parse','HEAD'],text=True).strip(),
    'ares_r_head':subprocess.check_output(['git','-C',str(a.ares_r),'rev-parse','HEAD'],text=True).strip(),
    'matrix_convention':'column vectors, p_parent=T_parent_child @ p_child','BODY_axes':m.world['frame'],
    'research_T_world_body':np.eye(4).tolist(),'research_world_body_status':'CHOSEN_RESEARCH_COORDINATES_NOT_MEASURED',
    'meters_per_unit':UsdGeom.GetStageMetersPerUnit(scene),'up_axis':UsdGeom.GetStageUpAxis(scene),'arms':{},'samples':[]}
guesses={}
poses=read(a.ares_r/'config/named_poses.json')['poses']
for side in ['left','right']:
    b=m.body_bases[side];w=world(cache,scene.GetPrimAtPath(f'/World/robot/jaka_minicobo_{side}/Link_0'));guesses[side]=w@np.linalg.inv(b)
    tool=m.audits[side]['diagnostics']['tool_data'];tcp=transform_xyz_rpy(np.array(tool['pose_mm_rad'][:3])*.001,tool['pose_mm_rad'][3:])
    frame['arms'][side]={'site_T_body_controller_base':__import__('ares_r.perception.robot_collision',fromlist=['_world_arm_transform'])._world_arm_transform(m.world,side).tolist(),
        'T_controller_model':m.audits[side]['T_controller_model'],'T_body_model_base':b.tolist(),
        'original_T_usd_world_link0':w.tolist(),'original_candidate_T_world_body':guesses[side].tolist(),
        'T_original_world_to_research_body_subtree':(b@np.linalg.inv(w)).tolist(),
        'controller_tool':tool,'T_link6_tcp_historical':tcp.tolist(),
        'joint_order_urdf':[f'joint{i}' for i in range(1,7)],'joint_order_usd':[f'joint_{i}' for i in range(1,7)],
        'source_sdk_checks':len(m.audits[side]['samples']),
        'historical_sdk_max_position_error_mm':max(x['position_error_mm'] for x in m.audits[side]['samples']),
        'october_q_current':None,'october_tcp':None,'october_gripper_opening':None}
    qs=[x['q_rad'] for x in m.audits[side]['samples']]
    for index,q in enumerate(qs):
        u=arm_link_transforms(m.urdf,q);s=usd_fk(arm,q);errors=[]
        for i in range(7):
            t1=b@u['base_link' if i==0 else f'link{i}'];t2=b@s[f'Link_{i}']
            errors.append({'link':i,'position_error_m':float(np.linalg.norm(t1[:3,3]-t2[:3,3])),
                'rotation_error_rad':float(Rotation.from_matrix(t1[:3,:3]@t2[:3,:3].T).magnitude())})
        frame['samples'].append({'side':side,'sample_id':index,'q_rad':q,'evidence':'REPLAY_VERIFIED_USD_JOINT_FRAMES_VS_URDF_NO_PHYSX','errors':errors})
delta=np.linalg.inv(guesses['left'])@guesses['right']
frame['single_global_transform_consistent']=bool(np.allclose(guesses['left'],guesses['right'],atol=1e-4))
frame['global_candidate_disagreement']={'translation_m':float(np.linalg.norm(delta[:3,3])),'rotation_deg':float(np.degrees(Rotation.from_matrix(delta[:3,:3]).magnitude()))}
frame['usd_urdf_max_position_error_m']=max(e['position_error_m'] for row in frame['samples'] for e in row['errors'])
frame['usd_urdf_max_rotation_error_rad']=max(e['rotation_error_rad'] for row in frame['samples'] for e in row['errors'])
frame['site_historical_limits']=read(a.ares_r/'config/jaka_mini2_motion.site.json')
frame['chassis_historical_proxies']=m.collision_model['fixed_body_boxes']
system=read(a.ares_r/'config/system.json');pipeline=read(a.ares_r/'config/pointcloud_pipeline.site.json')
frame['camera']={'system_epic_pointcloud':{k:v for k,v in system['epic_pointcloud'].items() if k.startswith('T_body_camera')},
    'pipeline_top_level_state':pipeline['state'],'pipeline_T_body_camera':pipeline['T_body_camera'],
    'authority':'September system value crosschecked by dated report; October use requires actual run revisions'}
(a.output/'FRAME_ALIGNMENT.json').write_text(json.dumps(frame,indent=2))

mesh=UsdGeom.Mesh(scene.GetPrimAtPath('/World/Tray/Mesh'));raw=np.array(mesh.GetPointsAttr().Get())
tm=world(cache,mesh.GetPrim());root=world(cache,scene.GetPrimAtPath('/World/Tray'));world_points=raw@tm[:3,:3].T+tm[:3,3]
in_root=world_points@np.linalg.inv(root)[:3,:3].T+np.linalg.inv(root)[:3,3]
geometry={'evidence':'CAD_OLD_USD_NOT_VERIFIED_FIELD_OBJECT','mesh_vertices':len(raw),'raw_mesh_coordinate_bounds':bounds(raw),
    'T_world_mesh':tm.tolist(),'T_world_tray_root':root.tolist(),'world_bounds':bounds(world_points),
    'tray_root_frame_bounds':bounds(in_root),'root_to_mesh_bbox_center_m':bounds(in_root)['center_m'],
    'markers':{},'collision_proxies':{},'october_attached_object':None,'same_field_object_verified':False}
for prim in Usd.PrimRange(scene.GetPrimAtPath('/World/Tray')):
    path=str(prim.GetPath())
    if 'GraspFrames/' in path or path.endswith(('tray_grasp_point','tray_grasp_init_point')):
        w=world(cache,prim);geometry['markers'][path]={'T_world_frame':w.tolist(),'T_tray_root_frame':(np.linalg.inv(root)@w).tolist(),
            'offset_from_mesh_bbox_center_m':(w[:3,3]-np.array(geometry['world_bounds']['center_m'])).tolist()}
    if prim.IsA(UsdGeom.Cube):
        geometry['collision_proxies'][path]={'T_world':world(cache,prim).tolist(),'cube_size':UsdGeom.Cube(prim).GetSizeAttr().Get()}
legacy=[geometry['markers'].get('/World/Tray/'+key) for key in ['tray_grasp_point','tray_grasp_init_point']]
geometry['legacy_marker_separation_m']=float(np.linalg.norm(np.array(legacy[0]['T_world_frame'])[:3,3]-np.array(legacy[1]['T_world_frame'])[:3,3]))
np.savez_compressed(a.output/'old_tray_geometry.npz',vertices_world_m=world_points,vertices_tray_root_m=in_root,face_counts=np.array(mesh.GetFaceVertexCountsAttr().Get()),face_indices=np.array(mesh.GetFaceVertexIndicesAttr().Get()))
(a.output/'TRAY_GEOMETRY.json').write_text(json.dumps(geometry,indent=2))

snapshot=a.ares_r/'worklog/evidence/2026-09-21-p3-production-demo/clear_final/scene/snapshot.json';frozen=read(snapshot)
obs=frozen['environment']['observation']
field={'evidence':'HISTORICAL_SCAN_PROXY_ONLY_NOT_OCTOBER_RECONSTRUCTION','source':str(snapshot),'snapshot':frozen,
    'source_sha256':sha(snapshot),'raw_cloud_present':False,'raw_cloud_sha256_reported':obs['pointcloud']['sha256'],
    'known_free_space_verified':False,'cabinet_cavity_reconstructed':False,'solid_occupancy_from_residual_aabb_allowed':False,
    'october_target_reported_body_m':[.604866,-.184327,1.152342],'october_preland_reported_body_m':[.604866,-.184327,1.222342],
    'october_yaw_reported_rad':.044793,'october_parameters_evidence':'UNVERIFIED_SECONDARY_USER_AND_SITE_20261012_REFERENCE',
    'october_space_object_camera':[2,4,1],'october_original_archive_present':False}
(a.output/'FIELD_REFERENCE.json').write_text(json.dumps(field,indent=2))
sources=[main,bare,m.urdf,a.ares_r/'config/robot_world.json',a.ares_r/'config/system.json',a.ares_r/'config/pointcloud_pipeline.site.json',a.ares_r/'config/robot_collision_model.json',a.ares_r/'config/jaka_mini2_motion.site.json',snapshot,
    a.ares/'research/skill_handoff_lab/SITE_20261012_REFERENCE.md']
sources.extend(a.ares_r/f'worklog/evidence/2026-09-21-p3-production-demo/block_final/{side}_fk_audit.json' for side in ['left','right'])
(a.output/'source_manifest.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'hardware_io':False,'physics_training':False,'sources':[{'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size} for p in sources],
    'october_archive_missing':True,'source_program_sha256':sha(Path(__file__))},indent=2))
print(json.dumps({'global_transform_consistent':frame['single_global_transform_consistent'],'global_difference':frame['global_candidate_disagreement'],
    'independent_joint_samples':len(frame['samples']),'max_fk_position_error_m':frame['usd_urdf_max_position_error_m'],
    'tray_world_dims_m':geometry['world_bounds']['dimensions_m'],'legacy_grasp_marker_separation_m':geometry['legacy_marker_separation_m']}))
