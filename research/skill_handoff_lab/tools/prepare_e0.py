"""Frozen, bounded E0 held-envelope transport (not contact grasp or site CAD).

Keep the full historical chassis proxy and site limits. Correct object-axis
mapping explicitly; never change ARES-R production configuration.
"""
import argparse, hashlib, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from research.skill_handoff_lab.tools.prepare_oracle_batch import digest
from ares_r.world import SceneObject, SceneObjectRole, PoseSE3
from ares_r.manipulation.attached_collision import attach_scene_object, build_attached_collision
from ares_r.motion.tcp_orientation import level_rotation
from ares_r.motion.curobo_params import DEFAULTS
from ares_r.motion.independent_path_validation import validate_dense_world


def main():
    p=argparse.ArgumentParser(); p.add_argument('--ares',type=Path,required=True)
    p.add_argument('--ares-r',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--remote-output',required=True); p.add_argument('--tray-usd',type=Path,required=True)
    p.add_argument('--chassis-policy',choices=['full','historical_lower'],default='full')
    p.add_argument('--continuous-goal',action='store_true',help='one deterministic IK continuation per retained start')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    m=PinnedRobotModel(a.ares,a.ares_r)
    stage=Usd.Stage.Open(str(a.tray_usd)); tray=stage.GetPrimAtPath('/World/Tray')
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy']).ComputeWorldBound(tray).ComputeAlignedRange()
    # A normalized BODY-aligned research envelope, not the mesh's unusual local axes.
    dims=np.array(bbox.GetSize(),float)
    provenance={'usd':str(a.tray_usd),'usd_sha256':hashlib.sha256(a.tray_usd.read_bytes()).hexdigest(),
        'prim':'/World/Tray','observed_world_aabb_m':dims.tolist(),
        'geometry_scope':'research cuboid enclosing this existing USD tray; not measured physical tray CAD',
        'grasp_scope':'already held with fixed attachment; no contact-grasp claim',
        'offset_scope':'frozen research edge-grasp assumption, not live calibrated grasp'}
    root=ET.parse(m.urdf)
    for link in root.findall('link'):
        for item in list(link):
            if item.tag in ('visual','collision'):link.remove(item)
    for joint in root.findall('joint'):
        name=joint.get('name','')
        if name.startswith('4C2_'):
            joint.set('type','fixed')
            for item in list(joint):
                if item.tag in ('mimic','limit','axis'):joint.remove(item)
        if name.startswith('joint'):
            i=int(name[5:])-1; limit=joint.find('limit')
            limit.set('lower',str(m.lower[i]));limit.set('upper',str(m.upper[i]))
    root.write(a.output/'robot.urdf')
    cfg=yaml.safe_load((m.urdf.parent/'jaka_minicobo_curobo.yml').read_text());k=cfg['kinematics']
    names=k['cspace']['joint_names']; ix=[names.index('joint%d'%i) for i in range(1,7)]
    for key,value in list(k['cspace'].items()):
        if isinstance(value,list) and len(value)==len(names): k['cspace'][key]=[value[i] for i in ix]
    k['cspace']['joint_names']=['joint%d'%i for i in range(1,7)]
    k.update(urdf_path=a.remote_output+'/robot.urdf',asset_root_path=a.remote_output,load_meshes=False,
        mesh_link_names=None,lock_joints=None,tool_frames=['link6'])
    (a.output/'robot.yml').write_text(yaml.safe_dump(cfg))
    cuboids={};inverse=np.linalg.inv(m.body_base)
    def add(name,center,rotation,d):
        t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=center;t=inverse@t
        q=Rotation.from_matrix(t[:3,:3]).as_quat()
        cuboids[name]={'dims':list(d),'pose':[*t[:3,3],q[3],*q[:3]]}
    for b in m.fixed_context_boxes():add(b.geometry_id,b.center_body_m,b.rotation_body,np.array(b.half_extents_m)*2)
    for b in m.collision_model['fixed_body_boxes']:
        if b['kind']!='central_exclusion' and b.get('filter_owned'):
            center=b['center_body_m']; size=b['dims_m']
            if b['geometry_id']=='body/chassis' and a.chassis_policy=='historical_lower':
                # Exact historical CLEAR planning cuboid in BODY, not a CAD correction.
                historical=json.loads((a.ares_r/'worklog/evidence/2026-09-21-p3-production-demo/clear_final/scene/planner_request.json').read_text())
                old=historical['compiled_scene']['cuboids']['body_chassis_lower']; t=np.eye(4)
                t[:3,:3]=Rotation.from_quat([*old['pose'][4:],old['pose'][3]]).as_matrix();t[:3,3]=old['pose'][:3]
                t=np.array(historical['T_body_model'])@t
                add('body/chassis_historical_lower',t[:3,3],t[:3,:3],old['dims']);continue
            add(b['geometry_id'],center,np.eye(3),size)
    # Wide table below the whole transport region; no artificial aperture failures.
    add('E0/table',[.58,-.48,.84],np.eye(3),[.8,.65,.06])
    frozen={'seed':6200,'ik_attempts_per_pose':12,'tcp_pairs_m':[
        [[.38,-.40,1.40],[.48,-.40,1.40]],
        [[.40,-.48,1.45],[.50,-.48,1.45]]],
        'yaw_rad':[-np.pi/4,0.],'roll_branches':[0,1],
        'max_ik_per_endpoint':2,'gpu_jobs_max':8,'collision_cell_m':.020,'chassis_policy':a.chassis_policy,
        'continuous_goal':a.continuous_goal}
    (a.output/'frozen_design.json').write_text(json.dumps({'design':frozen,'tray':provenance},indent=2))
    base={'planning_only':True,'execution_allowed':False,'mode':'CLEAR','active_arm':'right',
        'T_body_model':m.body_base.tolist(),'T_link6_tcp':m.link6_tool.tolist(),
        'geometry_revision':digest(m.collision_model),'inactive_arm_revision':digest(m.parked_left_q),
        'collision_model':m.collision_model,'robot_yaml':a.remote_output+'/robot.yml',
        'robot_yaml_urdf':a.remote_output+'/robot.urdf','expected_commit':'8e734f3ced1df898990bcd92de40abce475907db',
        'planning_parameters':dict(DEFAULTS,num_trajopt_seeds=4,num_ik_seeds=4,max_attempts=2,active_sphere_cell_m=.020),
        'benchmark_runs':1,'site_soft_limits':{'lower_rad':m.lower.tolist(),'upper_rad':m.upper.tolist()},
        'collision_fidelity':{'active_arm_cell_m':.020},
        'coverage_limitations':[f'chassis policy {a.chassis_policy}; not measured CAD; upper-body safety unqualified',
            'research held cuboid, no contact grasp or physics'],
        'experiment_role':'E0_HELD_TRANSPORT_PLANNING_ONLY'}
    jobs=[]; rows=[]
    for pi,pair in enumerate(frozen['tcp_pairs_m']):
      for yi,yaw in enumerate(frozen['yaw_rad']):
       for roll in frozen['roll_branches']:
        r=level_rotation(yaw,roll); rz=Rotation.from_euler('z',yaw).as_matrix()
        # Object is flat in BODY, its near X edge lies 5 mm in front of TCP.
        object_center=np.array(pair[0])+rz@np.array([dims[0]/2+.005,0,0])
        xyzw=Rotation.from_matrix(rz).as_quat()
        geometry=SceneObject('usd_tray_envelope',SceneObjectRole.TARGET,'cuboid',
            PoseSE3('body',tuple(object_center),(xyzw[3],*xyzw[:3])),tuple(dims),0.,digest(provenance))
        tcp=np.eye(4);tcp[:3,:3]=r;tcp[:3,3]=pair[0]
        attached=build_attached_collision(attach_scene_object(geometry,tcp,side='right',source_revision=digest(provenance)),m.link6_tool,.008)
        endpoints=[]
        for ei,xyz in enumerate(pair):
            solutions=m.solve(xyz,yaw,roll,[m.commissioned_predecessor_q],6200+pi*100+yi*10+roll,attempts=12)[:2]
            legal=[]
            for si,sol in enumerate(solutions):
                q=sol[1];request=dict(base,attached_object_collision=attached,start_rad=q.tolist(),goal_rad=q.tolist(),
                    compiled_scene={'cuboids':cuboids},motion_constraints={'central_exclusion':True,'attached_object_revision':attached['revision']})
                request['robot_yaml_urdf']=str(a.output/'robot.urdf')
                v=validate_dense_world([q,q],request,subdivisions=2)
                row={'pair':pi,'yaw':yaw,'roll':roll,'endpoint':ei,'ik':si,'q':q.tolist(),'fk':m.fk(q).tolist(),'world':v}
                rows.append(row)
                if v['collision_free']:legal.append((si,q))
            endpoints.append(legal)
        combinations=[(si,start,gi,goal) for si,start in endpoints[0] for gi,goal in endpoints[1]]
        if a.continuous_goal:
            combinations=[]
            for si,start in endpoints[0]:
                solutions=m.solve(pair[1],yaw,roll,[start],6200,attempts=1)
                if solutions:combinations.append((si,start,si,solutions[0][1]))
        for si,start,gi,goal in combinations:
          if len(jobs)>=8:continue
          sid=f'E0-p{pi}-y{yi}-r{roll}';cid=f'{sid}-ik{si}-{gi}'
          binding=digest({'scene':cuboids,'design':frozen,'tray':provenance,'sid':sid})
          scene={'planning_scope':'P3_PRODUCTION_PLANNING_ONLY','execution_allowed':False,
              'scene_snapshot_id':sid,'digest':binding,'cuboids':cuboids,
              'planning_context_digest':digest(cuboids),'calibration_revision':m.tool_contract}
          request=dict(base,compiled_scene=scene,scene_snapshot_id=sid,scene_digest=binding,candidate_id=cid,
              start_rad=start.tolist(),goal_rad=goal.tolist(),attached_object_collision=attached,
              motion_constraints={'central_exclusion':True,'attached_object_revision':attached['revision']},
              orientation_lock={'policy':'FIXED_ROTATION_V1','criteria_rpy':[1.,1.,1.],
                  'max_error_deg':3.,'target_R_body_tcp':r.tolist()},tray_provenance=provenance,
              candidate_features={'pair':pi,'yaw_rad':yaw,'roll_branch':roll,'start_ik':si,'goal_ik':gi})
          name=f'request_{len(jobs)}.json';(a.output/name).write_text(json.dumps(request,indent=2));jobs.append(a.remote_output+'/'+name)
    (a.output/'endpoint_preflight.json').write_text(json.dumps(rows,indent=2))
    (a.output/'batch.json').write_text(json.dumps(jobs,indent=2))
    print(json.dumps({'endpoints':len(rows),'collision_free':sum(r['world']['collision_free'] for r in rows),'jobs':len(jobs)}))

if __name__=='__main__':main()
