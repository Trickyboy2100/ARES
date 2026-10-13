"""Export a bounded frozen-data batch without relabeling CPU proxy results."""
import argparse, json, hashlib, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from research.skill_handoff_lab.bench.model import PinnedRobotModel
from ares_r.world import AttachedObject,SceneObject,SceneObjectRole,PoseSE3
from ares_r.manipulation.attached_collision import build_attached_collision
from ares_r.motion.curobo_params import DEFAULTS


def read_rows(path): return [json.loads(line) for line in path.read_text().splitlines()]
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--ares',type=Path,required=True)
    p.add_argument('--ares-r',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--remote-output',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    model=PinnedRobotModel(a.ares,a.ares_r); root=ET.parse(model.urdf)
    for link in root.findall('link'):
        for item in list(link):
            if item.tag in ('visual','collision'):link.remove(item)
    for joint in root.findall('joint'):
        if joint.get('name','').startswith('4C2_'):
            joint.set('type','fixed')
            for item in list(joint):
                if item.tag in ('mimic','limit','axis'):joint.remove(item)
        if joint.get('name','').startswith('joint'):
            idx=int(joint.get('name')[5:])-1; limit=joint.find('limit')
            limit.set('lower',str(model.lower[idx]));limit.set('upper',str(model.upper[idx]))
    root.write(a.output/'robot.urdf')
    cfg=yaml.safe_load((model.urdf.parent/'jaka_minicobo_curobo.yml').read_text())
    kin=cfg['kinematics'];kin.update(urdf_path=a.remote_output+'/robot.urdf',
        asset_root_path=a.remote_output,load_meshes=False,mesh_link_names=None,lock_joints=None,
        tool_frames=['link6'])
    names=kin['cspace']['joint_names']; indexes=[names.index('joint%d'%i) for i in range(1,7)]
    for key,value in list(kin['cspace'].items()):
        if isinstance(value,list) and len(value)==len(names):kin['cspace'][key]=[value[i] for i in indexes]
    kin['cspace']['joint_names']=['joint%d'%i for i in range(1,7)]
    (a.output/'robot.yml').write_text(yaml.safe_dump(cfg))
    scenarios={x['scenario_id']:x for x in read_rows(a.data/'scenarios.jsonl')}
    transitions={x['candidate_id']:x for x in read_rows(a.data/'transitions.jsonl')}
    candidates=read_rows(a.data/'candidates.jsonl');batch=[];counts={}
    for c in candidates:
        sid=c['scenario_id'];t=transitions[c['candidate_id']]
        if sid not in ('cabinet-4201-p21','mechanism-5102') or not c['candidate_legal'] or not t['geometric_start_set_member']:continue
        if counts.get(sid,0)>=2:continue
        counts[sid]=counts.get(sid,0)+1;s=scenarios[sid];cuboids={}
        inverse=np.linalg.inv(model.body_base)
        def add_box(name,center,rotation,dims):
            pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=center;pose=inverse@pose
            xyzw=Rotation.from_matrix(pose[:3,:3]).as_quat()
            cuboids[name]={'dims':list(dims),'pose':[*pose[:3,3],xyzw[3],*xyzw[:3]]}
        for box in model.fixed_context_boxes():add_box(box.geometry_id,box.center_body_m,box.rotation_body,np.asarray(box.half_extents_m)*2)
        for box in s['obstacles']:add_box(box['id'],box['center_body_m'],np.eye(3),box['dims_m'])
        # Preserve conservative BODY/chassis proxy rather than silently dropping it.
        for box in model.collision_model['fixed_body_boxes']:
            if box['kind']!='central_exclusion' and box.get('filter_owned'):
                add_box(box['geometry_id'],box['center_body_m'],np.eye(3),box['dims_m'])
        obj=s['object_contract']; geometry=SceneObject(obj['object_id'],SceneObjectRole.TARGET,
            'cuboid',PoseSE3('body',(0.,0.,0.),(1.,0.,0.,0.)),tuple(obj['dimensions_m']),0.,s['scene_revision'])
        attachment=AttachedObject(obj['object_id'],'right',PoseSE3('tcp',tuple(obj['tcp_offset_m']),
            (1.,0.,0.,0.)),geometry,obj['revision'])
        attached=build_attached_collision(attachment,model.link6_tool,inflation_m=0.)
        scene={'planning_scope':'P3_PRODUCTION_PLANNING_ONLY','execution_allowed':False,
            'scene_snapshot_id':sid,'digest':s['scene_revision'],'cuboids':cuboids,
            'planning_context_digest':digest(cuboids),'calibration_revision':model.tool_contract}
        request={'planning_only':True,'execution_allowed':False,'mode':'CLEAR','compiled_scene':scene,
            'scene_snapshot_id':sid,'scene_digest':s['scene_revision'],'candidate_id':c['candidate_id'],
            'source_candidate':c,'source_scene':s,'source_transition':t,'start_rad':c['center_joints_rad'],
            'goal_rad':t['preplace_joints_rad'],'active_arm':'right','T_body_model':model.body_base.tolist(),
            'T_link6_tcp':model.link6_tool.tolist(),'geometry_revision':digest(model.collision_model),
            'inactive_arm_revision':digest(model.parked_left_q),'collision_model':model.collision_model,
            'robot_yaml':a.remote_output+'/robot.yml','robot_yaml_urdf':a.remote_output+'/robot.urdf',
            'expected_commit':'8e734f3ced1df898990bcd92de40abce475907db',
            'planning_parameters':dict(DEFAULTS,num_trajopt_seeds=4,num_ik_seeds=4,max_attempts=2),
            'benchmark_runs':1,'attached_object_collision':attached,
            'site_soft_limits':{'lower_rad':model.lower.tolist(),'upper_rad':model.upper.tolist()},
            'motion_constraints':{'central_exclusion':True,'attached_object_revision':attached['revision']},
            'orientation_lock':{'policy':'LEVEL_YAW_TARGET_V1','criteria_rpy':[1.,1.,0.],
                'max_error_deg':3.,'final_yaw_target_rad':s['entrance_yaw_rad']},
            'coverage_limitations':['chassis uses uncommissioned conservative proxy',
                'independent self-collision not yet qualified','synthetic aperture is not measured site CAD']}
        path=a.output/('request_%d.json'%len(batch));path.write_text(json.dumps(request,indent=2))
        batch.append(a.remote_output+'/'+path.name)
    diagnostic=json.loads((a.ares_r/'worklog/evidence/2026-09-21-p3-production-demo/clear_final/scene/planner_request.json').read_text())
    diagnostic.update(robot_yaml=a.remote_output+'/robot.yml',robot_yaml_urdf=a.remote_output+'/robot.urdf',
        benchmark_runs=1,experiment_role='HISTORICAL_SCENE_REPLAY_NOT_SKILL_HANDOFF_DATA',
        site_soft_limits={'lower_rad':model.lower.tolist(),'upper_rad':model.upper.tolist()})
    diagnostic['planning_parameters'].update(num_trajopt_seeds=4,num_ik_seeds=4,max_attempts=2)
    (a.output/'request_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    batch.insert(0,a.remote_output+'/request_diagnostic.json')
    (a.output/'batch.json').write_text(json.dumps(batch,indent=2))
    (a.output/'manifest.json').write_text(json.dumps({'counts':counts,'data_sha256':{
        path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in a.data.glob('*.jsonl')},
        'source_urdf_sha256':hashlib.sha256(model.urdf.read_bytes()).hexdigest(),
        'planning_only':True,'groove_114mm_original_preserved':True},indent=2))
    print(json.dumps({'counts':counts,'jobs':len(batch)}))


if __name__=='__main__':main()
