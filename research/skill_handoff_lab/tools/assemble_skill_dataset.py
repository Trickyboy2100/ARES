"""Qualify real paths, bind causal suffix requests, then freeze planning cache."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import yaml
from controlled_paths import require_private_location

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def qualify_one(job):
    folder,root,urdf,limits,ignore=job;folder=Path(folder);root=Path(root)
    from fast_proxy_collision import independent_validator
    from research.skill_handoff_lab.cash import qualified_planning as qp
    qp.validate_dense_world=independent_validator()[0]
    request=read(folder/'request.json');result=read(folder/'result.json');start=time.perf_counter()
    q=result.get('trajectory_points_rad') or [];qualification={'qualified':False,'independent_validation':{}}
    curve=None
    if result.get('observed_result')=='SUCCESS' and len(q)>=2:
        curve=qp.smooth_timing(q,limits)
        qualification=qp.qualify(curve['joints_rad'],request,Path(urdf),limits,ignore)
        np.savez_compressed(folder/'c2_curve.npz',**curve)
    qualification.update(request_sha256=sha(folder/'request.json'),result_sha256=sha(folder/'result.json'),wall_s=time.perf_counter()-start)
    (folder/'contract_qualification.json').write_text(json.dumps(qualification,indent=2))
    return str(folder),qualification

def main(a):
    require_private_location(a.root);os.umask(0o077);root=a.root.resolve()
    graph=read(root/'PLANNING_GRAPH.json');family=read(root/'TASK_FAMILY.json')
    by_hash={}
    for folder in a.sources:
        for file in (root/folder).glob('*/validation.json'):
            if file.with_name('result.json').exists():by_hash[sha(file.with_name('request.json'))]=file.parent
    ignore=yaml.safe_load(a.yaml.read_text())['kinematics']['self_collision_ignore']
    needed=[edge for task in graph['tasks'] for edge in task['edges'] if edge['from']=='CENTER' and (edge['to']=='PRELAND' or edge['profile']=='FAST')]
    if a.mode=='freeze':needed+=read(root/'CAUSAL_SUFFIX_GRAPH.json')['edges']
    missing=[edge['edge_id'] for edge in needed if edge['request_sha256'] not in by_hash]
    if missing:raise RuntimeError(f'Missing {len(missing)} actually attempted edges; do not invent outcomes: {missing[:3]}')
    jobs=[(str(by_hash[e['request_sha256']]),str(root),str(a.urdf.resolve()),family['limits'],ignore)
          for e in needed if not (by_hash[e['request_sha256']]/'contract_qualification.json').exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for folder,qualification in pool.map(qualify_one,jobs):
            print(Path(folder).parent.name,Path(folder).name,qualification['qualified'],flush=True)
    if a.mode=='suffix':
        directory=root/'causal_suffix_requests';directory.mkdir(mode=0o700,exist_ok=False)
        edges=[];batches={p:[] for p in ['FAST','FALLBACK']}
        for task in graph['tasks']:
            for edge in task['edges']:
                if edge['from']!='CENTER' or edge['to']=='PRELAND' or edge['profile']!='FAST':continue
                parent=by_hash[edge['request_sha256']]
                if not read(parent/'contract_qualification.json')['qualified']:continue
                end=read(parent/'result.json')['trajectory_points_rad'][-1]
                for profile in ['FAST','FALLBACK']:
                    template=next(e for e in task['edges'] if e['from']==edge['to'] and e['profile']==profile)
                    request=read(root/template['request']);request['start_rad']=end
                    name=edge['edge_id']+'__NEXT_'+profile
                    p=directory/(name+'.json');p.write_text(json.dumps(request,indent=2))
                    item=dict(template,edge_id=name,from_node='AFTER_'+edge['edge_id'],to_node='PRELAND',
                              request=str(p.relative_to(root)),request_sha256=sha(p),parent_edge=edge['edge_id'],task_id=task['task_id'])
                    edges.append(item);batches[profile].append(a.remote_directory+'/'+str(p.relative_to(root)))
        (root/'CAUSAL_SUFFIX_GRAPH.json').write_text(json.dumps({'edges':edges,'exact_predecessor_terminal_binding':True},indent=2))
        for profile,values in batches.items():(root/f'batch_causal_{profile}.json').write_text(json.dumps(values,indent=2))
        print('Causal suffixes:',len(edges));return
    dataset={'fixture':False,'evidence_level':'REAL_CUROBO_PLANNING_CACHE','tasks':[],'edges':{}}
    suffix=read(root/'CAUSAL_SUFFIX_GRAPH.json')['edges']
    all_edges=needed
    for task in graph['tasks']:
        initial=list(family['prefix_request']['start_rad'])
        item={'task_id':task['task_id'],'scene_group':task['task_id'],'split':task['split'],
              'scene_digest':task['scene_digest'],'initial_q':initial,'goal_xyz':task['goal_body_m'],
              'observed_context':task['station_shift_body_m']+[np.sin(family['original_request']['orientation_lock']['final_yaw_target_rad']),
                  np.cos(family['original_request']['orientation_lock']['final_yaw_target_rad'])], 'actions':{'CENTER':[]}}
        for edge in all_edges:
            if not edge['edge_id'].startswith(task['task_id']+'__'):continue
            folder=by_hash[edge['request_sha256']];request=read(folder/'request.json');result=read(folder/'result.json');qualification=read(folder/'contract_qualification.json')
            prefix=edge.get('from')=='CENTER'
            destination=edge.get('to','PRELAND');node='CENTER' if prefix else edge['from_node']
            target='PRELAND' if destination=='PRELAND' else 'AFTER_'+edge['edge_id']
            q=result.get('trajectory_points_rad') or []
            qualified=bool(qualification['qualified'])
            # Preserve original desired Preland position/yaw; don't move the
            # task goal to whichever endpoint the numerical IK happened to find.
            if qualified and destination=='PRELAND':
                from ares_r.motion.runtime_goal_ik import RuntimeGoalIK
                solver=RuntimeGoalIK(a.urdf,request['T_body_model'],request['T_link6_tcp'])
                if np.linalg.norm(solver.fk(q[-1])[:3,3]-task['goal_body_m'])>.003:qualified=False
            curve=np.load(folder/'c2_curve.npz') if (folder/'c2_curve.npz').exists() else None
            roles=[('request','request.json'),('result','result.json'),('validation','contract_qualification.json'),
                   ('oracle_report','validation.json')]
            if curve is not None:roles.append(('timing','c2_curve.npz'))
            artifacts=[{'role':role,'path':str((folder/name).relative_to(root)),'sha256':sha(folder/name)} for role,name in roles]
            original_report=read(folder/'validation.json')
            failure_class=original_report.get('failure_class')
            status='QUALIFICATION_REJECT' if result.get('observed_result')=='SUCCESS' else 'PLANNER_FAILURE'
            if failure_class in ('TIMEOUT','SOFTWARE_ERROR','UNKNOWN'):status=failure_class
            if failure_class in ('OOM','DRIVER_COMPATIBILITY'):status='SOFTWARE_ERROR'
            duration=float(curve['duration_s']) if curve is not None else 0.
            planning=float(sum(result.get('timing_s',{}).get('planning_samples',[]) or [0.]))
            record={'edge_id':edge['edge_id'],'from_node':node,'to_node':target,'scene_digest':task['scene_digest'],
                    'start_q':request['start_rad'],'end_q':q[-1] if qualified else request['start_rad'],
                    'qualified':qualified,'planner_success':result.get('observed_result')=='SUCCESS','attempt_status':status,
                    'original_oracle_failure_class':failure_class,
                    'planning_wall_s':planning,'planned_duration_s':duration,'attempt_cap':edge['attempt_cap'],
                    'independent_validation':qualification['independent_validation'],'artifacts':artifacts,
                    'request_sha256':sha(folder/'request.json'),'curve':str((folder/'c2_curve.npz').relative_to(root)) if curve is not None else None}
            dataset['edges'][edge['edge_id']]=record
            features=[float(destination==d) for d in ['PRELAND','TRANSIT','LIFT']]+[float(edge['profile']=='FAST'),float(edge['profile']=='FALLBACK')]
            features+=task['goal_body_m']+[float(np.linalg.norm(np.array(request['goal_rad'])-request['start_rad'])),edge['attempt_cap']/3.]
            endpoint=np.array(request['goal_rad']);margin=family['limits']['soft_limit_margin_rad']
            endpoint_legal=bool(np.all(endpoint>=np.array(family['limits']['lower_rad'])+margin)
                                and np.all(endpoint<=np.array(family['limits']['upper_rad'])-margin))
            action={'edge_id':edge['edge_id'],'features':features,'endpoint_legal':endpoint_legal}
            item['actions'].setdefault(node,[]).append(action)
        for action in item['actions']['CENTER']:
            record=dataset['edges'][action['edge_id']]
            if record['qualified'] and record['to_node']!='PRELAND' and record['to_node'] not in item['actions']:
                raise ValueError('missing causal successor state')
        dataset['tasks'].append(item)
    (root/'SKILL_RL_DATASET.json').write_text(json.dumps(dataset,indent=2))
    print('Dataset tasks/edges:',len(dataset['tasks']),len(dataset['edges']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--mode',choices=['suffix','freeze'],required=True);p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--yaml',type=Path,required=True);p.add_argument('--remote-directory',default='');main(p.parse_args())
