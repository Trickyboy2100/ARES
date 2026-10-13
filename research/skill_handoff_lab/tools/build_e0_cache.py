"""Actual validated E0 transport cost cache, one PREPLACE decision, not D0-D2.

Only complete pre-frozen two-action groups qualify. No success-rate learning
claim: all retained actions are known feasible under the declared proxy model.
Rejected/unknown requests stay in raw data, never become false transitions.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from research.skill_handoff_lab.tools.prepare_oracle_batch import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--execution',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();groups={};excluded=[]
    for job in sorted(a.execution.iterdir()):
        if not (job/'e0_independent_validation.json').exists():continue
        r=json.loads((job/'request.json').read_text());v=json.loads((job/'e0_independent_validation.json').read_text())
        groups.setdefault(r['scene_group'],[]).append((job,r,v))
    episodes=[]
    for group,rows in groups.items():
        if len(rows)!=2 or not all(v['model_validated_path'] for _,_,v in rows):
            excluded.append({'group':group,'reason':'incomplete or unqualified; no negative label'});continue
        actions=[];transitions=[];example=rows[0][1]
        for job,r,v in rows:
            # Feature sources are request geometry/joints only. No outcome or ID feature.
            start=np.array(r['start_rad']);goal=np.array(r['goal_rad'])
            features=[*start,*goal,float(r['candidate_features']['roll_branch']),float(np.linalg.norm(goal-start))]
            actions.append({'action_id':r['candidate_id'],'stage':'PREPLACE_PLAN','features':features,'legal':True})
            flags={'collision':v['world']['collision_free'],'self_collision':not v['self_collision_codes'],
                'attached_object':not v['held_arm_collision_codes'],'soft_limits':v['soft_limits'],
                'orientation':v['orientation']['passed'],'scene_binding':v['scene_binding']}
            # Bind cache evidence to the existing independent report, not invented labels.
            report_path=job/'cache_validation.json'
            report={'independent_validation':flags,'source_validation_sha256':hashlib.sha256((job/'e0_independent_validation.json').read_bytes()).hexdigest(),
                'model_scope':v['model_scope'],'physical_success':False,'full_chassis_safety_qualified':False}
            report_path.write_text(json.dumps(report,indent=2))
            artifacts=[]
            for role,name in [('request','request.json'),('result','result.json'),('validation','cache_validation.json'),('independent_validation','e0_independent_validation.json')]:
                path=job/name;artifacts.append({'role':role,'path':str(path.resolve().relative_to(a.output.parent.resolve())),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            transitions.append({'node':'root','next_node':None,'episode_id':group,'stage':'PREPLACE_PLAN',
                'action_id':r['candidate_id'],'downstream_success':True,'execution_cost':v['joint_path_rad'],
                'evidence_level':'CUROBO_INDEPENDENTLY_VALIDATED','planner_success':True,'executed':False,'failure_codes':[],
                'metadata':{'independent_validation':flags,'source_artifacts':artifacts,
                    'model_scope':v['model_scope'],'full_chassis_safety_qualified':False,
                    'successor_effect_target':[1.,v['tcp_path_m'],v['joint_path_rad'],v['world']['min_clearance_m'],
                        v['minimum_joint_margin_rad'],v['orientation']['max_orientation_error_deg']/3.]}})
        obs={'episode_id':group,'stage':'PREPLACE_PLAN','robot_state':[],
            'remaining_goal':example['candidate_features']['goal_body_m'],
            'successor_constraints':example['tray_provenance']['observed_world_aabb_m']+[.84,.06,3./180.],
            'actions':actions,'scene_revision':'sha256:'+digest({'group':group,'requests':[r['scene_digest'] for _,r,_ in rows]}),
            'history_digest':digest({'stage':'already held transport','group':group})}
        episodes.append({'scene_group':group,'split':example['split'],'initial_node':'root','observations':{'root':obs},'transitions':transitions})
    dataset={'fixture':False,'scope':'ONE_STEP_PLANNING_COST_ONLY_CONDITIONAL_ON_ALL_OPTIONS_QUALIFIED',
        'physical_simulation':False,'full_chassis_safety_qualified':False,'reward':'1 - measured joint trajectory length (rad)',
        'cost_units':'joint path rad, not seconds or physical execution cost',
        'dimensions':{'state_dim':9,'action_dim':14,'max_actions':2,'cost_weight':1.},
        'episodes':episodes,'excluded_groups':excluded,'candidate_id_used_as_feature':False,
        'observation_stage':'PREPLACE only, successor goal is observed; no PICK or future-observation claim'}
    a.output.write_text(json.dumps(dataset,indent=2));print(json.dumps({'episodes':len(episodes),'excluded':excluded}))
if __name__=='__main__':main()
