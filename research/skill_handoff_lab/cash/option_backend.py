"""Causal two-stage, fingerprinted REAL-planning SMDP backend (not PhysX online).

Cache outcomes are inspected only by execute(), never by actor features/masks.
Suffix starts are exact preceding planner terminals; unknown attempts consume
budget without moving state or claiming geometric infeasibility.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .contracts import HandoffAction,HandoffObservation,ExecutedTransition,PLANNING_EVIDENCE,PLANNING_ATTEMPT_EVIDENCE


class PlanningOptionBackend:
    def __init__(self,path,split='train',permute=True):
        self.path=Path(path);self.root=self.path.parent;data=json.loads(self.path.read_text())
        if data.get('fixture'):raise ValueError('fixtures cannot train research policies')
        if data['evidence_level']!='REAL_CUROBO_PLANNING_CACHE':raise ValueError('wrong evidence level')
        groups={}
        for task in data['tasks']:
            if groups.setdefault(task['scene_group'],task['split'])!=task['split']:raise ValueError('group leakage')
        self.tasks=[t for t in data['tasks'] if t['split']==split]
        if not self.tasks:raise ValueError('split has no complete causal task')
        self.edges=data['edges'];self.permute=permute;self.logs=[];self.episode_index=0
        # Verify sources once per environment, including every result and qualification.
        for edge in self.edges.values():
            sources={}
            for artifact in edge['artifacts']:
                p=(self.root/artifact['path']).resolve()
                if not p.is_relative_to(self.root.resolve()):raise ValueError('unbound source path')
                if hashlib.sha256(p.read_bytes()).hexdigest()!=artifact['sha256']:raise ValueError('source hash mismatch')
                sources[artifact['role']]=p
            request=json.loads(sources['request'].read_text());result=json.loads(sources['result'].read_text())
            validation=json.loads(sources['validation'].read_text())
            if request['scene_digest']!=edge['scene_digest'] or request['start_rad']!=edge['start_q']:
                raise ValueError('request state/scene fingerprint differs from cache')
            if validation['independent_validation']!=edge['independent_validation']:raise ValueError('cached validation mismatch')
            if edge['qualified'] and (not validation['qualified'] or result.get('observed_result')!='SUCCESS'
                    or result['trajectory_points_rad'][-1]!=edge['end_q']):
                raise ValueError('cache claims an unobserved successful transition')
    def reset(self,seed):
        self.rng=np.random.default_rng(seed);self.task=self.tasks[int(self.rng.integers(len(self.tasks)))]
        self.episode_index+=1;self.episode_id=f'{self.task["task_id"]}:{self.episode_index}'
        self.node='CENTER';self.q=list(self.task['initial_q']);self.budget=4;self.history=[]
        return self._observation()
    def _observation(self):
        stage='CENTER_TERMINAL' if self.node=='CENTER' else 'PREPLACE_PLAN'
        rows=self.task['actions'][self.node];order=self.rng.permutation(len(rows)) if self.permute else np.arange(len(rows))
        actions=[]
        for i in order:
            row=rows[int(i)]
            # Features are declared action/IK geometry/profile only. No success,
            # path cost or cached result is exposed before choosing an action.
            actions.append(HandoffAction(row['edge_id'],stage,tuple(row['features']),bool(row['endpoint_legal']),('predecision_endpoint_limits',)))
        return HandoffObservation(self.episode_id,stage,tuple(self.q),
            tuple(self.task['goal_xyz']+[self.budget/4.,float(self.node!='CENTER')]),
            tuple(self.task['observed_context']),tuple(actions),'sha256:'+self.task['scene_digest'],
            'sha256:'+hashlib.sha256(json.dumps(self.history).encode()).hexdigest())
    def execute(self,observation,action_id):
        edge=self.edges[action_id]
        if edge['from_node']!=self.node or edge['scene_digest']!=self.task['scene_digest']:raise ValueError('scene/node mismatch')
        if not np.array_equal(np.asarray(edge['start_q']),np.asarray(self.q)):
            raise ValueError('cache refuses differing actual start; live planning required')
        before=self.budget;self.budget-=1;self.history.append(action_id)
        qualified=bool(edge['qualified']);old_q=list(self.q)
        success=False;attempt_status=None
        if qualified:
            self.q=list(edge['end_q']);self.node=edge['to_node'];success=self.node=='PRELAND'
        else:attempt_status=edge['attempt_status']
        metadata={'source_artifacts':edge['artifacts'],'independent_validation':edge['independent_validation'],
            'budget_before':before,'budget_after':self.budget,'planner_calls':1,'attempt_cap':edge['attempt_cap'],
            'cache_lookup':True,'new_online_planner_calls':0,
            'planning_wall_s':edge['planning_wall_s'],'planned_execution_s':edge['planned_duration_s'],
            'option_duration_s':edge['planning_wall_s']+(edge['planned_duration_s'] if qualified else 0),
            'evidence':'planning-only cache, NOT PhysX online','state_before':old_q,'state_after':list(self.q),
            'actual_start_request_fingerprint':edge['request_sha256'],'state_unchanged':self.q==old_q,
            'attempt_status':attempt_status,'terminal_reason':'SUCCESS' if success else ('BUDGET_EXHAUSTED' if self.budget==0 else None)}
        cost=metadata['option_duration_s']+2. # explicit per-call overhead cost, not a success label
        terminal=success or self.budget==0
        if not qualified:
            level=PLANNING_ATTEMPT_EVIDENCE
        else:level=PLANNING_EVIDENCE
        # Qualified Prepare on the last available call still fails the finite-
        # budget task, not geometry. Preserve completed validator evidence.
        if terminal and not success and qualified:
            metadata['terminal_reason']='BUDGET_EXHAUSTED'
        transition=ExecutedTransition(self.episode_id,observation.stage,action_id,
            None if terminal else self._observation(),success if terminal else None,float(cost),level,
            edge['planner_success'],False,() if qualified else (attempt_status,),metadata)
        self.logs.append({'episode':self.episode_id,'task_id':self.task['task_id'],'action_id':action_id,
                          'from_node':edge['from_node'],'to_node':self.node,**metadata,'success':success})
        return transition
