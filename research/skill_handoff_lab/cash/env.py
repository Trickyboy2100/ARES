"""Dependency-light Gymnasium-shaped SMDP environment.

The backend alone supplies outcomes. The environment never converts a hand
score, endpoint IK result, or timeout into a success label.
"""
from __future__ import annotations
from typing import Protocol
import numpy as np
from .contracts import (EXECUTION_EVIDENCE,EpisodeResult,ExecutedTransition,
                        HandoffObservation,validate_observation,validate_transition_evidence)

class HandoffBackend(Protocol):
    def reset(self,seed:int)->HandoffObservation: ...
    def execute(self,observation:HandoffObservation,action_id:str)->ExecutedTransition: ...

class SkillHandoffEnv:
    metadata={"render_modes":[]}
    def __init__(self,backend:HandoffBackend,strict_execution:bool=True,allow_planning:bool=False,cost_weight:float=.001):
        self.backend=backend; self.strict_execution=strict_execution
        self.allow_planning=allow_planning
        if not np.isfinite(cost_weight) or cost_weight<0: raise ValueError('invalid cost weight')
        self.cost_weight=float(cost_weight)
        self.current=None; self.total_cost=0.; self.decisions=[]; self.failures=[]
    @staticmethod
    def encode(observation:HandoffObservation):
        validate_observation(observation)
        state=np.asarray(observation.robot_state+observation.remaining_goal+
                         observation.successor_constraints,dtype=np.float32)
        action=np.asarray([x.features for x in observation.actions],dtype=np.float32)
        mask=np.asarray([x.legal for x in observation.actions],dtype=bool)
        return {"state":state,"action_features":action,"action_mask":mask,
                "stage":observation.stage,"action_ids":tuple(x.action_id for x in observation.actions)}
    def reset(self,seed=None,options=None):
        self.current=self.backend.reset(int(seed or 0)); validate_observation(self.current)
        self.total_cost=0.; self.decisions=[]; self.failures=[]
        return self.encode(self.current),{"episode_id":self.current.episode_id}
    def action_masks(self):
        if self.current is None: raise RuntimeError("reset first")
        return np.asarray([x.legal for x in self.current.actions],dtype=bool)
    def step(self,action:int):
        if self.current is None: raise RuntimeError("reset first")
        if not (0<=action<len(self.current.actions)): raise ValueError("action index out of range")
        selected=self.current.actions[action]
        if not selected.legal: raise ValueError("masked/illegal action selected")
        transition=self.backend.execute(self.current,selected.action_id)
        if (transition.episode_id!=self.current.episode_id or
                transition.action_id!=selected.action_id or transition.stage!=self.current.stage):
            raise ValueError("backend transition lineage mismatch")
        if self.strict_execution:
            validate_transition_evidence(transition,self.allow_planning)
        self.decisions.append(selected.action_id); self.total_cost+=float(transition.execution_cost)
        self.failures.extend(transition.failure_codes)
        terminal=transition.downstream_success is not None
        reward=(float(bool(transition.downstream_success)) if terminal else 0.)
        reward-=self.cost_weight*float(transition.execution_cost)
        if terminal:
            obs={"state":np.zeros(0,dtype=np.float32),"action_features":np.zeros((0,0),dtype=np.float32),
                 "action_mask":np.zeros(0,dtype=bool),"stage":"TERMINAL","action_ids":()}
            info={"result":EpisodeResult(self.current.episode_id,bool(transition.downstream_success),
                self.total_cost,tuple(self.decisions),transition.evidence_level,tuple(self.failures)),
                "transition":transition}
            self.current=None
            # Remaining planning budget is part of the task state. Exhausting
            # it is an MDP terminal (not geometric impossibility), not an
            # external time-limit truncation. SB3 would otherwise bootstrap a
            # value from the zero-packed terminal observation.
            return obs,reward,True,False,info
        if transition.next_observation is None: raise ValueError("nonterminal transition lacks observation")
        if transition.next_observation.episode_id!=self.current.episode_id:
            raise ValueError("next observation episode mismatch")
        self.current=transition.next_observation; validate_observation(self.current)
        return self.encode(self.current),reward,False,False,{"transition":transition}
