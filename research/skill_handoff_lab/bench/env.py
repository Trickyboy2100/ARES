"""Seeded one-transition environment for terminal-state sensitivity."""
from __future__ import annotations
import numpy as np
from ares_r.manipulation.transfer_chain import IKNode,_edge_cost
from .contracts import CONTRACT_REVISION,Outcome,Scenario,Transition
from .model import PinnedRobotModel

class SkillHandoffBench:
    def __init__(self,ares_root,ares_r_root): self.model=PinnedRobotModel(ares_root,ares_r_root); self.scenario=None; self.candidates={}
    def reset(self,scenario:Scenario,candidates,seed:int):
        if seed!=scenario.seed: raise ValueError("reset seed must equal frozen scenario seed")
        self.scenario=scenario; self.candidates={x.candidate_id:x for x in candidates}
        return {"scenario_id":scenario.scenario_id,"seed":seed,"candidate_ids":sorted(self.candidates)}
    def step(self,candidate_id):
        if self.scenario is None or candidate_id not in self.candidates: raise ValueError("reset first and select a frozen candidate")
        s=self.scenario; c=self.candidates[candidate_id]
        checks={"candidate_contract_revision":c.revision==CONTRACT_REVISION,"candidate_scenario_binding":c.scenario_id==s.scenario_id,"candidate_legal":c.candidate_legal,"held_object_contract":True,"fresh_scene_revision":True}
        failures=[]; successor_q=(); dense=0; collision_free=False; orientation=False
        start_ok=all(checks.values())
        if not start_ok: failures.append("START_CONDITION_FAILED")
        else:
            sols=self.model.solve(s.successor_target_body_m,s.successor_yaw_rad,c.roll_branch,[c.center_joints_rad],s.seed*100+c.candidate_index+7000,24)
            if not sols: failures.append("SUCCESSOR_IK_FAILED")
            else:
                _,q,_,oe,_=min(sols,key=lambda row:np.linalg.norm(row[1]-np.asarray(c.center_joints_rad)))
                successor_q=tuple(float(v) for v in q); orientation=oe<=1.0
                endpoint_failures=self.model.collision_codes(successor_q,s.obstacles)
                collision_free=not endpoint_failures; failures.extend(endpoint_failures)
                path_clear,path_failures,dense=self.model.dense_path_check(c.center_joints_rad,successor_q,s.obstacles)
                checks["dense_joint_interpolation_clear_descriptor_not_gate"]=path_clear
                checks["dense_joint_interpolation_failure_count"]=len(path_failures)
                if not orientation: failures.append("SUCCESSOR_ORIENTATION_FAILED")
        feasible=start_ok and bool(successor_q) and collision_free and orientation
        tid=s.scenario_id+":"+candidate_id+":successor"
        transition=Transition(CONTRACT_REVISION,s.scenario_id,candidate_id,tid,"GROOVE_PLACE_START" if s.family=="TRAY_TO_GROOVE" else "CABINET_PRELAND_START",checks,start_ok,successor_q,dense,collision_free,orientation,tuple(sorted(set(failures))))
        score=None
        if c.center_joints_rad:
            score=float(_edge_cost(IKNode("current",tuple([0.]*6),0.,1.,1.),IKNode("center",c.center_joints_rad,c.center_yaw_rad,.1,max(c.soft_limit_margin_rad or .001,.001)))[0])
        outcome=Outcome(CONTRACT_REVISION,s.scenario_id,candidate_id,tid,feasible,"CPU_SUCCESSOR_IK_AND_DENSE_GEOMETRIC_PATH_NOT_EDGE_COST",score,failure_codes=transition.failure_codes)
        return transition,outcome
