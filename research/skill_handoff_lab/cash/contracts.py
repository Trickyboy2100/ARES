"""Revisioned contracts for CASH; no simulator or robot side effects."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json,math
from typing import Mapping

def digest(value):
    return "sha256:"+hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(",",":"),default=str).encode()).hexdigest()

CASH_REVISION="cash-smdp-v0-20261013"
STAGES=("GRASP_BRANCH","CENTER_TERMINAL","PREPLACE_PLAN")
EXECUTION_EVIDENCE=("CUROBO_ISAAC_EXECUTED","REAL_ROBOT_EXECUTED")
PLANNING_EVIDENCE="CUROBO_INDEPENDENTLY_VALIDATED"
PLANNING_ATTEMPT_EVIDENCE="CUROBO_BUDGETED_ATTEMPT"

def validate_transition_evidence(value, allow_planning=False):
    if not math.isfinite(value.execution_cost) or value.execution_cost<0:
        raise ValueError("transition cost must be finite and nonnegative")
    if value.evidence_level in EXECUTION_EVIDENCE and value.executed:
        return
    if allow_planning and value.evidence_level==PLANNING_ATTEMPT_EVIDENCE:
        if value.executed or value.downstream_success is True:
            raise ValueError('an unsuccessful/unknown planning attempt cannot certify motion or success')
        meta=value.metadata
        if meta.get('budget_before',0)-meta.get('budget_after',0)!=1:
            raise ValueError('attempt must consume one recorded planner invocation')
        if meta.get('attempt_status') not in ('PLANNER_FAILURE','QUALIFICATION_REJECT','TIMEOUT','SOFTWARE_ERROR','UNKNOWN'):
            raise ValueError('explicit attempt classification required')
        if meta.get('state_unchanged') is not True or not meta.get('source_artifacts'):
            raise ValueError('failed attempt must preserve actual state and hashed lineage')
        if value.downstream_success is False and meta.get('terminal_reason')!='BUDGET_EXHAUSTED':
            raise ValueError('budget failure must not claim geometric infeasibility')
        return
    if allow_planning and value.evidence_level == PLANNING_EVIDENCE:
        required=("collision", "self_collision", "attached_object", "soft_limits", "orientation", "scene_binding")
        checks=value.metadata.get("independent_validation", {})
        if value.downstream_success is False and not value.planner_success:
            if value.metadata.get("failure_class") != "VERIFIED_GEOMETRIC_INFEASIBILITY":
                raise ValueError("solver failure/timeout/OOM cannot be a negative task label")
        elif not all(type(checks.get(key)) is bool for key in required):
            raise ValueError("planning reward requires completed independent trajectory validators")
        if value.downstream_success is True and not all(checks.get(key) is True for key in required):
            raise ValueError("successful reward requires every independent validator to pass")
        if not value.metadata.get("source_artifacts"):
            raise ValueError("planning reward requires hashed request/result artifacts")
        if value.downstream_success is True and not value.planner_success:
            raise ValueError("successful planning reward requires planner success")
        return
    raise RuntimeError("CASH reward requires qualified execution or validated planning evidence")

@dataclass(frozen=True)
class HandoffAction:
    action_id:str; stage:str; features:tuple[float,...]; legal:bool
    legality_reasons:tuple[str,...]=()

@dataclass(frozen=True)
class HandoffObservation:
    episode_id:str; stage:str; robot_state:tuple[float,...]
    remaining_goal:tuple[float,...]; successor_constraints:tuple[float,...]
    actions:tuple[HandoffAction,...]; scene_revision:str; history_digest:str

@dataclass(frozen=True)
class ExecutedTransition:
    episode_id:str; stage:str; action_id:str
    next_observation:HandoffObservation|None; downstream_success:bool|None
    execution_cost:float; evidence_level:str; planner_success:bool
    executed:bool; failure_codes:tuple[str,...]; metadata:Mapping
    def to_dict(self):
        value=asdict(self); value["record_digest"]=digest(value); return value

@dataclass(frozen=True)
class EpisodeResult:
    episode_id:str; success:bool; total_cost:float; decisions:tuple[str,...]
    evidence_level:str; failure_codes:tuple[str,...]

def validate_observation(value:HandoffObservation)->None:
    if value.stage not in STAGES: raise ValueError("unknown CASH stage")
    if not value.scene_revision.startswith("sha256:"): raise ValueError("scene revision required")
    ids=[x.action_id for x in value.actions]
    if len(ids)!=len(set(ids)): raise ValueError("duplicate action ids")
    if any(a.stage!=value.stage for a in value.actions): raise ValueError("action stage mismatch")
    numeric=value.robot_state+value.remaining_goal+value.successor_constraints
    if not all(math.isfinite(x) for x in numeric): raise ValueError("nonfinite observation")
    if any(not all(math.isfinite(x) for x in a.features) for a in value.actions):
        raise ValueError("nonfinite action features")
