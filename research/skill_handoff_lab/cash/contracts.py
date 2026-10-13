"""Revisioned contracts for CASH; no simulator or robot side effects."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
from typing import Mapping

def digest(value):
    return "sha256:"+hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(",",":"),default=str).encode()).hexdigest()

CASH_REVISION="cash-smdp-v0-20261013"
STAGES=("GRASP_BRANCH","CENTER_TERMINAL","PREPLACE_PLAN")
EXECUTION_EVIDENCE=("CUROBO_ISAAC_EXECUTED","REAL_ROBOT_EXECUTED")

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
