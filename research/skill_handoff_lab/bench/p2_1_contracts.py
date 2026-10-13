"""P2.1 revisioned contracts with real handoff-state lineage."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from .contracts import digest

P2_1_REVISION="skill-handoff-mechanism-v2-20261013"

class Record:
    def to_dict(self):
        value=asdict(self); value["record_digest"]=digest(value); return value

@dataclass(frozen=True)
class ScenarioV2(Record):
    revision:str; scenario_id:str; seed:int; family:str; population:str
    center_target_body_m:tuple; predecessor_target_body_m:tuple
    rear_preplace_body_m:tuple; preplace_body_m:tuple; entrance_yaw_rad:float
    obstacles:tuple; object_contract:dict; source_contracts:dict; scene_revision:str
    def __post_init__(self):
        if self.revision!=P2_1_REVISION: raise ValueError("unknown P2.1 revision")
        core={"scenario_id":self.scenario_id,"seed":self.seed,"family":self.family,
              "obstacles":self.obstacles,"rear":self.rear_preplace_body_m,
              "preplace":self.preplace_body_m,"object":self.object_contract}
        if self.scene_revision!=digest(core): raise ValueError("scene revision mismatch")

@dataclass(frozen=True)
class CandidateV2(Record):
    revision:str; scenario_id:str; scene_revision:str; candidate_id:str
    grasp_symmetry:str; roll_branch:int; center_yaw_rad:float; ik_branch_index:int
    predecessor_joints_rad:tuple; center_joints_rad:tuple
    center_fk_position_error_m:float|None; center_fk_orientation_error_deg:float|None
    center_limit_margin_rad:float|None; center_singularity:float|None
    attached_object_revision:str; candidate_state_revision:str
    candidate_legal:bool; failure_codes:tuple

@dataclass(frozen=True)
class TransitionV2(Record):
    revision:str; scenario_id:str; candidate_id:str; scene_revision:str
    start_checks:dict; rear_joints_rad:tuple; preplace_joints_rad:tuple
    rear_limit_margin_rad:float|None; preplace_limit_margin_rad:float|None
    rear_singularity:float|None; preplace_singularity:float|None
    center_to_rear_proxy:dict; rear_to_preplace_proxy:dict
    endpoint_failure_codes:tuple; geometric_start_set_member:bool

@dataclass(frozen=True)
class OutcomeV2(Record):
    revision:str; scenario_id:str; candidate_id:str
    geometric_start_set_member:bool; cpu_linear_proxy_clear:bool
    baseline_predecessor_to_center_cost:float|None
    full_chain_proxy_cost:float|None; evidence:dict; failure_codes:tuple
