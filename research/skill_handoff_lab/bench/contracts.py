"""Revisioned, JSON-safe contracts for the CPU skill-handoff smoke."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib, json
from typing import Any

CONTRACT_REVISION = "skill-handoff-bench-v1-20261013"

def digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()

class Contract:
    def to_dict(self) -> dict:
        value = asdict(self); value["record_digest"] = digest(value); return value

@dataclass(frozen=True)
class Scenario(Contract):
    revision: str; scenario_id: str; seed: int; family: str; frame: str; arm: str
    center_target_body_m: tuple[float, float, float]
    successor_target_body_m: tuple[float, float, float]
    successor_yaw_rad: float; obstacles: tuple[dict, ...]; source_contracts: dict
    evidence_level: str = "CPU_GEOMETRY_ONLY"
    def __post_init__(self):
        if self.revision != CONTRACT_REVISION: raise ValueError("unknown scenario contract revision")
        if self.family not in {"TRAY_TO_GROOVE", "TRAY_TO_CABINET_PRELAND"}: raise ValueError("unknown scenario family")
        if self.frame != "BODY" or self.arm != "right": raise ValueError("v1 bench is BODY/right-arm only")

@dataclass(frozen=True)
class Candidate(Contract):
    revision: str; scenario_id: str; candidate_id: str; candidate_index: int
    center_target_body_m: tuple[float, float, float]; center_yaw_rad: float; roll_branch: int
    ik_attempted: bool; ik_solution_found: bool; center_joints_rad: tuple[float, ...] = ()
    fk_position_error_m: float | None = None; fk_orientation_error_deg: float | None = None
    soft_limit_margin_rad: float | None = None; endpoint_collision_free: bool = False
    endpoint_orientation_pass: bool = False; candidate_legal: bool = False
    failure_codes: tuple[str, ...] = ()

@dataclass(frozen=True)
class Transition(Contract):
    revision: str; scenario_id: str; candidate_id: str; transition_id: str; successor: str
    start_checks: dict; successor_ik_attempted: bool; successor_joints_rad: tuple[float, ...] = ()
    dense_joint_samples: int = 0; collision_free: bool = False
    endpoint_orientation_pass: bool = False; failure_codes: tuple[str, ...] = ()

@dataclass(frozen=True)
class Outcome(Contract):
    revision: str; scenario_id: str; candidate_id: str; transition_id: str
    geometry_successor_start_feasible: bool; label_source: str
    edge_cost_baseline_score: float | None
    evidence: dict = field(default_factory=lambda: {"geometry": "TESTED_CPU_PINNED_MODEL",
        "curobo": "NOT_TESTED_CUDA_UNAVAILABLE", "physics_contact": "NOT_TESTED",
        "real_execution": "NOT_TESTED"})
    failure_codes: tuple[str, ...] = ()
