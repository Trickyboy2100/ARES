"""Execution-blocked adapter from P2.1 records to persistent-cuRobo jobs."""
from __future__ import annotations
from dataclasses import asdict,dataclass
from .contracts import digest
from .p2_1_contracts import P2_1_REVISION

ORACLE_REVISION="skill-handoff-curobo-oracle-adapter-v2-20261013"

@dataclass(frozen=True)
class OracleJob:
    revision:str; job_id:str; scenario_id:str; candidate_id:str
    source_scene_revision:str; attached_object_revision:str; segment:str
    start_joints_rad:tuple; goal_joints_rad:tuple; orientation_mode:str
    planning_only:bool=True; execution_allowed:bool=False
    required_validators:tuple=("PERSISTENT_CUROBO","ATTACHED_OBJECT_COLLISION",
        "INDEPENDENT_DENSE_COLLISION","SITE_SOFT_LIMITS","LEVEL_ORIENTATION")
    def to_dict(self):
        value=asdict(self); value["job_digest"]=digest(value); return value

def build_jobs(scene,candidate,transition):
    if scene.revision!=P2_1_REVISION or candidate.revision!=P2_1_REVISION:
        raise ValueError("P2.1 records required")
    if candidate.scene_revision!=scene.scene_revision or transition.scene_revision!=scene.scene_revision:
        raise ValueError("stale/mismatched SceneSnapshot binding")
    if candidate.attached_object_revision!=scene.object_contract["revision"]:
        raise ValueError("attached-object revision mismatch")
    if not candidate.candidate_legal or not transition.preplace_joints_rad:
        return []
    # Canonical current runtime contract: direct move_preplace after the fresh
    # place scene. The TransferChainPlanner rear waypoint is not substituted.
    rows=[("center_to_direct_preplace",candidate.center_joints_rad,
           transition.preplace_joints_rad,"LEVEL_YAW_TARGET")]
    return [OracleJob(ORACLE_REVISION,
        "%s:%s"%(candidate.candidate_id,name),scene.scenario_id,candidate.candidate_id,
        scene.scene_revision,scene.object_contract["revision"],name,tuple(start),tuple(goal),mode)
        for name,start,goal,mode in rows]

def qualification_status():
    return {"oracle_revision":ORACLE_REVISION,"adapter":"READY_FOR_REQUEST_EXPORT",
            "gpu_execution":"BLOCKED_NVML_DRIVER_LIBRARY_MISMATCH",
            "curobo_validated_this_round":False,"execution_allowed":False}
