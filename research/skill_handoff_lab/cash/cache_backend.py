"""Load immutable branching transition trees backed by real planner artifacts.

Each split is selected explicitly. Future place observations appear only after
the transition to PREPLACE_PLAN; observations are recorded rather than built
from future success labels. Artificial fixtures are rejected for research runs.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .contracts import HandoffAction,HandoffObservation,ExecutedTransition,validate_transition_evidence

def observation(row):
    value=dict(row)
    for name in ("robot_state","remaining_goal","successor_constraints"):
        value[name]=tuple(value[name])
    value["actions"]=tuple(HandoffAction(**dict(a,features=tuple(a["features"]),
        legality_reasons=tuple(a.get("legality_reasons",())))) for a in value["actions"])
    return HandoffObservation(**value)

class CachedPlanningBackend:
    def __init__(self,path,split="train",allow_fixture=False):
        self.path=Path(path); self.root=self.path.parent
        data=json.loads(self.path.read_text())
        if data.get("fixture",False) and not allow_fixture:
            raise ValueError("artificial fixture cannot be used for research training")
        self.episodes=[x for x in data["episodes"] if x["split"]==split]
        if not self.episodes: raise ValueError("no episodes for requested split")
        groups={}
        for e in data["episodes"]:
            old=groups.setdefault(e["scene_group"],e["split"])
            if old!=e["split"]: raise ValueError("scene-group leakage between train and test")
        for e in self.episodes:
            for row in e["transitions"]:
                roles=set()
                for artifact in row["metadata"].get("source_artifacts",[]):
                    roles.add(artifact.get("role"))
                    target=(self.root/artifact["path"]).resolve()
                    if not target.is_relative_to(self.root.resolve()):
                        raise ValueError("artifact outside dataset root")
                    if hashlib.sha256(target.read_bytes()).hexdigest()!=artifact["sha256"]:
                        raise ValueError("source artifact hash mismatch")
                    if artifact.get("role")=="validation":
                        report=json.loads(target.read_text())
                        if report["independent_validation"]!=row["metadata"].get("independent_validation"):
                            raise ValueError("cache validators differ from source validation report")
                if not {"request","result","validation"}<=roles:
                    raise ValueError("cache requires request, result and validation artifact roles")
    def reset(self,seed):
        self.episode=self.episodes[int(np.random.default_rng(seed).integers(len(self.episodes)))]
        self.node=self.episode["initial_node"]
        return observation(self.episode["observations"][self.node])
    def execute(self,obs,action_id):
        rows=[r for r in self.episode["transitions"] if r["node"]==self.node and r["action_id"]==action_id]
        if len(rows)!=1: raise ValueError("cache needs exactly one bound transition per legal action")
        row=dict(rows[0]); row.pop("node"); next_node=row.pop("next_node",None)
        row["next_observation"]=observation(self.episode["observations"][next_node]) if next_node else None
        row["failure_codes"]=tuple(row["failure_codes"])
        result=ExecutedTransition(**row)
        validate_transition_evidence(result,allow_planning=True)
        self.node=next_node
        return result
