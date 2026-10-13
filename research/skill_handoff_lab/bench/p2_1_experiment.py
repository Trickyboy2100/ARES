"""Bounded P2.1 handoff-mechanism experiment; CPU geometry evidence only."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from ares_r.manipulation.transfer_chain import IKNode,_edge_cost
from .contracts import digest
from .model import PinnedRobotModel
from .p2_1_contracts import (P2_1_REVISION,CandidateV2,OutcomeV2,ScenarioV2,
                             TransitionV2)

READY_Q=(-0.76582,-0.58930,1.55981,-2.52641,-1.56853,1.04180)
SYMMETRIES=(("NORMAL",0),("FLIPPED_180",1))

def _object_contract(family):
    dimensions=([.22,.12,.035] if family=="TRAY_TO_GROOVE" else [.18,.11,.04])
    core={"object_id":"held_tray","dimensions_m":dimensions,
          "tcp_offset_m":[0.,0.,.10],"geometry":"CUBOID_RESEARCH_PROXY",
          "physical_contact":"NOT_TESTED"}
    return dict(core,revision=digest(core))

def scenario(*,scenario_id,seed,family,population,center,rear,preplace,yaw,obstacles,source):
    obj=_object_contract(family)
    core={"scenario_id":scenario_id,"seed":seed,"family":family,
          "obstacles":tuple(obstacles),"rear":tuple(rear),"preplace":tuple(preplace),
          "object":obj}
    return ScenarioV2(P2_1_REVISION,scenario_id,seed,family,population,tuple(center),
        (center[0]-.10,center[1],center[2]+.035),tuple(rear),tuple(preplace),yaw,
        tuple(obstacles),obj,source,digest(core))

def load_v4_scenes(run_root):
    rows=[json.loads(x) for x in (Path(run_root)/"scenarios.jsonl").read_text().splitlines()]
    result=[]
    for row in rows:
        pre=np.asarray(row["successor_target_body_m"],dtype=float)
        rear=pre+np.array([-.15,0.,0.])
        result.append(scenario(scenario_id=row["scenario_id"]+"-p21",seed=row["seed"],
            family=row["family"],population="FROZEN_V4_GEOMETRY_REPLAY",
            center=row["center_target_body_m"],rear=rear,preplace=pre,
            yaw=row["successor_yaw_rad"],obstacles=row["obstacles"],
            source={"source_run":str(run_root),"geometry_policy":"UNCHANGED_V4",
              "predecessor":"ARES-R named pose folded: commissioned held-payload terminal"}))
    return result

def bounded_mechanism_scenes():
    """Four predeclared aperture cases; no seed search and not site evidence."""
    rows=[]
    definitions=[
      (5101,"TRAY_TO_GROOVE",.155,.565,-.235,1.105,math.radians(45)),
      (5102,"TRAY_TO_GROOVE",.185,.575,-.245,1.115,math.radians(55)),
      (5201,"TRAY_TO_CABINET_PRELAND",.205,.600,-.185,1.220,math.radians(2.5)),
      (5202,"TRAY_TO_CABINET_PRELAND",.245,.615,-.200,1.235,math.radians(7.5)),
    ]
    for seed,family,aperture,x,y,z,yaw in definitions:
        center=(.358,-.082,1.030); pre=(x,y,z); rear=(x-.15,y,z)
        wall=.025; depth=.055; height=.25
        obstacles=[{"id":"aperture/left","kind":"bounded_mechanism_wall",
                    "center_body_m":[x+depth/2,y+aperture/2+wall/2,z],
                    "dims_m":[depth,wall,height]},
                   {"id":"aperture/right","kind":"bounded_mechanism_wall",
                    "center_body_m":[x+depth/2,y-aperture/2-wall/2,z],
                    "dims_m":[depth,wall,height]}]
        rows.append(scenario(scenario_id="mechanism-%d"%seed,seed=seed,family=family,
            population="PREDECLARED_SYNTHETIC_APERTURE_MECHANISM",center=center,rear=rear,
            preplace=pre,yaw=yaw,obstacles=obstacles,
            source={"hypothesis":"grasp/yaw/IK terminal state changes entrance branch and swept geometry",
                    "aperture_m":aperture,"site_scene_claim":False,
                    "predecessor":"ARES-R named pose folded: commissioned held-payload terminal"}))
    return rows

def _node(model,stage,q,yaw,pe=0.,oe=0.,margin=None):
    margin=float(np.min(np.r_[np.asarray(q)-model.lower,model.upper-np.asarray(q)])) if margin is None else margin
    return IKNode(stage,tuple(float(v) for v in q),yaw,model.singularity(q),margin,pe,oe)

def _edge(left,right): return float(_edge_cost(left,right)[0])

def candidates(model,scene):
    # Fixed factorial support: 2 grasp symmetries x 3 meaningful center yaws x
    # at most 2 IK branches. Infeasible cells are retained, never replenished.
    yaw_offsets=(-30.,0.,30.); result=[]
    for symmetry,roll in SYMMETRIES:
        for yaw_index,offset in enumerate(yaw_offsets):
            yaw=scene.entrance_yaw_rad+math.radians(offset)
            # The prior P1/P2 baseline used an unjustified zero vector.  P2.1
            # uses the recorded, commissioned held-payload `folded` terminal
            # state as the explicit predecessor for every candidate.
            pred_q=np.asarray(model.commissioned_predecessor_q,dtype=float)
            sols=model.solve(scene.center_target_body_m,yaw,roll,[pred_q,READY_Q],
                             scene.seed*1000+roll*100+yaw_index+50,48)
            for branch in range(2):
                if branch>=len(sols):
                    result.append(_failed_candidate(scene,symmetry,roll,yaw,branch,
                                                     "CENTER_IK_BRANCH_NOT_FOUND",pred_q))
                    continue
                _,q,pe,oe,margin=sols[branch]
                codes,_=model.context_collision_codes(q,scene.obstacles,scene.object_contract)
                legal=not codes
                cid="%s:%s:y%d:b%d"%(scene.scenario_id,symmetry,yaw_index,branch)
                state_core={"scene_revision":scene.scene_revision,"q":list(map(float,q)),
                            "object_revision":scene.object_contract["revision"],
                            "symmetry":symmetry,"yaw":yaw}
                result.append(CandidateV2(P2_1_REVISION,scene.scenario_id,
                    scene.scene_revision,cid,symmetry,roll,yaw,branch,
                    tuple(map(float,pred_q)),tuple(map(float,q)),pe,oe,margin,
                    model.singularity(q),scene.object_contract["revision"],digest(state_core),
                    legal,tuple(codes)))
    return result

def _failed_candidate(scene,symmetry,roll,yaw,branch,code,pred_q=()):
    cid="%s:%s:y%+.6f:b%d"%(scene.scenario_id,symmetry,yaw,branch)
    state={"scene_revision":scene.scene_revision,"failure":code,"id":cid}
    return CandidateV2(P2_1_REVISION,scene.scenario_id,scene.scene_revision,cid,
        symmetry,roll,yaw,branch,tuple(map(float,pred_q)),(),None,None,
        None,None,scene.object_contract["revision"],digest(state),False,(code,))

def evaluate(model,scene,candidate):
    checks={"candidate_legal":candidate.candidate_legal,
      "candidate_scene_revision_matches":candidate.scene_revision==scene.scene_revision,
      "attached_object_revision_matches":candidate.attached_object_revision==scene.object_contract["revision"],
      "candidate_state_revision_present":candidate.candidate_state_revision.startswith("sha256:")}
    failures=list(candidate.failure_codes); rear_q=(); pre_q=(); rear_margin=pre_margin=None
    rear_sing=pre_sing=None; proxy1={"clear":False,"status":"NOT_RUN"}
    proxy2={"clear":False,"status":"NOT_APPLICABLE_REAR_PREPLACE_NOT_RUNTIME_GATE"}
    if all(checks.values()):
        # The commissioned runtime package enters the next skill through one
        # direct move_preplace.  A research-only rear_preplace waypoint exists
        # in TransferChainPlanner, but is not a qualified runtime gate here.
        pre=model.solve(scene.preplace_body_m,scene.entrance_yaw_rad,
                        candidate.roll_branch,[candidate.center_joints_rad],
                        scene.seed*10000+candidate.ik_branch_index+71,48)
        if not pre: failures.append("PREPLACE_IK_FAILED")
        else:
            chosen=min(pre,key=lambda x:np.linalg.norm(x[1]-np.asarray(candidate.center_joints_rad)))
            _,pq,_,_,pre_margin=chosen; pre_q=tuple(map(float,pq)); pre_sing=model.singularity(pq)
            codes,_=model.context_collision_codes(pq,scene.obstacles,scene.object_contract)
            failures.extend("PREPLACE_"+x for x in codes)
            proxy1=model.dense_context_path_check(candidate.center_joints_rad,pq,
                                                   scene.obstacles,scene.object_contract)
    endpoint_failures=tuple(sorted(set(x for x in failures if not x.startswith("PATH_"))))
    member=all(checks.values()) and bool(pre_q) and not endpoint_failures
    transition=TransitionV2(P2_1_REVISION,scene.scenario_id,candidate.candidate_id,
        scene.scene_revision,checks,rear_q,pre_q,rear_margin,pre_margin,rear_sing,pre_sing,
        proxy1,proxy2,endpoint_failures,member)
    baseline=full=None
    if candidate.predecessor_joints_rad and candidate.center_joints_rad:
        pred=_node(model,"predecessor",candidate.predecessor_joints_rad,candidate.center_yaw_rad)
        center=_node(model,"center",candidate.center_joints_rad,candidate.center_yaw_rad,
                     margin=candidate.center_limit_margin_rad)
        baseline=_edge(pred,center); full=baseline
        if pre_q:
            full+=_edge(center,_node(model,"preplace",pre_q,scene.entrance_yaw_rad,
                                    margin=pre_margin))
    outcome=OutcomeV2(P2_1_REVISION,scene.scenario_id,candidate.candidate_id,member,
        bool(proxy1.get("clear")),baseline,full,
        {"endpoint_geometry":"TESTED_CPU","linear_joint_proxy":"TESTED_NOT_A_PLANNER",
         "successor_contract":"CANONICAL_RUNTIME_DIRECT_MOVE_PREPLACE",
         "rear_preplace":"NOT_A_RUNTIME_GATE_NOT_QUALIFIED",
         "curobo":"NOT_TESTED_NVML_MISMATCH","physics_contact":"NOT_TESTED",
         "real_execution":"NOT_TESTED","context_coverage":model.context_coverage},endpoint_failures)
    return transition,outcome

def exact_v4_recheck(model,scenes,v4_root):
    """Re-evaluate the exact frozen 6x4 V4 states under the V2 context model.

    No candidate or successor IK is regenerated. The predecessor state was not
    recorded by V4 and therefore remains explicitly UNKNOWN.
    """
    root=Path(v4_root)
    old_candidates={row["candidate_id"]:row for row in
        (json.loads(x) for x in (root/"candidates.jsonl").read_text().splitlines())}
    old_transitions={row["candidate_id"]:row for row in
        (json.loads(x) for x in (root/"transitions.jsonl").read_text().splitlines())}
    scene_map={s.scenario_id.removesuffix("-p21"):s for s in scenes
               if s.population=="FROZEN_V4_GEOMETRY_REPLAY"}
    rows=[]
    for candidate_id,old in old_candidates.items():
        old_transition=old_transitions[candidate_id]; scene=scene_map[old["scenario_id"]]
        center_codes,_=model.context_collision_codes(old["center_joints_rad"],
            scene.obstacles,scene.object_contract)
        successor_q=old_transition.get("successor_joints_rad") or []
        successor_codes=[]
        if successor_q:
            successor_codes,_=model.context_collision_codes(successor_q,
                scene.obstacles,scene.object_contract)
        member=bool(old["candidate_legal"] and successor_q and
                    not center_codes and not successor_codes)
        rows.append({"scenario_id":old["scenario_id"],"candidate_id":candidate_id,
          "family":scene.family,"old_candidate_legal":old["candidate_legal"],
          "old_successor_feasible":bool(old_transition["collision_free"] and
                                         old_transition["endpoint_orientation_pass"]),
          "v2_center_context_clear":not center_codes,
          "v2_successor_context_clear":bool(successor_q and not successor_codes),
          "v2_geometric_start_set_member":member,
          "v2_center_failure_codes":center_codes,
          "v2_successor_failure_codes":successor_codes,
          "held_object_revision":scene.object_contract["revision"],
          "scene_revision":scene.scene_revision,
          "predecessor_state":"UNKNOWN_NOT_RECORDED_IN_V4",
          "curobo":"NOT_TESTED","physics_contact":"NOT_TESTED"})
    return rows

def run(ares_root,ares_r_root,v4_root,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    model=PinnedRobotModel(ares_root,ares_r_root)
    scenes=load_v4_scenes(v4_root)+bounded_mechanism_scenes()
    v4_recheck=exact_v4_recheck(model,scenes,v4_root)
    cc=[];tt=[];oo=[]
    for s in scenes:
        rows=candidates(model,s); cc.extend(rows)
        for c in rows:
            t,o=evaluate(model,s,c); tt.append(t); oo.append(o)
    for name,rows in (("scenarios",scenes),("candidates",cc),("transitions",tt),("outcomes",oo)):
        with (output/(name+".jsonl")).open("w") as stream:
            for row in rows: stream.write(json.dumps(row.to_dict(),sort_keys=True)+"\n")
    with (output/"v4_exact_6x4_recheck.jsonl").open("w") as stream:
        for row in v4_recheck: stream.write(json.dumps(row,sort_keys=True)+"\n")
    summary={"revision":P2_1_REVISION,"scenarios":len(scenes),"candidates":len(cc),
      "legal_candidates":sum(x.candidate_legal for x in cc),
      "geometric_start_members":sum(x.geometric_start_set_member for x in oo),
      "cpu_proxy_clear":sum(x.cpu_linear_proxy_clear for x in oo),"per_scene":{}}
    summary["v4_exact_recheck"]={"rows":len(v4_recheck),
      "old_successor_feasible":sum(x["old_successor_feasible"] for x in v4_recheck),
      "v2_geometric_start_set_member":sum(x["v2_geometric_start_set_member"] for x in v4_recheck)}
    for s in scenes:
        rows=[x for x in oo if x.scenario_id==s.scenario_id]
        summary["per_scene"][s.scenario_id]={"population":s.population,
          "candidates":len(rows),"legal":sum(x.candidate_legal for x in cc if x.scenario_id==s.scenario_id),
          "start_members":sum(x.geometric_start_set_member for x in rows),
          "proxy_clear":sum(x.cpu_linear_proxy_clear for x in rows),
          "endpoint_sensitive":len({x.geometric_start_set_member for x in rows})>1,
          "proxy_sensitive":len({x.cpu_linear_proxy_clear for x in rows})>1}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary
