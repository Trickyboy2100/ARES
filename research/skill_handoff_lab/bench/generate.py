"""Deterministic 6x4 CPU smoke generator; no CUDA, physics or robot control."""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
from .contracts import CONTRACT_REVISION,Candidate,Scenario
from .env import SkillHandoffBench

READY_Q=(-0.76582,-0.58930,1.55981,-2.52641,-1.56853,1.04180)
def _scene(seed,family):
    rng=np.random.default_rng(seed)
    if family=="TRAY_TO_GROOVE":
        center=(.358+rng.uniform(-.006,.006),-.080+rng.uniform(-.006,.006),1.028+rng.uniform(-.005,.005)); successor=(.545+rng.uniform(-.018,.018),-.255+rng.uniform(-.025,.025),1.105+rng.uniform(-.006,.006)); yaw=math.radians(45+rng.uniform(-4,4))
        obstacles=({"id":"groove/left_wall","kind":"groove_wall","center_body_m":[successor[0]+.025,successor[1]+.070,successor[2]-.015],"dims_m":[.05,.026,.22]},{"id":"groove/right_wall","kind":"groove_wall","center_body_m":[successor[0]+.025,successor[1]-.070,successor[2]-.015],"dims_m":[.05,.026,.22]})
    else:
        center=(.358+rng.uniform(-.006,.006),-.080+rng.uniform(-.006,.006),1.028+rng.uniform(-.005,.005)); successor=(.604866+rng.uniform(-.018,.018),-.184327+rng.uniform(-.018,.018),1.222342+rng.uniform(-.008,.008)); yaw=.044793+math.radians(rng.uniform(-4,4))
        obstacles=({"id":"cabinet/left_jamb","kind":"cabinet_jamb","center_body_m":[successor[0]+.025,successor[1]+.115,successor[2]-.055],"dims_m":[.10,.035,.34]},{"id":"cabinet/right_jamb","kind":"cabinet_jamb","center_body_m":[successor[0]+.025,successor[1]-.115,successor[2]-.055],"dims_m":[.10,.035,.34]},{"id":"cabinet/top","kind":"cabinet_lintel","center_body_m":[successor[0]+.025,successor[1],successor[2]+.125],"dims_m":[.10,.265,.035]})
    sid=("groove" if family=="TRAY_TO_GROOVE" else "cabinet")+"-%d"%seed
    return Scenario(CONTRACT_REVISION,sid,seed,family,"BODY","right",center,successor,yaw,obstacles,{"urdf":"pinned ARES jaka_minicobo_gripper.urdf","mount":"ARES-R BODY right mount + 2026-09-21 read-only FK correction","limits":"site soft limits +0.15 rad","tcp":"2026-09-21 audited tool2 transform; not current live TCP"})
def _candidates(bench,scene):
    # A single Cartesian center goal deliberately admits several joint-space
    # terminal states. This tests the handoff state, not four different goals.
    branch=0 if scene.family=="TRAY_TO_GROOVE" else 1
    yaw=math.pi/2+math.radians(15)
    sols=bench.model.solve(scene.center_target_body_m,yaw,branch,[READY_Q],scene.seed*100,48)
    legal=[]; rejected=[]
    for solution in sols:
        _,q,pe,oe,margin=solution; failures=bench.model.collision_codes(q,scene.obstacles)
        row=(solution,failures)
        (legal if not failures and oe<=1. and margin>=0. else rejected).append(row)
    selected=(legal+rejected)[:4]; result=[]
    for index in range(4):
        if index<len(selected):
            (_,q,pe,oe,margin),failures=selected[index]; ok=not failures and oe<=1. and margin>=0.
            result.append(Candidate(CONTRACT_REVISION,scene.scenario_id,scene.scenario_id+":c%d"%index,index,scene.center_target_body_m,yaw,branch,True,True,tuple(float(v) for v in q),pe,oe,margin,not failures,oe<=1.,ok,tuple(failures)))
        else:
            result.append(Candidate(CONTRACT_REVISION,scene.scenario_id,scene.scenario_id+":c%d"%index,index,scene.center_target_body_m,yaw,branch,True,False,failure_codes=("CENTER_IK_BRANCH_NOT_FOUND",)))
    return result
def run(ares_root:Path,ares_r_root:Path,output:Path):
    output=Path(output); output.mkdir(parents=True,exist_ok=False); bench=SkillHandoffBench(ares_root,ares_r_root)
    scenes=[_scene(x,"TRAY_TO_GROOVE") for x in (4101,4102,4103)]+[_scene(x,"TRAY_TO_CABINET_PRELAND") for x in (4201,4202,4203)]
    candidates=[]; transitions=[]; outcomes=[]
    for scene in scenes:
        rows=_candidates(bench,scene); candidates.extend(rows); bench.reset(scene,rows,scene.seed)
        for item in rows:
            t,o=bench.step(item.candidate_id); transitions.append(t); outcomes.append(o)
    for name,rows in (("scenarios",scenes),("candidates",candidates),("transitions",transitions),("outcomes",outcomes)):
        with (output/(name+".jsonl")).open("w") as stream:
            for row in rows: stream.write(json.dumps(row.to_dict(),sort_keys=True)+"\n")
    per={}
    for s in scenes:
        oo=[x for x in outcomes if x.scenario_id==s.scenario_id]; cc=[x for x in candidates if x.scenario_id==s.scenario_id and x.candidate_legal]; vv=[x.geometry_successor_start_feasible for x in oo]
        per[s.scenario_id]={"family":s.family,"candidate_attempts":4,"legal_candidates":len(cc),"successor_feasible":sum(vv),"successor_infeasible":len(vv)-sum(vv),"candidate_sensitive":len(set(vv))>1}
    summary={"contract_revision":CONTRACT_REVISION,"scenario_count":len(scenes),"candidate_attempts":len(candidates),"ik_solutions":sum(x.ik_solution_found for x in candidates),"legal_candidates":sum(x.candidate_legal for x in candidates),"successor_feasible":sum(x.geometry_successor_start_feasible for x in outcomes),"candidate_sensitive_scenarios":sum(x["candidate_sensitive"] for x in per.values()),"per_scene":per,"evidence_boundary":{"geometry":"CPU_TESTED","curobo":"NOT_TESTED_CUDA_UNAVAILABLE","physics_contact":"NOT_TESTED","real_execution":"NOT_TESTED"},"tool_contract":bench.model.tool_contract}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); return summary
