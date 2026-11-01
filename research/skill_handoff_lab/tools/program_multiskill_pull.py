"""Native100 PickTool -> UseTool: two physically separated parameter decisions.

Reuses the complete existing strong lookahead and post-grasp geometric search.
Fixed parameters are interventions, never a replacement for the strong baseline.
No inter-skill resets/state setters. No RL or method-gain claim.
"""
import os
import argparse, gzip, hashlib, json, random, subprocess, time
from pathlib import Path
import numpy as np
import torch
from pull_tool_physics import *

def execute_chain(env, seed, grasp_x=None, hook_y=None, pull_distance=.27, pull_policy='fixed',transfer_height=.08,transfer_stride=2):
    requested_grasp=grasp_x;requested_hook=hook_y
    lookahead=grasp_x is None;rule=True;feedback=True;yaw=0.
    grasp_x=.07 if grasp_x is None else float(grasp_x);hook_y=-.065 if hook_y is None else float(hook_y)
    env.skill_results=[]
    env.reset(seed=seed);u=env.unwrapped;p=TracePlanner(env,vis=False,base_pose=u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=.9,joint_acc_limits=.9);p.manual=True
    def move(pose,label):
        env.phase=label
        if p.move_to_pose_with_screw(pose)==-1:raise Failure('PLAN_'+label)
    try:
        cube_yaw=quat2euler(sp(u.cube).q)[2]
        if lookahead:
            trials=[];start=time.monotonic();tool=sp(u.l_shape_tool)
            for gx in [.07,.105]:
                closing=np.array([0.,1.,0.]);closing*=1 if np.dot(closing,quat2mat(sp(u.agent.tcp).q)[:,1])>=0 else -1
                predicted_grasp=u.agent.build_grasp_pose(np.array([0,0,-1]),closing,tool.p+[gx,0,0]);att=predicted_grasp.inv()*tool
                quat=axangle2quat([0,0,1],cube_yaw);origin=sp(u.cube).p-quat2mat(quat)@np.array([.126,.0625,0]);origin[2]=.026;target=sapien.Pose(origin,quat)*att.inv()
                rr=p.planner.plan_screw(np.r_[target.p,target.q],np.array(serial(u.agent.robot.get_qpos()[0])),time_step=u.control_timestep,use_point_cloud=False)
                trials.append(dict(grasp_x=gx,cost=len(rr['position']) if rr['status']=='Success' else 10000))
            grasp_x=min(trials,key=lambda z:z['cost'])['grasp_x'];env.plans.append(dict(phase='GRASP_SEARCH',step=0,planning_wall_s=time.monotonic()-start,candidates=trials,selected=grasp_x))
        env.decisions.append(dict(stage=0,skill='PICK_TOOL',boundary_step=len(env.records),
            parameters=dict(grasp_x_m=float(grasp_x)),observation=env.measure(),
            contact_observation_valid=False,
            selection='measured_geometry_lookahead' if lookahead else 'predeclared_parameter_intervention',
            precondition='tool on tabletop; handle grasp point within [0.035,0.105]m',
            next_skill_requirement='retain tool through lift and reach hook/pull poses within native budget'))
        tool=sp(u.l_shape_tool);closing=np.array([0.,1.,0.]);closing*=1 if np.dot(closing,quat2mat(sp(u.agent.tcp).q)[:,1])>=0 else -1;g=u.agent.build_grasp_pose(np.array([0,0,-1]),closing,tool.p+np.array([grasp_x,0,0]));move(g*sapien.Pose([0,0,-.065]),'REACH');move(g,'DESCEND');p.close_gripper(t=6)
        if not bool(u.agent.is_grasping(u.l_shape_tool)[0]):raise Failure('TOOL_NOT_GRASPED')
        move(sapien.Pose(sp(u.agent.tcp).p+[0,0,.055],sp(u.agent.tcp).q),'LIFT')
        if not bool(u.agent.is_grasping(u.l_shape_tool)[0]):raise Failure('TOOL_LOST_ON_LIFT')
        env.skill_results.append(dict(skill='PICK_TOOL',boundary_step=len(env.records),
            physical_success=True,observed_postcondition='grasp retained after 55mm commanded lift',
            measurement=env.measure(),remaining_steps=100-len(env.records)))
        attached=sp(u.agent.tcp).inv()*sp(u.l_shape_tool)
        def distance_for_angle(angle):
            if pull_policy=='fixed':return float(pull_distance)
            # Known native goal geometry, never a future success label. Target
            # a 20mm interior margin and keep the same legal 250..290mm range.
            v=sp(u.cube).p[:2]-sp(u.agent.robot).p[:2]
            direction=np.array([np.cos(angle),np.sin(angle)])
            along=float(v@direction);perpendicular2=float(v@v)-along*along
            distance=along-np.sqrt(max(0.,.58**2-perpendicular2))
            return float(np.clip(distance,.25,.29))
        # Task-conditioned bounded model-based search; no physical outcome lookup.
        tool_yaw=cube_yaw+yaw
        if rule:
            candidates=[];start=time.monotonic()
            for ang in [cube_yaw-.12,cube_yaw,cube_yaw+.12]:
                for hy in [-.055,-.065,-.075]:
                    quat=axangle2quat([0,0,1],ang);R=quat2mat(quat);extent=.02*(abs(np.cos(cube_yaw-ang))+abs(np.sin(cube_yaw-ang)));origin=sp(u.cube).p-R@np.array([.15-extent-.004,-hy,0]);origin[2]=transfer_height
                    candidate_distance=distance_for_angle(ang)
                    goals=[origin.copy(),np.r_[origin[:2],.026],np.r_[origin[:2],.026]+R@np.array([-candidate_distance,0,0])]
                    qq=np.array(serial(u.agent.robot.get_qpos()[0]));cost=0;ok=True
                    for goal in goals:
                        target=sapien.Pose(goal,quat)*attached.inv();res=p.planner.plan_screw(np.r_[target.p,target.q],qq,time_step=u.control_timestep,use_point_cloud=False)
                        if res['status']!='Success':ok=False;break
                        cost+=int(np.ceil(len(res['position'])/2));qq=np.r_[res['position'][-1],qq[-2:]]
                    candidates.append(dict(angle=ang,hook_y=hy,pull_distance_m=candidate_distance,valid=ok,predicted_steps=cost))
            good=[c for c in candidates if c['valid'] and (requested_hook is None or abs(c['hook_y']-requested_hook)<1e-8)]
            if good:
                best=min(good,key=lambda c:(c['predicted_steps'],abs(c['angle']),abs(c['hook_y']+.065)));tool_yaw=best['angle'];hook_y=best['hook_y']
            env.plans.append(dict(phase='SEARCH',step=len(env.records),planning_wall_s=time.monotonic()-start,candidates=candidates,selected=[tool_yaw,hook_y]))
        env.decisions.append(dict(stage=1,skill='USE_TOOL',boundary_step=len(env.records),
            parameters=dict(tool_yaw_rad=float(tool_yaw),hook_y_m=float(hook_y),pull_distance_m=distance_for_angle(tool_yaw),pull_policy=pull_policy,transfer_height_m=transfer_height,transfer_path_stride=transfer_stride),
            contact_observation_valid=True,
            observation=env.measure(),previous_skill_result=env.skill_results[-1],
            selection='measured_attachment_geometric_search' if requested_hook is None else 'hook_parameter_intervention_with_yaw_search',
            precondition='measured grasp retained after lift; attachment refreshed',
            next_skill_requirement='native cube goal region by 100 physical steps'))
        tool_quat=axangle2quat([0,0,1],tool_yaw);R=quat2mat(tool_quat);extent=.02*(abs(np.cos(cube_yaw-tool_yaw))+abs(np.sin(cube_yaw-tool_yaw)))
        def tool_goal(xyz,label):
            nonlocal attached
            if feedback:attached=sp(u.agent.tcp).inv()*sp(u.l_shape_tool)
            previous_stride=getattr(p,'stride',2)
            p.stride=transfer_stride if label=='TRANSFER' else previous_stride
            try:move(sapien.Pose(xyz,tool_quat)*attached.inv(),label)
            finally:p.stride=previous_stride
        cube=sp(u.cube).p;origin=cube-R@np.array([.15-extent-.004,-hook_y,0]);origin[2]=transfer_height;tool_goal(origin,'TRANSFER')
        cube=sp(u.cube).p;origin=cube-R@np.array([.15-extent-.004,-hook_y,0]);origin[2]=.026;tool_goal(origin,'HOOK')
        selected_distance=env.decisions[-1]['parameters']['pull_distance_m']
        origin=sp(u.l_shape_tool).p.copy()+R@np.array([-selected_distance,0,0]);tool_goal(origin,'PULL')
    finally:p.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=740000)
    ap.add_argument('--grasp-x',type=float)
    ap.add_argument('--hook-y',type=float)
    ap.add_argument('--pull-distance',type=float,default=.27)
    ap.add_argument('--pull-policy',choices=['fixed','goal'],default='fixed')
    ap.add_argument('--transfer-stride',type=int,choices=[1,2],default=2)
    ap.add_argument('--transfer-height',type=float,choices=[.08,.10,.12],default=.08)
    ap.add_argument('--label',default='strong_lookahead')
    args=ap.parse_args()
    if args.grasp_x is not None and not .035<=args.grasp_x<=.105:
        ap.error('grasp-x must lie within the audited 35..105 mm handle interval')
    if args.hook_y is not None and not any(abs(args.hook_y-x)<1e-8 for x in [-.055,-.065,-.075]):
        ap.error('hook-y must be one of -0.055,-0.065,-0.075 m')
    if not .25<=args.pull_distance<=.29:ap.error('pull-distance must lie within 250..290 mm')
    args.out.mkdir(parents=True,exist_ok=False)
    random.seed(31031);np.random.seed(31031);torch.manual_seed(31031)
    root=Path(__file__).parent
    sources={}
    for name in ['program_multiskill_pull.py','pull_tool_physics.py','handoff_physics.py']:
        data=(root/name).read_bytes();sources[name]=hashlib.sha256(data).hexdigest();(args.out/name).write_bytes(data)
    pinned=Path('/home/andyee/robot-research/ARES-Skilllab/external/benchmark_scout_20261027/ManiSkill')
    native=pinned/'mani_skill/envs/tasks/tabletop/pull_cube_tool.py'
    manifest=dict(resource_context=os.environ.get('ARES_RESOURCE_CONTEXT','UNSPECIFIED'),task='PullCubeTool-v1',native_horizon=100,initial_distribution='NATIVE_UNCHANGED',
        source_kind='ACTUAL_PHYSX_EXECUTION',hardware_commands=False,optimizer_updates=0,
        seed=args.seed,random_seed=31031,requested_parameters=dict(grasp_x=args.grasp_x,hook_y=args.hook_y,pull_distance=args.pull_distance,pull_policy=args.pull_policy,transfer_height=args.transfer_height,transfer_stride=args.transfer_stride),
        baseline='unchanged measured-attachment lookahead + nine-candidate post-grasp geometry search',
        decision_boundaries=['before physical grasp','after retained grasp and commanded 55mm lift'],
        reset_count=1,source_sha256=sources,native_source_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        reset_contact_measurement='UNVALIDATED_UNTIL_FIRST_ENV_STEP; raw initial contact may reflect stale simulator buffer',
        maniskill_revision=subprocess.check_output(['git','-C',str(pinned),'rev-parse','HEAD'],text=True).strip(),
        maniskill_license_sha256=hashlib.sha256((pinned/'LICENSE').read_bytes()).hexdigest(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        fairness='same native100 full physics; fixed parameters are diagnostic interventions; no oracle outcome selection')
    (args.out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    env=Recorder(gym.make('PullCubeTool-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',
        reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1))
    error=None;started=time.monotonic()
    try:
        try:execute_chain(env,args.seed,args.grasp_x,args.hook_y,args.pull_distance,args.pull_policy,args.transfer_height,args.transfer_stride)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        if len(env.records)<100:
            q=np.array(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]))
            grip=-1 if any(s['phase']=='GRASP' for s in env.records) else 1
            env.phase='TERMINAL_HOLD'
            while len(env.records)<100:env.step(np.r_[q,grip])
        final=env.measure()
        if len(env.decisions)>=2:
            env.skill_results.append(dict(skill='USE_TOOL',boundary_step=100,
                physical_success=final['native_success'],measurement=final,
                physical_steps=100-env.decisions[1]['boundary_step'],termination='native_horizon',error=error))
        elif not env.skill_results:
            env.skill_results.append(dict(skill='PICK_TOOL',boundary_step=len(env.records),physical_success=False,
                measurement=final,termination='predecessor_failure',error=error))
        row=dict(seed=args.seed,label=args.label,success=final['native_success'],error=error,steps=len(env.records),
            initial_sha256=env.initial_sha,decisions=len(env.decisions),skill_results=env.skill_results,
            parameters=[d['parameters'] for d in env.decisions],wall_s=time.monotonic()-started,
            planning_wall_s=sum(p['planning_wall_s'] for p in env.plans),
            phases={k:sum(s['phase']==k for s in env.records) for k in sorted({s['phase'] for s in env.records})})
        path=args.out/'TRAJECTORY.json.gz'
        data=dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,
                  plans=env.plans,decisions=env.decisions,skill_results=env.skill_results)
        payload=json.dumps(data,separators=(',',':'))
        with gzip.open(path,'wt') as f:f.write(payload)
        row.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (args.out/'RESULTS.json').write_text(json.dumps([row],indent=2))
        print(json.dumps(row),flush=True)
    finally:env.close()

if __name__=='__main__':main()
