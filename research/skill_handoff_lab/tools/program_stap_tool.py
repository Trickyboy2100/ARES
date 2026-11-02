"""Offline native-derived tool-load study. Original100-step goal retained.
Shared native-timed paths (no decimation), measured stable grasp boundary.
Based on existing ARES tool controller; this is common infrastructure, not RL.
"""
import argparse,gzip,hashlib,json,random,time,subprocess
from pathlib import Path
import torch
from program_multiskill_pull import *
from program_multiskill_contact_probe import ContactRecorder
from mani_skill.envs.tasks.tabletop.pull_cube_tool import PullCubeToolEnv
from mani_skill.utils.registration import register_env

@register_env('ARES-ToolLoad-v0',max_episode_steps=100)
class ToolLoadEnv(PullCubeToolEnv):
    def __init__(self,*args,density_scale=1.,**kwargs):
        self.density_scale=float(density_scale)
        super().__init__(*args,**kwargs)
    def _build_l_shaped_tool(self,handle_length,hook_length,width,height):
        # Same native primitive geometry/material; scale both collision densities
        # before GPU construction. Never change a held object's mass/state.
        b=self.scene.create_actor_builder()
        mat=sapien.render.RenderMaterial();mat.set_base_color([1,0,0,1]);mat.metallic=1.;mat.roughness=0.;mat.specular=1.
        for center,half,density in [([handle_length/2,0,0],[handle_length/2,width/2,height/2],500.),([handle_length-hook_length/2,width,0],[hook_length/2,width,height/2],1000.)]:
            pose=sapien.Pose(center)
            b.add_box_collision(pose,half,density=density*self.density_scale)
            b.add_box_visual(pose,half,material=mat)
        return b.build(name='l_shape_tool')

def check_stability(env,p):
    rows=[];streak=0
    q=np.asarray(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));prev=env.measure()
    for _ in range(min(env.stability_max,100-len(env.records))):
        env.phase='GRASP_STABILITY';env.step(np.r_[q,-1.]);now=env.measure()
        def delta(key):
            a=np.array(prev[key]);b=np.array(now[key]);return float(np.linalg.norm(a[:3]-b[:3])),float(2*np.arccos(np.clip(abs((a[3:]/np.linalg.norm(a[3:]))@(b[3:]/np.linalg.norm(b[3:]))),0,1)))
        dp,da=delta('tcp_T_tool');wp,wa=delta('tool')
        ok=now['grasped'] and dp<=.0005 and da<=np.deg2rad(.5) and wp/.05<.02 and wa/.05<.1
        streak=streak+1 if ok else 0
        rows.append(dict(step=len(env.records),relative_translation_m=dp,relative_rotation_rad=da,world_speed_m_s=wp/.05,world_angular_rad_s=wa/.05,instantaneous_linear=now['tool_linear_velocity'],instantaneous_angular=now['tool_angular_velocity'],pass_=bool(ok),streak=streak))
        prev=now
        if streak>=3:break
    env.stability=dict(pass_=streak>=3,checks=rows)
    return dict(pass_=streak>=3,**{'pass':streak>=3},checks=rows)

def execute_chain(env, seed, grasp_x=None, hook_y=None, pull_distance=.27, pull_policy='fixed',transfer_height=.08,transfer_stride=1):
    requested_grasp=grasp_x;requested_hook=hook_y
    lookahead=grasp_x is None;rule=True;feedback=True;yaw=0.
    grasp_x=.07 if grasp_x is None else float(grasp_x);hook_y=-.065 if hook_y is None else float(hook_y)
    env.skill_results=[]
    env.reset(seed=seed);u=env.unwrapped;p=TracePlanner(env,vis=False,base_pose=u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.8,joint_acc_limits=env.acceleration);p.manual=True;p.stride=1
    selector=getattr(env,'parameter_selector',None)
    if selector is not None:
        grasp_x=float(selector.choose(env,0));requested_grasp=grasp_x;lookahead=False
    def move(pose,label):
        env.phase=label
        if p.move_to_pose_with_screw(pose)==-1:raise Failure('PLAN_'+label)
    try:
        cube_yaw=quat2euler(sp(u.cube).q)[2]
        nominal_yaw=cube_yaw if env.target_y is None else float(np.arctan2(sp(u.cube).p[1]-env.target_y,sp(u.cube).p[0]-(sp(u.agent.robot).p[0]+.57)))
        if lookahead:
            trials=[];start=time.monotonic();tool=sp(u.l_shape_tool)
            for gx in getattr(env,'grasp_candidates',[.07,.105]):
                closing=np.array([0.,1.,0.]);closing*=1 if np.dot(closing,quat2mat(sp(u.agent.tcp).q)[:,1])>=0 else -1
                predicted_grasp=u.agent.build_grasp_pose(np.array([0,0,-1]),closing,tool.p+[gx,0,0]);att=predicted_grasp.inv()*tool
                quat=axangle2quat([0,0,1],cube_yaw);origin=sp(u.cube).p-quat2mat(quat)@np.array([.126,.0625,0]);origin[2]=.026;target=sapien.Pose(origin,quat)*att.inv()
                rr=p.planner.plan_screw(np.r_[target.p,target.q],np.array(serial(u.agent.robot.get_qpos()[0])),time_step=u.control_timestep,use_point_cloud=False)
                trials.append(dict(grasp_x=gx,cost=len(rr['position']) if rr['status']=='Success' else 10000))
            grasp_x=min(trials,key=lambda z:z['cost'])['grasp_x'];env.plans.append(dict(phase='GRASP_SEARCH',step=0,planning_wall_s=time.monotonic()-start,candidates=trials,selected=grasp_x))
        env.decisions.append(dict(stage=0,skill='PICK_TOOL',boundary_step=len(env.records),
            parameters=dict(grasp_x_m=float(grasp_x)),observation=env.measure(),
            contact_observation_valid=False,
            selection=selector.method if selector is not None else ('measured_geometry_lookahead' if lookahead else 'predeclared_parameter_intervention'),
            precondition='tool on tabletop; handle grasp point within [0.035,0.105]m',
            next_skill_requirement='retain tool through lift and reach hook/pull poses within native budget'))
        tool=sp(u.l_shape_tool);closing=np.array([0.,1.,0.]);closing*=1 if np.dot(closing,quat2mat(sp(u.agent.tcp).q)[:,1])>=0 else -1;g=u.agent.build_grasp_pose(np.array([0,0,-1]),closing,tool.p+np.array([grasp_x,0,0]));move(g*sapien.Pose([0,0,-.065]),'REACH');move(g,'DESCEND');p.close_gripper(t=6)
        if not bool(u.agent.is_grasping(u.l_shape_tool)[0]):raise Failure('TOOL_NOT_GRASPED')
        move(sapien.Pose(sp(u.agent.tcp).p+[0,0,.055],sp(u.agent.tcp).q),'LIFT')
        if not bool(u.agent.is_grasping(u.l_shape_tool)[0]):raise Failure('TOOL_LOST_ON_LIFT')
        stable = check_stability(env, p)
        if not stable['pass'] and not env.allow_unstable:raise Failure('UNSTABLE_GRASP_CONTRACT')
        env.skill_results.append(dict(skill='PICK_TOOL',boundary_step=len(env.records),
            physical_success=True,observed_postcondition='retained grasp and measured stability',stability=stable,
            measurement=env.measure(),remaining_steps=100-len(env.records)))
        if selector is not None:
            requested_hook=selector.choose(env,1)
            if requested_hook is not None:hook_y=float(requested_hook)
        attached=sp(u.agent.tcp).inv()*sp(u.l_shape_tool)
        def distance_for_angle(angle):
            if env.target_y is not None:
                target=np.array([sp(u.agent.robot).p[0]+.57,env.target_y]);direction=np.array([np.cos(angle),np.sin(angle)]);return float(np.clip((sp(u.cube).p[:2]-target)@direction,.20,.36))
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
            guard=selector is not None and selector.method=='stap_guard'
            original_hook=(-.065+.01*selector.logs[0]['planned_actions'][1][0]) if guard else None
            hook_candidates=[-.055,-.065,-.075] if requested_hook is None else [float(requested_hook)]
            if guard and abs(original_hook-requested_hook)>1e-8:hook_candidates.append(original_hook)
            for ang in [nominal_yaw-.12,nominal_yaw,nominal_yaw+.12]:
                for hy in hook_candidates:
                    quat=axangle2quat([0,0,1],ang);R=quat2mat(quat);extent=.02*(abs(np.cos(cube_yaw-ang))+abs(np.sin(cube_yaw-ang)));origin=sp(u.cube).p-R@np.array([.15-extent-.004,-hy,0]);origin[2]=transfer_height
                    candidate_distance=distance_for_angle(ang)
                    goals=[origin.copy(),np.r_[origin[:2],.026],np.r_[origin[:2],.026]+R@np.array([-candidate_distance,0,0])]
                    qq=np.array(serial(u.agent.robot.get_qpos()[0]));cost=0;ok=True
                    for goal in goals:
                        target=sapien.Pose(goal,quat)*attached.inv();res=p.planner.plan_screw(np.r_[target.p,target.q],qq,time_step=u.control_timestep,use_point_cloud=False)
                        if res['status']!='Success':ok=False;break
                        cost+=len(res['position']);qq=np.r_[res['position'][-1],qq[-2:]]
                    candidates.append(dict(angle=ang,hook_y=hy,pull_distance_m=candidate_distance,valid=ok,predicted_steps=cost))
            good=[c for c in candidates if c['valid'] and (requested_hook is None or abs(c['hook_y']-requested_hook)<1e-8)]
            if guard and (not good or min(c['predicted_steps'] for c in good)>100-len(env.records)):
                fallback=[c for c in candidates if c['valid'] and abs(c['hook_y']-original_hook)<1e-8]
                if fallback and (not good or min(c['predicted_steps'] for c in fallback)<min(c['predicted_steps'] for c in good)):
                    env.plans.append(dict(phase='CONTRACT_BUDGET_GUARD',step=len(env.records),requested_hook=requested_hook,retained_hook=original_hook,remaining_steps=100-len(env.records),proposed_min_steps=None if not good else min(c['predicted_steps'] for c in good),retained_min_steps=min(c['predicted_steps'] for c in fallback),uses_future_physics=False))
                    good=fallback
            if good:
                best=min(good,key=lambda c:(c['predicted_steps'],abs(c['angle']),abs(c['hook_y']+.065)));tool_yaw=best['angle'];hook_y=best['hook_y']
            env.plans.append(dict(phase='SEARCH',step=len(env.records),planning_wall_s=time.monotonic()-start,candidates=candidates,selected=[tool_yaw,hook_y]))
        env.decisions.append(dict(stage=1,skill='USE_TOOL',boundary_step=len(env.records),
            parameters=dict(tool_yaw_rad=float(tool_yaw),hook_y_m=float(hook_y),pull_distance_m=distance_for_angle(tool_yaw),pull_policy=pull_policy,transfer_height_m=transfer_height,transfer_path_stride=transfer_stride),
            contact_observation_valid=True,
            observation=env.measure(),previous_skill_result=env.skill_results[-1],
            selection=selector.method if selector is not None else ('measured_attachment_geometric_search' if requested_hook is None else 'hook_parameter_intervention_with_yaw_search'),
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
        # Shared observed-goal correction, never native-success oracle.
        if env.target_y is not None and env.target_feedback:
            for correction in range(2):
                delta=np.array([sp(u.agent.robot).p[0]+.57,env.target_y])-sp(u.cube).p[:2]
                if np.linalg.norm(delta)<=.012 or 100-len(env.records)<6:break
                delta*=min(1.,.08/max(1e-9,np.linalg.norm(delta)))
                xyz=sp(u.l_shape_tool).p.copy();xyz[:2]+=delta
                tool_goal(xyz,'TARGET_FEEDBACK')

    finally:p.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',default='743000,743001');ap.add_argument('--mass-scales',default='.75,1.25');ap.add_argument('--grasps',default='.07,.105');ap.add_argument('--allow-unstable',action='store_true');ap.add_argument('--stability-max',type=int,default=6);ap.add_argument('--acceleration',type=float,default=3.);ap.add_argument('--target-y',type=float);ap.add_argument('--hook-y',type=float);ap.add_argument('--target-feedback',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(31102);np.random.seed(31102);random.seed(31102);rows=[]
    for density in map(float,a.mass_scales.split(',')):
        env=ContactRecorder(gym.make('ARES-ToolLoad-v0',density_scale=density,num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));env.allow_unstable=a.allow_unstable;env.stability_max=a.stability_max;env.acceleration=a.acceleration;env.target_y=a.target_y;env.target_feedback=a.target_feedback
        try:
            for seed in map(int,a.seeds.split(',')):
                for gx in map(float,a.grasps.split(',')):
                    out=a.out/f'{seed}_m{density}_g{gx}';out.mkdir();error=None;env.stability=None;start=time.monotonic()
                    source={}
                    for name in ['program_stap_tool.py','program_stap_transfer_probe.py','program_multiskill_pull.py','program_multiskill_contact_probe.py','pull_tool_physics.py','handoff_physics.py']:
                        data=Path(__file__).with_name(name).read_bytes();(out/name).write_bytes(data);source[name]=hashlib.sha256(data).hexdigest()
                    manifest=dict(task='ARES-ToolTarget-v0' if a.target_y is not None else 'ARES-ToolLoad-v0',native_parent='PullCubeTool-v1',native_goal_and_horizon_unchanged=True,seed=seed,density_scale=density,grasp_x=gx,allow_unstable=a.allow_unstable,source_sha256=source,planner_velocity_limit=1.8,planner_acceleration_limit=a.acceleration,stability_max=a.stability_max,target_y=a.target_y,hook_y=a.hook_y,target_feedback=a.target_feedback,path_stride=1,scope='ACTUAL_OFFLINE_PHYSICS; no hardware or RL',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
                    (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2))
                    try:execute_chain(env,seed,grasp_x=gx,hook_y=a.hook_y,transfer_stride=1)
                    except Exception as exc:error=repr(exc)
                    q=np.asarray(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));env.phase='TERMINAL_HOLD'
                    while len(env.records)<100:env.step(np.r_[q,-1.])
                    body=env.unwrapped.l_shape_tool._bodies[0]
                    physical=dict(mass_kg=float(body.mass),inertia=list(map(float,body.inertia)),densities=[float(x.density) for x in body.get_collision_shapes()])
                    row=dict(seed=seed,density_scale=density,grasp_x=gx,success=env.records[-1]['native_success'],error=error,steps=len(env.records),stability=env.stability,initial_sha256=env.initial_sha,decisions=len(env.decisions),wall_s=time.monotonic()-start,physical=physical,target_y=a.target_y,target_distance_m=None if a.target_y is None else float(np.linalg.norm(np.array(env.records[-1]['cube'][:2])-np.array([sp(env.unwrapped.agent.robot).p[0]+.57,a.target_y]))))
                    row['task_success']=row['success'] and (row['target_distance_m'] is None or row['target_distance_m']<=.04)
                    data=dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans,decisions=env.decisions,skill_results=env.skill_results)
                    with gzip.open(out/'TRAJECTORY.json.gz','wt') as f:json.dump(data,f,separators=(',',':'))
                    row['path']=str(out/'TRAJECTORY.json.gz');row['sha256']=hashlib.sha256((out/'TRAJECTORY.json.gz').read_bytes()).hexdigest();(out/'RESULTS.json').write_text(json.dumps([row],indent=2));rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(json.dumps({k:row[k] for k in ['seed','density_scale','grasp_x','success','error','decisions','wall_s']}),flush=True)
        finally:env.close()
if __name__=='__main__':main()
