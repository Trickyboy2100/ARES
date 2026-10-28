"""Native100 PullCubeTool probes: official solver and measured grasp-transform replanning."""
import argparse,gzip,hashlib,json,time,subprocess
from pathlib import Path
import gymnasium as gym
import numpy as np
import sapien
import mani_skill.envs
from PIL import Image,ImageDraw
from transforms3d.quaternions import axangle2quat,quat2mat
from transforms3d.euler import quat2euler
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.solutions import pull_cube_tool as official
from handoff_physics import serial

class Horizon(Exception):pass
class Failure(Exception):pass

def sp(actor):return actor.pose.sp

def rawpose(p):return np.r_[p.p,p.q].tolist()

class Recorder(gym.Wrapper):
    def __init__(self,env):super().__init__(env);self.horizon={'PullCubeTool-v1':100,'PlugCharger-v1':200}[env.unwrapped.spec.id];self.video=None;self.phase='RESET';self.records=[];self.plans=[];self.writer=None
    def reset(self,**kwargs):
        r=self.env.reset(**kwargs);self.records=[];self.plans=[];self.decisions=[];self.phase='RESET';self.initial=serial(self.unwrapped.get_state_dict());self.initial_sha=hashlib.sha256(json.dumps(self.initial,sort_keys=True).encode()).hexdigest();self.initial_measure=self.measure();return r
    def measure(self):
        u=self.unwrapped;actor=u.l_shape_tool if hasattr(u,'l_shape_tool') else u.charger;other=u.cube if hasattr(u,'cube') else u.receptacle;tcp=sp(u.agent.tcp);tool=sp(actor);cube=sp(other)
        return dict(q=serial(u.agent.robot.get_qpos()[0]),qvel=serial(u.agent.robot.get_qvel()[0]),tcp=rawpose(tcp),tool=rawpose(tool),cube=rawpose(cube),tcp_T_tool=rawpose(tcp.inv()*tool),tool_linear_velocity=serial(actor.linear_velocity[0]),tool_angular_velocity=serial(actor.angular_velocity[0]),cube_linear_velocity=serial(other.linear_velocity[0]),grasped=bool(u.agent.is_grasping(actor)[0]),tool_cube_force_N=float(np.linalg.norm(serial(u.scene.get_pairwise_contact_forces(actor,other)[0]))),native_success=bool(u.evaluate()['success'][0]),cube_base_distance_m=float(np.linalg.norm(cube.p[:2]-sp(u.agent.robot).p[:2])))
    def step(self,action):
        if len(self.records)>=self.horizon:raise Horizon('NATIVE_HORIZON')
        ret=self.env.step(action);self.records.append(dict(step=len(self.records)+1,phase=self.phase,action=serial(action),**self.measure()))
        if getattr(self,'progress',None):self.progress.write_text(json.dumps(dict(steps=len(self.records),last=self.records[-1])))
        if self.video:
            frame=self.env.render();frame=frame.cpu().numpy() if hasattr(frame,'cpu') else np.asarray(frame);frame=frame.astype(np.uint8).squeeze();im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,im.width,52),fill='white');draw.text((8,4),f'{self.label} | step {len(self.records)}/{self.horizon}\n{self.phase} | SIMULATED PHYSICS / NO ROBOT MOTION',fill='black');frame=np.asarray(im)
            if self.writer is None:self.writer=subprocess.Popen(['/usr/bin/ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{im.width}x{im.height}','-r','20','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(self.video)],stdin=subprocess.PIPE)
            self.writer.stdin.write(frame.tobytes())
        return ret
    def finish_video(self):
        if self.writer:self.writer.stdin.close();assert self.writer.wait()==0;self.writer=None

class TracePlanner(PandaArmMotionPlanningSolver):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.number=0;self.manual=False
    def move_to_pose_with_screw(self,pose,**kwargs):
        if not self.manual:self.env.phase=['REACH','DESCEND','LIFT','TRANSFER','HOOK','PULL'][min(self.number,5)];self.number+=1
        started=time.monotonic();r=super().move_to_pose_with_screw(pose,dry_run=True)
        if self.manual and isinstance(r,int):
            r=self.planner.plan_qpos_to_pose(np.r_[pose.p,pose.q],self.robot.get_qpos()[0].cpu().numpy(),time_step=self.base_env.control_timestep,planning_time=.5,use_point_cloud=False,wrt_world=True)
            if r['status']!='Success':r=-1
        wall=time.monotonic()-started
        bad=isinstance(r,int);self.env.plans.append(dict(phase=self.env.phase,step=len(self.env.records),target=rawpose(pose),planning_wall_s=wall,status='FAIL' if bad else r['status'],position=None if bad else r['position'].tolist()))
        if bad:return -1
        if len(r['position'])==0:
            r=dict(r,position=self.robot.get_qpos()[0,:7].cpu().numpy()[None],velocity=np.zeros((1,7)))
        if self.manual:
            # Legal absolute-joint commands; simulation evaluates measured motion.
            # Keep original planner path in evidence; no claim of native TOPP timing.
            idx=list(range(0,len(r['position']),getattr(self,'stride',2)))
            if idx[-1]!=len(r['position'])-1:idx.append(len(r['position'])-1)
            r=dict(r,position=r['position'][idx],velocity=r['velocity'][idx])
        return self.follow_path(r,refine_steps=kwargs.get('refine_steps',0))
    def close_gripper(self,*args,**kwargs):self.env.phase='GRASP';return super().close_gripper(*args,**kwargs)

official.PandaArmMotionPlanningSolver=TracePlanner

def strong(env,seed,grasp_x=.07,yaw=0.,hook_y=-.065,feedback=True,rule=False,lookahead=False):
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
        tool=sp(u.l_shape_tool);closing=np.array([0.,1.,0.]);closing*=1 if np.dot(closing,quat2mat(sp(u.agent.tcp).q)[:,1])>=0 else -1;g=u.agent.build_grasp_pose(np.array([0,0,-1]),closing,tool.p+np.array([grasp_x,0,0]));move(g*sapien.Pose([0,0,-.065]),'REACH');move(g,'DESCEND');p.close_gripper(t=6)
        if not bool(u.agent.is_grasping(u.l_shape_tool)[0]):raise Failure('TOOL_NOT_GRASPED')
        move(sapien.Pose(sp(u.agent.tcp).p+[0,0,.055],sp(u.agent.tcp).q),'LIFT')
        if not bool(u.agent.is_grasping(u.l_shape_tool)[0]):raise Failure('TOOL_LOST_ON_LIFT')
        attached=sp(u.agent.tcp).inv()*sp(u.l_shape_tool)
        # Task-conditioned bounded model-based search; no physical outcome lookup.
        tool_yaw=cube_yaw+yaw
        if rule:
            candidates=[];start=time.monotonic()
            for ang in [cube_yaw-.12,cube_yaw,cube_yaw+.12]:
                for hy in [-.055,-.065,-.075]:
                    quat=axangle2quat([0,0,1],ang);R=quat2mat(quat);extent=.02*(abs(np.cos(cube_yaw-ang))+abs(np.sin(cube_yaw-ang)));origin=sp(u.cube).p-R@np.array([.15-extent-.004,-hy,0]);origin[2]=.08
                    goals=[origin.copy(),np.r_[origin[:2],.026],np.r_[origin[:2],.026]+R@np.array([-.27,0,0])]
                    qq=np.array(serial(u.agent.robot.get_qpos()[0]));cost=0;ok=True
                    for goal in goals:
                        target=sapien.Pose(goal,quat)*attached.inv();res=p.planner.plan_screw(np.r_[target.p,target.q],qq,time_step=u.control_timestep,use_point_cloud=False)
                        if res['status']!='Success':ok=False;break
                        cost+=int(np.ceil(len(res['position'])/2));qq=np.r_[res['position'][-1],qq[-2:]]
                    candidates.append(dict(angle=ang,hook_y=hy,valid=ok,predicted_steps=cost))
            good=[c for c in candidates if c['valid']]
            if good:
                best=min(good,key=lambda c:(c['predicted_steps'],abs(c['angle']),abs(c['hook_y']+.065)));tool_yaw=best['angle'];hook_y=best['hook_y']
            env.plans.append(dict(phase='SEARCH',step=len(env.records),planning_wall_s=time.monotonic()-start,candidates=candidates,selected=[tool_yaw,hook_y]))
        tool_quat=axangle2quat([0,0,1],tool_yaw);R=quat2mat(tool_quat);extent=.02*(abs(np.cos(cube_yaw-tool_yaw))+abs(np.sin(cube_yaw-tool_yaw)))
        def tool_goal(xyz,label):
            nonlocal attached
            if feedback:attached=sp(u.agent.tcp).inv()*sp(u.l_shape_tool)
            move(sapien.Pose(xyz,tool_quat)*attached.inv(),label)
        cube=sp(u.cube).p;origin=cube-R@np.array([.15-extent-.004,-hook_y,0]);origin[2]=.08;tool_goal(origin,'TRANSFER')
        cube=sp(u.cube).p;origin=cube-R@np.array([.15-extent-.004,-hook_y,0]);origin[2]=.026;tool_goal(origin,'HOOK')
        origin=sp(u.l_shape_tool).p.copy()+R@np.array([-.27,0,0]);tool_goal(origin,'PULL')
    finally:p.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',default='710000,710001,710002,710003');ap.add_argument('--modes',default='official,strong');ap.add_argument('--grasp-x',type=float,default=.07);ap.add_argument('--yaw',type=float,default=0.);ap.add_argument('--hook-y',type=float,default=-.065);ap.add_argument('--grid',action='store_true');ap.add_argument('--pair',action='store_true');ap.add_argument('--video',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    env=Recorder(gym.make('PullCubeTool-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));results=[]
    try:
        for seed in map(int,a.seeds.split(',')):
            digest=None
            configs=[(m,a.grasp_x,a.yaw,a.hook_y) for m in a.modes.split(',')] if not a.grid else [('rule',.07,0.,-.065)]+[(f'g{x:.3f}_y{y:.2f}',x,y,-.065) for x in [.035,.07,.105] for y in [-.12,0.,.12]]
            if a.pair:configs=[('end_grasp',.035,0.,-.065),('center_grasp',.105,0.,-.065)]
            for mode,gx,ya,hy in configs:
                if mode=='rule70':gx=.07
                name=f'{seed}_{mode}';env.progress=a.out/f'{name}.progress.json';env.video=a.out/f'{name}.mp4' if a.video else None;env.label=name;started=time.monotonic();error=None
                try:
                    if mode=='official':
                        r=official.solve(env,seed=seed)
                        if isinstance(r,int):raise Failure('OFFICIAL_PLANNER_FAILURE')
                    else:strong(env,seed,gx,ya,hy,feedback=mode!='open_loop',rule=mode in ['rule','rule70','lookahead'],lookahead=mode=='lookahead')
                except (Horizon,Failure) as e:error=str(e)
                if digest is None:digest=env.initial_sha
                assert digest==env.initial_sha
                # Finished early: hold actual measured joint state to the native horizon.
                if len(env.records)<100:
                    q=np.array(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));grip=-1 if any(z['phase']=='GRASP' for z in env.records) else 1;env.phase='TERMINAL_HOLD'
                    while len(env.records)<100:env.step(np.r_[q,grip])
                env.finish_video()
                row=dict(seed=seed,mode=mode,parameters=dict(grasp_x=gx,yaw=ya,hook_y=hy),error=error,success=env.records[-1]['native_success'],success_once=any(x['native_success'] for x in env.records),initial_success=env.initial_measure['native_success'],initial_sha256=env.initial_sha,steps=len(env.records),wall_s=time.monotonic()-started,planning_wall_s=sum(x['planning_wall_s'] for x in env.plans),phases={k:sum(x['phase']==k for x in env.records) for k in {x['phase'] for x in env.records}},final=env.records[-1])
                path=a.out/f'{name}.json.gz'
                with gzip.open(path,'wt') as f:json.dump(dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans),f,separators=(',',':'))
                row['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();results.append(row);(a.out/'RESULTS.json').write_text(json.dumps(results,indent=2));print(json.dumps({k:row[k] for k in ['seed','mode','success','error','steps','wall_s','phases']}),flush=True)
    finally:env.close()
if __name__=='__main__':main()
