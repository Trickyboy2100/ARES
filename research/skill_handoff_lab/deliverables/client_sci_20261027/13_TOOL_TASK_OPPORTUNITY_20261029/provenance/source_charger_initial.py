"""Native200 second-task probe; observed grasp transform and bounded pose feedback."""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import gymnasium as gym
import numpy as np
import sapien,trimesh
from transforms3d.euler import euler2quat
from mani_skill.examples.motionplanning.panda.solutions import plug_charger as official
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb
from pull_tool_physics import Recorder,TracePlanner,Horizon,Failure,sp,serial

class OfficialCharger(TracePlanner):
    def move_to_pose_with_screw(self,pose,**kwargs):
        self.env.phase=['REACH','DESCEND','ALIGN','ALIGN_REFINE','INSERT'][min(self.number,4)];self.number+=1
        # Keep official time parameterization and control calls unchanged.
        n=self.number;self.number=0
        started=time.monotonic()
        from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
        r=PandaArmMotionPlanningSolver.move_to_pose_with_screw(self,pose,dry_run=True);self.number=n
        bad=isinstance(r,int);self.env.plans.append(dict(phase=self.env.phase,step=len(self.env.records),planning_wall_s=time.monotonic()-started,status='FAIL' if bad else r['status'],target=np.r_[pose.p,pose.q].tolist(),position=None if bad else r['position'].tolist()))
        return -1 if bad else self.follow_path(r,refine_steps=kwargs.get('refine_steps',0))
official.PandaArmMotionPlanningSolver=OfficialCharger

def strong(env,seed,angle=15.,standoff=.04,feedback=True):
    env.reset(seed=seed);u=env.unwrapped;p=TracePlanner(env,vis=False,base_pose=u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False,joint_vel_limits=1.,joint_acc_limits=1.);p.manual=True;p.stride=1
    def move(target,label):
        env.phase=label
        if p.move_to_pose_with_screw(target)==-1:raise Failure('PLAN_'+label)
    try:
        box=trimesh.primitives.Box(extents=np.array(u._base_size)*2,transform=u.charger_base_pose.sp.to_transformation_matrix());g=compute_grasp_info_by_obb(box,approaching=np.array([0,0,-1]),target_closing=sp(u.agent.tcp).to_transformation_matrix()[:3,1],depth=.025);pose=u.agent.build_grasp_pose(np.array([0,0,-1]),g['closing'],g['center'])*sapien.Pose(q=euler2quat(0,np.deg2rad(angle),0))
        move(pose*sapien.Pose([0,0,-.05]),'REACH');move(pose,'DESCEND');p.close_gripper()
        if not bool(u.agent.is_grasping(u.charger)[0]):raise Failure('CHARGER_NOT_GRASPED')
        move(sapien.Pose(sp(u.agent.tcp).p+[0,0,.04],sp(u.agent.tcp).q),'LIFT')
        if not bool(u.agent.is_grasping(u.charger)[0]):raise Failure('CHARGER_LOST_ON_LIFT')
        attached=sp(u.agent.tcp).inv()*sp(u.charger)
        goal=u.goal_pose.sp
        move(goal*sapien.Pose([-standoff,0,0])*attached.inv(),'ALIGN')
        if feedback:attached=sp(u.agent.tcp).inv()*sp(u.charger)
        move(goal*attached.inv(),'INSERT')
        for k in range(3 if feedback else 0):
            if bool(u.evaluate()['success'][0]):break
            attached=sp(u.agent.tcp).inv()*sp(u.charger);move(goal*attached.inv(),f'REFINE_{k}')
    finally:p.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',default='720000,720001,720002,720003');p.add_argument('--modes',default='official,strong');p.add_argument('--grid',action='store_true');p.add_argument('--video',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);env=Recorder(gym.make('PlugCharger-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));results=[]
    try:
        for seed in map(int,a.seeds.split(',')):
            digest=None;cfg=[(m,15.,.04) for m in a.modes.split(',')] if not a.grid else [('rule',15.,.04)]+[(f'a{ang}_d{d}',ang,d) for ang in [0.,15.,30.] for d in [.025,.05,.075]]
            for mode,angle,standoff in cfg:
                name=f'{seed}_{mode}';env.video=a.out/f'{name}.mp4' if a.video else None;env.label=name;error=None;started=time.monotonic()
                try:
                    if mode=='official':r=official.solve(env,seed=seed)
                    else:strong(env,seed,angle,standoff,mode!='open_loop')
                except (Horizon,Failure) as e:error=str(e)
                if digest is None:digest=env.initial_sha
                assert digest==env.initial_sha
                if len(env.records)<200:
                    q=np.array(serial(env.unwrapped.agent.robot.get_qpos()[0,:7]));grip=-1 if any(z['phase']=='GRASP' for z in env.records) else 1;env.phase='TERMINAL_HOLD'
                    while len(env.records)<200:env.step(np.r_[q,grip])
                env.finish_video();measure=serial(env.unwrapped.evaluate());row=dict(task='PlugCharger-v1',seed=seed,mode=mode,parameters=dict(angle=angle,standoff=standoff),success=env.records[-1]['native_success'],error=error,initial_sha256=env.initial_sha,steps=len(env.records),wall_s=time.monotonic()-started,planning_wall_s=sum(z['planning_wall_s'] for z in env.plans),phases={k:sum(z['phase']==k for z in env.records) for k in {z['phase'] for z in env.records}},final=measure)
                path=a.out/f'{name}.json.gz'
                with gzip.open(path,'wt') as f:json.dump(dict(summary=row,initial=env.initial,initial_measure=env.initial_measure,steps=env.records,plans=env.plans),f,separators=(',',':'))
                row['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();results.append(row);(a.out/'RESULTS.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
    finally:env.close()
if __name__=='__main__':main()
