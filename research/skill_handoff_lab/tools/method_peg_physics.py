"""Native 100-step Peg planner/feedback authority probe. No teleport or task patch.

Based on the Apache-2.0 pinned ManiSkill Panda peg solution. Runtime edits are
only planner speed fractions (<=1), refinement count, and legal goal parameters.
"""
import argparse, hashlib, inspect, json, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs
from method_stack_physics import serial

class BudgetEnd(Exception):pass

class Recorded(gym.Wrapper):
    def __init__(self,env):super().__init__(env);self.rows=[];self.writer=None;self.last=None;self.stage=0
    def reset(self,**kw):
        obs,info=self.env.reset(**kw);self.rows=[];self.initial=serial(self.unwrapped.get_state_dict());self.last=None
        return obs,info
    def step(self,action):
        if len(self.rows)>=100:raise BudgetEnd()
        before=self.unwrapped.peg.pose.raw_pose.clone();result=self.env.step(action);u=self.unwrapped;self.last=action
        self.rows.append(serial(dict(step=len(self.rows)+1,stage=self.stage,action=action,q=u.agent.robot.get_qpos(),qvel=u.agent.robot.get_qvel(),tcp=u.agent.tcp.pose.raw_pose,peg=u.peg.pose.raw_pose,box=u.box.pose.raw_pose,goal=u.goal_pose.raw_pose,effect=u.peg.pose.raw_pose-before,reward=result[1],info=result[-1],terminated=result[2],truncated=result[3])))
        if self.writer:self.writer.append_data(u.render_rgb_array().detach().cpu().numpy()[0])
        return result

def solver(fraction,passes,correction,param,stride=1):
    from mani_skill.examples.motionplanning.panda.solutions import peg_insertion_side as upstream
    from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb
    ns=dict(upstream.__dict__)
    def current_gpu_obb(actor,to_world_frame=True,vis=False):
        box=get_actor_obb(actor,to_world_frame=False,vis=False)
        if to_world_frame:
            import trimesh
            transform=actor.pose.to_transformation_matrix()[0].detach().cpu().numpy().astype(float)
            return trimesh.primitives.Box(extents=np.array(box.primitive.extents,copy=True),transform=transform@np.array(box.primitive.transform))
        return box
    ns['get_actor_obb']=current_gpu_obb
    if stride>1:
        original_planner=upstream.PandaArmMotionPlanningSolver
        class TrackedPlanner(original_planner):
            def follow_path(self,result,refine_steps=0):
                self.env.stage+=1
                positions=result['position'];indices=list(range(0,len(positions),stride))
                if indices[-1]!=len(positions)-1:indices.append(len(positions)-1)
                for index in indices:
                    action=np.hstack([positions[index],self.gripper_state]);output=self.env.step(action);self.elapsed_steps+=1
                # Use actual readback before passing to the next physical stage.
                for _ in range(8):
                    error=float(np.max(np.abs(self.robot.get_qpos()[0,:len(positions[-1])].cpu().numpy()-positions[-1])))
                    if error<.025:break
                    output=self.env.step(np.hstack([positions[-1],self.gripper_state]));self.elapsed_steps+=1
                return output
        ns['PandaArmMotionPlanningSolver']=TrackedPlanner
    text=inspect.getsource(upstream.solve)
    assert 0<fraction<=1 and passes in [0,1,3]
    text=text.replace('joint_vel_limits=0.75','joint_vel_limits='+str(fraction)).replace('joint_acc_limits=0.75','joint_acc_limits='+str(fraction)).replace('range(3)','range('+str(passes)+')')
    # Nominal goal remains untouched for (0,0,0); parameter units m,m,rad.
    lateral_y,lateral_z,yaw=param
    text=text.replace('    insert_pose = env.goal_pose',f'    desired_goal = env.goal_pose * sapien.Pose([0,{lateral_y},{lateral_z}], [np.cos({yaw}/2),0,0,np.sin({yaw}/2)])\n    insert_pose = desired_goal')
    text=text.replace('delta_pose = env.goal_pose','delta_pose = desired_goal')
    if correction:
        text=text.replace('    res = planner.move_to_pose_with_screw(insert_pose * (sapien.Pose([0.05, 0, 0])))','    insert_pose = desired_goal * env.peg.pose.inv() * env.agent.tcp.pose\n    res = planner.move_to_pose_with_screw(insert_pose * (sapien.Pose([0.05, 0, 0])))')
    exec(compile(text,'<documented-peg-controller-only>','exec'),ns)
    return ns['solve'],hashlib.sha256(inspect.getsource(upstream.solve).encode()).hexdigest(),hashlib.sha256(text.encode()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=8);p.add_argument('--seed',type=int,default=110000);p.add_argument('--video',action='store_true');p.add_argument('--iteration2',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    variants=[('official',.75,3,False,[0,0,0]),('budget_tuned',.95,1,False,[0,0,0]),('observed_final',.95,1,True,[0,0,0]),('y_minus',.95,1,True,[-.003,0,0]),('y_plus',.95,1,True,[.003,0,0])]
    if a.iteration2:variants=[('tracked_stride2',.95,1,False,[0,0,0]),('tracked_feedback2',.95,1,True,[0,0,0]),('tracked_feedback3',.95,1,True,[0,0,0])]
    env=Recorded(gym.make('PegInsertionSide-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));rows=[];started=time.monotonic()
    for i in range(a.cases):
        initial_hash=None
        for label,speed,passes,correction,param in variants:
            solve,original,modified=solver(speed,passes,correction,param,stride=3 if label.endswith('3') else (2 if a.iteration2 else 1));failure=None
            if a.video and i<2:env.writer=imageio.get_writer(a.output/f'case{i:03d}_{label}.mp4',fps=20,codec='libx264')
            at=time.monotonic()
            try:
                result=solve(env,seed=a.seed+i,debug=False,vis=False)
                if isinstance(result,int) and result==-1:failure='PLANNER_FAILURE'
            except BudgetEnd:failure='NATIVE_HORIZON'
            except Exception:
                import traceback;failure=traceback.format_exc()
            if env.writer:env.writer.close();env.writer=None
            digest=hashlib.sha256(json.dumps(env.initial,sort_keys=True).encode()).hexdigest()
            if initial_hash is None:initial_hash=digest
            assert initial_hash==digest,'Paired initial states differ'
            # Keep failure and exact consumed budget; do not fabricate missing suffix.
            final=serial(env.unwrapped.evaluate());ever=any(r['info']['success'][0] for r in env.rows)
            summary=dict(seed=a.seed+i,variant=label,requested=param,clipped=param,applied=param,parameter_contract='goal-frame lateral Y/Z [m], yaw Z [rad]; planner fraction<=1',speed_fraction=speed,trajectory_stride=(3 if label.endswith('3') else (2 if a.iteration2 else 1)),refinement_passes=passes,observed_final_correction=correction,steps=len(env.rows),success=final['success'][0],success_ever=ever,terminal=final,failure=failure,wall_s=time.monotonic()-at,initial_sha256=digest,upstream_solver_sha256=original,controller_sha256=modified)
            file=a.output/f'case{i:03d}_{label}.json';file.write_text(json.dumps(dict(summary=summary,initial=env.initial,transitions=env.rows),separators=(',',':')));summary['trajectory_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(summary);print(json.dumps(summary),flush=True)
            (a.output/'results.json').write_text(json.dumps(dict(source_kind='NEW_PHYSICAL_SIMULATION',task='PegInsertionSide-v1',simulator='SAPIEN/PhysX GPU0',native_horizon=100,rows=rows,wall_s=time.monotonic()-started,physical_interactions=sum(r['steps'] for r in rows)),indent=2))
    env.close()

if __name__=='__main__':main()
