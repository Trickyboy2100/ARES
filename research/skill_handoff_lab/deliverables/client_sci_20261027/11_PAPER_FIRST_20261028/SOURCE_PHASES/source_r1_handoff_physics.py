"""Two physically verified option boundaries; native Panda/StackPyramid physics.

Every transition is real env.step. No state setters, object teleport, reset between
skills, cached outcomes, or future-success actor features are used.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import sapien
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb,compute_grasp_info_by_obb
from transforms3d.quaternions import quat2mat
from method_stack_physics import serial

class BudgetEnd(Exception):pass
class SkillFailure(Exception):pass

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npv(value):return value.detach().cpu().numpy()[0]
def obb(actor):
    import trimesh
    box=get_actor_obb(actor,to_world_frame=False,vis=False)
    return trimesh.primitives.Box(extents=np.array(box.primitive.extents,copy=True),transform=npv(actor.pose.to_transformation_matrix()).astype(float)@np.array(box.primitive.transform))

class NativeRecorder(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env);self.rows=[];self.stage=0;self.label='RESET';self.writer=None;self.trace=True
    def reset(self,**kw):
        result=self.env.reset(**kw);self.rows=[];self.steps=0;self.stage=0;self.label='RESET';self.initial=serial(self.unwrapped.get_state_dict());self.initial_sha=hashlib.sha256(json.dumps(self.initial,sort_keys=True).encode()).hexdigest();return result
    def step(self,action):
        if self.steps>=250:raise BudgetEnd()
        action=np.asarray(action);assert np.isfinite(action).all();result=self.env.step(action);self.steps+=1
        if self.trace:self.rows.append(serial(dict(step=self.steps,stage=self.stage,segment=self.label,action=action,q=self.unwrapped.agent.robot.get_qpos(),qvel=self.unwrapped.agent.robot.get_qvel(),tcp=self.unwrapped.agent.tcp.pose.raw_pose,state=self.unwrapped.get_state_dict(),info=result[-1])))
        if self.writer:
            im=self.unwrapped.render_rgb_array().detach().cpu().numpy()[0];self.writer.append_data(im)
        return result

class TrackedPlanner(PandaArmMotionPlanningSolver):
    def follow_path(self,result,refine_steps=0):
        positions=result['position'];indices=list(range(0,len(positions),3))
        if indices[-1]!=len(positions)-1:indices.append(len(positions)-1)
        for index in indices:output=self.env.step(np.hstack([positions[index],self.gripper_state]));self.elapsed_steps+=1
        for _ in range(6):
            if np.max(abs(npv(self.robot.get_qpos())[:len(positions[-1])]-positions[-1]))<.025:break
            output=self.env.step(np.hstack([positions[-1],self.gripper_state]));self.elapsed_steps+=1
        return output

class HandoffExecutor:
    def __init__(self,trace=True):
        self.env=NativeRecorder(gym.make('StackPyramid-v1',num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));self.env.trace=trace;self.u=self.env.unwrapped
    def reset(self,seed):
        self.env.reset(seed=seed);self.seed=seed;self.planner=TrackedPlanner(self.env,vis=False,debug=False,base_pose=self.u.agent.robot.pose,visualize_target_grasp_pose=False,print_env_info=False);self.plans=[];self.boundaries=[];self.stage=0;self.start_A=npv(self.u.cubeA.pose.raw_pose).copy();self.start_B=npv(self.u.cubeB.pose.raw_pose).copy();self.start_C=npv(self.u.cubeC.pose.raw_pose).copy();return self.observe()
    def observe(self):
        u=self.u;A=npv(u.cubeA.pose.raw_pose);B=npv(u.cubeB.pose.raw_pose);C=npv(u.cubeC.pose.raw_pose);tcp=npv(u.agent.tcp.pose.raw_pose);q=npv(u.agent.robot.get_qpos());v=npv(u.agent.robot.get_qvel());axis=A[:2]-B[:2];dist=np.linalg.norm(axis);axis=axis/max(dist,1e-6)
        geom=np.r_[A[:3]-B[:3],C[:3]-(A[:3]+B[:3])/2,tcp[:3]-C[:3],A[3:],B[3:],C[3:],q,v]
        effect=np.r_[A[:3]-self.start_A[:3],B[:3]-self.start_B[:3],C[:3]-self.start_C[:3]]
        contract=np.r_[dist,.061568-dist,axis, .02,.01,.5,(250-u.elapsed_steps.item())/250]
        return np.r_[geom,effect,contract,float(self.stage)].astype(np.float32)
    def move(self,pose,label):
        self.env.label=label;t=time.monotonic();result=self.planner.move_to_pose_with_screw(pose,dry_run=True);wall=time.monotonic()-t
        if isinstance(result,int):raise SkillFailure('PLANNER_'+label)
        self.plans.append(dict(stage=self.stage,label=label,start_q=serial(self.u.agent.robot.get_qpos()),target_p=pose.p.tolist(),target_q=pose.q.tolist(),position=result['position'].tolist(),native_time=result.get('time',np.arange(len(result['position']))*.05).tolist(),planning_wall_s=wall))
        self.planner.follow_path(result)
    def hold(self,n,label):
        self.env.label=label;q=npv(self.u.agent.robot.get_qpos())[:7]
        for _ in range(n):self.env.step(np.r_[q,self.planner.gripper_state])
    def grasp(self,actor,name):
        tcp=npv(self.u.agent.tcp.pose.to_transformation_matrix());g=compute_grasp_info_by_obb(obb(actor),approaching=np.array([0,0,-1]),target_closing=tcp[:3,1],depth=.025)
        pose=self.u.agent.build_grasp_pose(np.array([0,0,-1]),g['closing'],g['center'])
        self.move(pose*sapien.Pose([0,0,-.06]),name+'_REACH');self.move(pose,name+'_DESCEND');self.env.label=name+'_CLOSE';self.planner.close_gripper(t=6)
        if not bool(self.u.agent.is_grasping(actor).item()):raise SkillFailure(name+'_NOT_GRASPED')
        lift=sapien.Pose(pose.p+[0,0,.10],pose.q);self.move(lift,name+'_LIFT')
        if not bool(self.u.agent.is_grasping(actor).item()):raise SkillFailure(name+'_LOST_ON_LIFT')
        return pose
    def verification(self,actor,goal,local):
        A,B,C=[npv(x.pose.raw_pose) for x in [self.u.cubeA,self.u.cubeB,self.u.cubeC]];opening=float(npv(self.u.agent.robot.get_qpos())[-2:].sum());not_grasped=not bool(self.u.agent.is_grasping(actor).item());static=all(bool(x.is_static(lin_thresh=.01,ang_thresh=.5).item()) for x in [self.u.cubeA,self.u.cubeB,self.u.cubeC]);robot_static=bool(np.max(abs(npv(self.u.agent.robot.get_qvel())[:7]))<.15);p=npv(actor.pose.p);on_table=abs(p[2]-.02)<.008;target_ok=np.linalg.norm(p[:2]-goal[:2])<.015
        valid=opening>.065 and not_grasped and static and robot_static and (on_table and target_ok if local else True)
        return dict(valid=bool(valid),finger_opening_m=opening,not_grasped=not_grasped,objects_static=static,robot_static=robot_static,on_table=bool(on_table),target_xy_error_m=float(np.linalg.norm(p[:2]-goal[:2])),AB_distance_m=float(np.linalg.norm(A[:2]-B[:2])),support_distance_margin_m=float(.061568-np.linalg.norm(A[:2]-B[:2])),A=A.tolist(),B=B.tolist(),C=C.tolist(),native_success=bool(self.u.evaluate()['success'].item()),step=self.env.steps)
    def release_verify(self,actor,goal,pose,name,local):
        self.env.label=name+'_OPEN';self.planner.open_gripper(t=6);retreat=sapien.Pose(pose.p+[0,0,.06],pose.q);self.move(retreat,name+'_RETREAT')
        consecutive=0;result=None
        for _ in range(12):
            self.hold(1,name+'_SETTLE');result=self.verification(actor,goal,local);consecutive=consecutive+1 if result['valid'] else 0
            if consecutive>=3:break
        result['consecutive_valid_steps']=consecutive;result['valid']=consecutive>=3;self.boundaries.append(result)
        if not result['valid']:raise SkillFailure(name+'_BOUNDARY_NOT_VERIFIED')
    def execute(self,action):
        request=np.asarray(action,dtype=float);assert request.shape==(1,);clipped=np.clip(request,-1,1);start=self.env.steps;wall=time.monotonic();self.env.stage=self.stage;u=self.u
        if self.stage==0:
            pose=self.grasp(u.cubeA,'A');A=npv(u.cubeA.pose.raw_pose);B=npv(u.cubeB.pose.raw_pose);radius=.02*(abs(quat2mat(A[3:])[0,:]).sum()+abs(quat2mat(B[3:])[0,:]).sum());gap=radius+.002+clipped[0]*.006;goal=B[:3].copy()+[gap,0,0];goal[2]=.022
            offset=npv(u.agent.tcp.pose.p)-npv(u.cubeA.pose.p);above=goal+offset+[0,0,.09];self.move(sapien.Pose(above,pose.q),'A_PREPLACE');place=sapien.Pose(goal+offset,pose.q);self.move(place,'A_PLACE');self.release_verify(u.cubeA,goal,place,'A',True)
        else:
            pose=self.grasp(u.cubeC,'C');A=npv(u.cubeA.pose.raw_pose);B=npv(u.cubeB.pose.raw_pose);axis=A[:2]-B[:2];axis/=max(np.linalg.norm(axis),1e-6);goal=(A[:3]+B[:3])/2;goal[:2]+=axis*clipped[0]*.008;goal[2]=max(A[2],B[2])+.043
            offset=npv(u.agent.tcp.pose.p)-npv(u.cubeC.pose.p);self.move(sapien.Pose(goal+offset+[0,0,.07],pose.q),'C_PREPLACE');place=sapien.Pose(goal+offset,pose.q);self.move(place,'C_PLACE');self.release_verify(u.cubeC,goal,place,'C',False)
        record=dict(stage=self.stage,request=request.tolist(),clipped=clipped.tolist(),applied_m=float(clipped[0]*(.006 if self.stage==0 else .008)),goal=goal.tolist(),physical_steps=self.env.steps-start,wall_s=time.monotonic()-wall,skill_result=self.boundaries[-1]);self.stage+=1;return self.observe(),record
    def rollout(self,seed,actions):
        start=time.monotonic();self.reset(seed);records=[];error=None
        try:
            for a in actions:
                _,r=self.execute([a]);records.append(r)
        except (BudgetEnd,SkillFailure) as e:error=str(e) or 'NATIVE_HORIZON'
        success=bool(self.u.evaluate()['success'].item());return dict(seed=seed,initial_sha256=self.env.initial_sha,steps=self.env.steps,success=success,error=error,boundaries=self.boundaries,options=records,wall_s=time.monotonic()-start,planning_wall_s=sum(p['planning_wall_s'] for p in self.plans),planner_calls=len(self.plans))
    def close(self):self.env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=4);p.add_argument('--seed',type=int,default=410000);p.add_argument('--video',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);ex=HandoffExecutor();rows=[]
    for i in range(a.cases):
        digest=None
        for label,actions in [('nominal',[0,0]),('A_minus',[-1,0]),('A_plus',[1,0]),('C_minus',[0,-1]),('C_plus',[0,1])]:
            if a.video and i<2:
                import imageio.v2 as imageio
                ex.env.writer=imageio.get_writer(a.output/f'{i:03d}_{label}.mp4',fps=20,codec='libx264')
            row=ex.rollout(a.seed+i,actions)
            if ex.env.writer:ex.env.writer.close();ex.env.writer=None
            if digest is None:digest=row['initial_sha256']
            assert digest==row['initial_sha256'];row['policy']=label
            file=a.output/f'{i:03d}_{label}.json';file.write_text(json.dumps(dict(summary=row,initial=ex.env.initial,transitions=ex.env.rows,plans=ex.plans),separators=(',',':')));row['trajectory_sha256']=sha(file);rows.append(row);print(json.dumps({k:row[k] for k in ['seed','policy','success','steps','error','wall_s','planner_calls']}),flush=True)
            (a.output/'RESULTS.json').write_text(json.dumps(dict(rows=rows,physical_interactions=sum(r['steps'] for r in rows),native_horizon=250,task='StackPyramid-v1',scope='NEW_DESIGN_PHYSICS_NOT_TEST'),indent=2)+'\n')
    ex.close()
if __name__=='__main__':main()
