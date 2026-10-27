"""Cold-reset native strong planners, with only current GPU pose observation fix."""
import argparse,hashlib,inspect,json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs
from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS
from mani_skill.examples.motionplanning.base_motionplanner.utils import get_actor_obb
from mani_skill.utils import gym_utils
from method_stack_physics import serial

class End(Exception):pass
class Recorder(gym.Wrapper):
    def reset(self,**kw):
        result=self.env.reset(**kw);self.initial=serial(self.unwrapped.get_state_dict());self.rows=[];self.first_contract_active=float(torch.linalg.vector_norm(self.unwrapped.cubeA.pose.p-self.unwrapped.cubeB.pose.p,dim=1)[0])>.07 if hasattr(self.unwrapped,'cubeA') else False;return result
    def step(self,action):
        if len(self.rows)>=self.horizon:raise End()
        result=self.env.step(action);self.rows.append(serial(dict(step=len(self.rows)+1,action=action,state=self.unwrapped.get_state_dict(),q=self.unwrapped.agent.robot.get_qpos(),qvel=self.unwrapped.agent.robot.get_qvel(),tcp=self.unwrapped.agent.tcp.pose.raw_pose,info=result[-1])))
        if self.writer:
            im=self.env.render().detach().cpu().numpy();self.writer.append_data(im[0] if im.ndim==4 else im)
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--task',default='StackPyramid-v1');p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=4);p.add_argument('--seed',type=int,default=121000);p.add_argument('--video',action='store_true');p.add_argument('--parameter-probe',action='store_true');p.add_argument('--release-first',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    env=Recorder(gym.make(a.task,num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));env.horizon=gym_utils.find_max_episode_steps_value(env);env.writer=None;rows=[]
    original=MP_SOLUTIONS[a.task];original_text=inspect.getsource(original)
    ns=dict(original.__globals__)
    def current_obb(actor,to_world_frame=True,vis=False):
        box=get_actor_obb(actor,to_world_frame=False,vis=False)
        if not to_world_frame:return box
        import trimesh
        transform=actor.pose.to_transformation_matrix()[0].detach().cpu().numpy().astype(float)
        return trimesh.primitives.Box(extents=np.array(box.primitive.extents,copy=True),transform=transform@np.array(box.primitive.transform))
    ns['get_actor_obb']=current_obb
    for i in range(a.cases):
        digest=None
        variants=[('native_planner',1,0.),('tracked_planner2',2,0.)] if not a.parameter_probe else [('tracked_nominal',2,0.),('first_dx_minus',2,-.012),('first_dx_plus',2,.012)]
        for label,stride,parameter in variants:
            if a.release_first:label=label.replace('tracked_nominal','released_nominal').replace('first_dx','released_dx')
            local=dict(ns)
            if stride>1:
                base=ns['PandaArmMotionPlanningSolver']
                class Tracked(base):
                    def follow_path(self,result,refine_steps=0):
                        positions=result['position'];indices=list(range(0,len(positions),2))
                        if indices[-1]!=len(positions)-1:indices.append(len(positions)-1)
                        for index in indices:output=self.env.step(np.hstack([positions[index],self.gripper_state]));self.elapsed_steps+=1
                        for _ in range(6):
                            error=float(np.max(abs(self.robot.get_qpos()[0,:len(positions[-1])].cpu().numpy()-positions[-1])))
                            if error<.025:break
                            output=self.env.step(np.hstack([positions[-1],self.gripper_state]));self.elapsed_steps+=1
                        return output
                local['PandaArmMotionPlanningSolver']=Tracked
            text=original_text
            if a.parameter_probe:
                assert a.task=='StackPyramid-v1'
                assert text.count('target_cube.pose.sp.p * 0.8')==1
                text=text.replace('target_cube.pose.sp.p * 0.8','target_cube.pose.sp.p * 0.8 + np.array(['+str(parameter)+',0,0])')
            if a.release_first:
                token='        planner.move_to_pose_with_screw(goal_pose)'
                assert text.count(token)==1
                text=text.replace(token,token+'\n        planner.open_gripper()')
            exec(compile(text,'<native-planner-gpu-pose-adapter>','exec'),local)
            if a.video and i<2:env.writer=imageio.get_writer(a.output/f'{i:03d}_{label}.mp4',fps=20,codec='libx264')
            error=None
            try:local['solve'](env,seed=a.seed+i,debug=False,vis=False)
            except End:error='NATIVE_HORIZON'
            except Exception:
                import traceback;error=traceback.format_exc()
            if env.writer:env.writer.close();env.writer=None
            initial_sha=hashlib.sha256(json.dumps(env.initial,sort_keys=True).encode()).hexdigest()
            if digest is None:digest=initial_sha
            assert digest==initial_sha
            final=serial(env.unwrapped.evaluate());row=dict(release_first=a.release_first,modified_solver_sha256=hashlib.sha256(text.encode()).hexdigest(),first_parameter_requested_m=parameter,first_parameter_applied_m=parameter if env.first_contract_active else 0.,first_contract_active=env.first_contract_active,task=a.task,seed=a.seed+i,policy=label,success=final['success'][0],steps=len(env.rows),initial_sha256=digest,error=error,upstream_solver_sha256=hashlib.sha256(original_text.encode()).hexdigest(),native_horizon=env.horizon)
            file=a.output/f'{i:03d}_{label}.json';file.write_text(json.dumps(dict(summary=row,initial=env.initial,transitions=env.rows),separators=(',',':')));row['trajectory_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(row);print(json.dumps(row),flush=True)
            (a.output/'RESULTS.json').write_text(json.dumps(dict(rows=rows,task=a.task,source_kind='NEW_PHYSICAL_SIMULATION',physical_interactions=sum(r['steps'] for r in rows),native_horizon=env.horizon),indent=2)+'\n')
    env.close()

if __name__=='__main__':main()
