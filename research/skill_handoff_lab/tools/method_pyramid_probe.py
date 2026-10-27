"""Native 250-step, sparse-reward StackPyramid multi-stage authority probe.

First place A beside B; then place C over their observed midpoint. Every
object movement is produced by native EE/gripper actions and PhysX contacts.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from method_stack_physics import serial

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=16);p.add_argument('--num-envs',type=int,default=8);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    env=gym.make('StackPyramid-v1',num_envs=a.num_envs,obs_mode='state',control_mode='pd_ee_delta_pos',sim_backend='physx_cuda',render_backend='gpu',render_mode='rgb_array',reward_mode='sparse',reconfiguration_freq=1);u=env.unwrapped;rows=[]
    variants={'fixed_gap45':[.045,0.],'geometry':[0.,0.],'gap_minus':[-.008,0.],'gap_plus':[.008,0.],'top_lower':[0.,-.012]}
    for batch in range(a.cases//a.num_envs):
        seeds=list(range(120000+batch*a.num_envs,120000+(batch+1)*a.num_envs));reference=None
        for label,param in variants.items():
            env.reset(seed=seeds);initial=serial(u.get_state_dict());digest=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
            if reference is None:reference=digest
            assert reference==digest
            phase=torch.zeros(a.num_envs,dtype=torch.long,device=u.device);dwell=torch.zeros_like(phase);pickup=u.cubeA.pose.p.clone();states=[]
            for step in range(250):
                second=phase>=8;local=phase%8;cube=torch.where(second[:,None],u.cubeC.pose.p,u.cubeA.pose.p);base=u.cubeB.pose.p;tcp=u.agent.tcp.pose.p
                rotation_a=quaternion_to_matrix(u.cubeA.pose.q);rotation_b=quaternion_to_matrix(u.cubeB.pose.q)
                required_gap=.02*(rotation_a[:,0,:].abs().sum(-1)+rotation_b[:,0,:].abs().sum(-1))+.001
                gap=torch.full_like(required_gap,.045) if label=='fixed_gap45' else (required_gap+param[0]).clamp(.035,.060)
                goal_a=base.clone();goal_a[:,0]+=gap;goal_a[:,2]=.02
                goal_c=(u.cubeA.pose.p+u.cubeB.pose.p)/2;goal_c[:,2]+=.04+param[1]
                goal=torch.where(second[:,None],goal_c,goal_a)
                target=cube.clone();target[:,2]+=.08
                mask=(local==1)|(local==2);target[mask]=pickup[mask];target[mask,2]+=.003
                mask=local==3;target[mask]=pickup[mask];target[mask,2]=.16
                mask=local>=4;target[mask]=goal[mask];target[mask,2]=.16
                mask=(local==5)|(local==6);target[mask]=goal[mask];target[mask,2]+=.003
                error=target-tcp;command=torch.zeros((a.num_envs,4),device=u.device);command[:,:3]=(error/.1).clamp(-.6,.6);command[:,3]=torch.where((local>=2)&(local<=5),-1.,1.)
                near=torch.linalg.vector_norm(error,dim=1)<.012
                advance=((local==0)&near)|((local==1)&near)|((local==2)&(dwell>=6))|((local==3)&near&(cube[:,2]>.07))|((local==4)&near)|((local==5)&near)|((local==6)&(dwell>=6))|((phase==7)&near)
                before=phase.clone();dwell+=1;phase[advance]=(phase[advance]+1).clamp_max(15);dwell[advance]=0
                switch=(before==7)&(phase==8);pickup[switch]=u.cubeC.pose.p[switch]
                obs,reward,term,trunc,info=env.step(command)
                states.append(serial(dict(step=step+1,phase=before,gap_m=gap,target_m=target,action=command,q=u.agent.robot.get_qpos(),tcp=u.agent.tcp.pose.raw_pose,cubeA=u.cubeA.pose.raw_pose,cubeB=u.cubeB.pose.raw_pose,cubeC=u.cubeC.pose.raw_pose,success=info['success'],reward=reward)))
            result=dict(policy=label,seeds=seeds,parameter=param,initial_sha256=digest,success=serial(u.evaluate()['success']),steps=250,final_phase=serial(phase));file=a.output/f'{label}_{batch:03d}.json';file.write_text(json.dumps(dict(summary=result,initial=initial,transitions=states),separators=(',',':')));result['trajectory_sha256']=hashlib.sha256(file.read_bytes()).hexdigest();rows.append(result);print(json.dumps(result),flush=True)
            (a.output/'RESULTS.json').write_text(json.dumps(dict(source_kind='NEW_PHYSICAL_SIMULATION',task='StackPyramid-v1',native_horizon=250,reward_mode='sparse unmodified',rows=rows,physical_interactions=len(rows)*a.num_envs*250,scope='MULTISTAGE_AUTHORITY_SCOUT_NOT_RL_GAIN',sealed_test_used=False),indent=2)+'\n')
    env.close()

if __name__=='__main__':main()
