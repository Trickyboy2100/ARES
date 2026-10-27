"""Native Stack skill parameter authority, measured effects and matched PPO interface.

State-based experiment: simulator poses are explicit privileged state baseline.
Never claims RGB deployment or changes native success, reward or 50-step budget.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs

def serial(value):
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,(torch.Tensor,np.ndarray)):return value.detach().cpu().tolist() if isinstance(value,torch.Tensor) else value.tolist()
    if isinstance(value,(np.integer,np.floating)):return value.item()
    return value

class ParameterStack(gym.Wrapper):
    """Three legal parameters: place XY offset ±15mm; place height ±12mm.

    Parameters only affect place phases 4/5/6. phase/dwell logic is identical
    to the prior strong feedback baseline. Both PPO and selectors share this.
    """
    def __init__(self,env,mode='parameter'):
        super().__init__(env);self.mode=mode;u=self.unwrapped;self.phase=torch.zeros(u.num_envs,dtype=torch.long,device=u.device);self.dwell=torch.zeros_like(self.phase)
        self.pickup=u.cubeA.pose.p.clone();self.previous=u.cubeA.pose.p.clone();self.effect=torch.zeros_like(self.previous)
        self.applied=torch.zeros((u.num_envs,3),device=u.device)
        self.single_action_space=gym.spaces.Box(-1,1,(4 if mode=='residual' else 3,),np.float32);self.action_space=gym.vector.utils.batch_space(self.single_action_space,u.num_envs)
        n=env.observation_space.shape[-1]+8+3+3+1
        self.single_observation_space=gym.spaces.Box(-np.inf,np.inf,(n,),np.float32);self.observation_space=gym.vector.utils.batch_space(self.single_observation_space,u.num_envs)
    def observe(self,obs):
        u=self.unwrapped;remaining=(50-u.elapsed_steps).float().unsqueeze(-1)/50
        return torch.cat((obs,torch.nn.functional.one_hot(self.phase,8).float(),self.effect,self.applied,remaining),-1)
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);idx=(kwargs.get('options') or {}).get('env_idx',slice(None));self.phase[idx]=0;self.dwell[idx]=0;self.pickup[idx]=self.unwrapped.cubeA.pose.p[idx];self.previous[idx]=self.pickup[idx];self.effect[idx]=0;self.applied[idx]=0
        return self.observe(obs),info
    def step(self,action):
        u=self.unwrapped;tcp,cube,base=u.agent.tcp.pose.p,u.cubeA.pose.p,u.cubeB.pose.p;p,d=self.phase,self.dwell
        requested=torch.as_tensor(action,device=u.device);clipped=requested.clamp(-1,1);parameters=clipped[:,:3]*torch.tensor([.015,.015,.012],device=u.device)
        if self.mode=='residual':parameters=torch.zeros_like(parameters)
        target=cube.clone();target[:,2]+=.08
        mask=(p==1)|(p==2);target[mask]=self.pickup[mask];target[mask,2]+=.003
        mask=p==3;target[mask]=self.pickup[mask];target[mask,2]=.14
        mask=p>=4;target[mask]=base[mask];target[mask,2]=.14
        target[mask,:2]+=parameters[mask,:2]
        mask=(p==5)|(p==6);target[mask,2]=base[mask,2]+.043+parameters[mask,2]
        self.applied=torch.where((p>=4).unsqueeze(-1),parameters,torch.zeros_like(parameters))
        error=target-tcp;command=torch.zeros((u.num_envs,4),device=u.device);command[:,:3]=(error/.1).clamp(-.6,.6);command[:,3]=torch.where((p>=2)&(p<=5),-1.,1.)
        if self.mode=='residual':
            command=(command+.15*clipped).clamp(-1,1)
            self.applied=.015*clipped[:,:3]
        near=torch.linalg.vector_norm(error,dim=1)<.012
        advance=((p==0)&near)|((p==1)&near)|((p==2)&(d>=6))|((p==3)&near&(cube[:,2]>.07))|((p==4)&near)|((p==5)&near)|((p==6)&(d>=6))
        phase_before=p.clone();self.dwell+=1;self.phase[advance]=(self.phase[advance]+1).clamp_max(7);self.dwell[advance]=0
        before=cube.clone();obs,reward,terminated,truncated,info=self.env.step(command);self.effect=u.cubeA.pose.p-before
        info['authority']={'requested':requested.detach().clone(),'clipped':clipped.detach().clone(),'applied_m':self.applied.clone(),'native_command':command.clone(),'phase_before':phase_before,'effect_m':self.effect.clone(),'target_m':target.clone()}
        return self.observe(obs),reward,terminated,truncated,info

def make(n):
    return ParameterStack(gym.make('StackCube-v1',num_envs=n,obs_mode='state',control_mode='pd_ee_delta_pos',sim_backend='physx_cuda',render_backend='gpu',render_mode='rgb_array',reconfiguration_freq=1))

def probe(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);env=make(args.num_envs);rows=[];start=time.monotonic()
    variants={'nominal':[0,0,0],'x_minus':[-1,0,0],'x_plus':[1,0,0],'y_minus':[0,-1,0],'y_plus':[0,1,0],'height_minus':[0,0,-1],'height_plus':[0,0,1]}
    for batch in range(args.cases//args.num_envs):
        seeds=list(range(args.seed+batch*args.num_envs,args.seed+(batch+1)*args.num_envs));initial_hash=None
        for label,param in variants.items():
            obs,_=env.reset(seed=seeds);u=env.unwrapped;initial=serial(u.get_state_dict());digest=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest()
            if initial_hash is None:initial_hash=digest
            assert digest==initial_hash,'Paired reset differs'
            states=[];ever=torch.zeros(args.num_envs,dtype=torch.bool,device=u.device)
            for step in range(50):
                obs,reward,term,trunc,info=env.step(torch.tensor(param,device=u.device).expand(args.num_envs,3));ever|=info['success'];states.append(serial(dict(step=step+1,cubeA=u.cubeA.pose.raw_pose,cubeB=u.cubeB.pose.raw_pose,q=u.agent.robot.get_qpos(),tcp=u.agent.tcp.pose.raw_pose,reward=reward,success=info['success'],authority=info['authority'])))
            final=serial(u.evaluate());entry=dict(variant=label,seeds=seeds,initial_sha256=digest,success=final['success'],success_ever=serial(ever),terminal_cube=serial(u.cubeA.pose.raw_pose),steps=50)
            trajectory=out/f'{batch:03d}_{label}.json';trajectory.write_text(json.dumps(dict(summary=entry,initial=initial,transitions=states),separators=(',',':')));entry['trajectory_sha256']=hashlib.sha256(trajectory.read_bytes()).hexdigest();rows.append(entry);print(json.dumps(entry),flush=True)
        (out/'results.json').write_text(json.dumps(dict(task='StackCube-v1',simulator='SAPIEN/PhysX GPU0',source_kind='NEW_PHYSICAL_SIMULATION',rows=rows,physical_interactions=len(rows)*args.num_envs*50,native_horizon=50,wall_s=time.monotonic()-start),indent=2))
    env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=int,default=64);p.add_argument('--num-envs',type=int,default=16);p.add_argument('--seed',type=int,default=100000);a=p.parse_args();assert a.cases%a.num_envs==0;probe(a)
