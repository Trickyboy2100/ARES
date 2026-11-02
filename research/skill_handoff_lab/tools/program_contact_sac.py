"""Ordinary masked-action critic-LN SAC on real PickTool/contact boundaries.
No novel method claim. Stage0 has 1 effective action, stage1 has2. Entropy masks
unused stage0 coordinate. Remaining budget observed, gamma1 native finite horizon.
"""
import sys,argparse,json,time,gzip,hashlib,random,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'external/sb3-2.7.0'))
import numpy as np,torch,gymnasium as gym
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.policies import ContinuousCritic
from program_sac_policy import CriticLayerNormPolicy
from program_stap_learning import features
import program_post_contact as physical

def mask(obs):return torch.stack([torch.ones_like(obs[:,44]),(obs[:,44]>.5).float()],dim=1)
class MaskedActor(Actor):
 def forward(self,obs,deterministic=False):return super().forward(obs,deterministic)*mask(obs)
 def action_log_prob(self,obs):
  mean,std,kwargs=self.get_action_dist_params(obs);a=self.action_dist.actions_from_params(mean,std,**kwargs);raw=self.action_dist.gaussian_actions
  log=(self.action_dist.distribution.log_prob(raw)-torch.log(1-a*a+self.action_dist.epsilon))*mask(obs)
  return a*mask(obs),log.sum(1)
class MaskedCritic(ContinuousCritic):
 def forward(self,obs,actions):return super().forward(obs,actions*mask(obs))
class BoundaryPolicy(CriticLayerNormPolicy):
 def make_actor(self,features_extractor=None):return MaskedActor(**self._update_features_extractor(self.actor_kwargs,features_extractor)).to(self.device)
 def make_critic(self,features_extractor=None):
  c=MaskedCritic(**self._update_features_extractor(self.critic_kwargs,features_extractor))
  for i,network in enumerate(c.q_networks):
   modules=[];last=max(j for j,m in enumerate(network)if isinstance(m,nn.Linear))
   for j,m in enumerate(network):
    modules.append(m)
    if isinstance(m,nn.Linear)and j!=last:modules.append(nn.LayerNorm(m.out_features))
   replacement=nn.Sequential(*modules);c.add_module(f'qf{i}',replacement);c.q_networks[i]=replacement
  return c.to(self.device)
class BoundarySAC(SAC):
 def _store_transition(self,replay_buffer,buffer_action,new_obs,reward,dones,infos):
  buffer_action=buffer_action.copy();buffer_action[self._last_obs[:,44]<.5,1]=0
  return super()._store_transition(replay_buffer,buffer_action,new_obs,reward,dones,infos)

class ContactEnv(gym.Env):
 def __init__(self,out,seeds,correction='local_yaw',transform=None):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.seeds=seeds;self.counter=0;self.rows=[];self.stage=0;self.transform=transform;self.env=physical.make_env();self.env.correction=correction
  self.action_space=gym.spaces.Box(-1.,1.,(2,),np.float32);self.observation_space=gym.spaces.Box(-np.inf,np.inf,(54,),np.float32)
 def obs(self):
  e=self.env;m=e.measure();event=e.contact_event
  progress=np.zeros(2)if event is None else np.array(m['cube'][:2])-np.array(e.skill_results[0]['measurement']['cube'][:2]);goal=np.zeros(3)if e.pending is None else np.array(e.pending['goal_tool'][:3]);return np.r_[features(m,100-len(e.records),1.,0.,self.stage),m['cube_linear_velocity'],np.log1p(m['tool_cube_force_N'])if self.stage else 0.,progress,goal].astype(np.float32)
 def reset(self,seed=None,options=None):
  super().reset(seed=seed);self.case=self.seeds[self.counter%len(self.seeds)];self.counter+=1;self.stage=0;self.actions=[];self.start=time.monotonic();self.env.contact_event=None;self.env.pending=None
  outer=(random.getstate(),np.random.get_state(),torch.get_rng_state());self.env.reset(seed=self.case);random.setstate(outer[0]);np.random.set_state(outer[1]);torch.set_rng_state(outer[2]);return self.obs(),{}
 def step(self,action):
  e=self.env;action=np.clip(np.asarray(action),-1,1);before=len(e.records);outer=(random.getstate(),np.random.get_state(),torch.get_rng_state());error=None
  try:
   if self.stage==0:
    action=action.copy();action[1]=0.;self.grasp=None if getattr(self,'geometry_grasp_search',False)else .035+.035*(float(action[0])+1);self.actions.append(dict(stage=0,normalized=action.tolist(),step=before));error=physical.prefix(e,self.case,self.grasp)
    if self.grasp is None:
     self.grasp=e.decisions[0]['parameters']['grasp_x_m'];self.actions[0]['normalized']=[(self.grasp-.07)/.035,0.];self.actions[0]['selection']='existing geometric pregrasp lookahead, five legal candidates'
    self.physics_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state());self.stage=1
    if e.contact_event is not None:return self.obs(),0.,False,False,dict(duration_steps=len(e.records)-before)
   else:
    random.setstate(self.physics_rng[0]);np.random.set_state(self.physics_rng[1]);torch.set_rng_state(self.physics_rng[2]);actual=action if self.transform is None else self.transform(self.obs(),action);self.actions.append(dict(stage=1,normalized=action.tolist(),physical_action=np.asarray(actual).tolist(),step=before));error=physical.suffix(e,actual)
   physical.finish(e);row=physical.save(e,self.out,self.case,self.grasp,self.actions[-1]['normalized']if len(self.actions)>1 else [0,0],error,self.start,self.counter);path=Path(row['path']);d=json.load(gzip.open(path,'rt'));d['policy_actions']=self.actions
   with gzip.open(path,'wt')as f:json.dump(d,f)
   row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();row['rl_decisions']=len(self.actions);self.rows.append(row);(self.out/'RESULTS.json').write_text(json.dumps(self.rows,indent=2));print('EP',self.counter,self.case,row['success'],row['contact_step'],error,flush=True)
   return self.obs(),float(row['success']),True,False,dict(duration_steps=len(e.records)-before)
  finally:random.setstate(outer[0]);np.random.set_state(outer[1]);torch.set_rng_state(outer[2])
 def close(self):self.env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--decisions',type=int,default=512);p.add_argument('--seed',type=int,default=31161);p.add_argument('--scene-start',type=int,default=748010);p.add_argument('--scenes',type=int,default=32);p.add_argument('--correction',default='local_yaw');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 for name in ['program_contact_sac.py','program_post_contact.py','program_tool_contract_control.py','program_stap_tool.py']:shutil.copy2(Path(__file__).with_name(name),a.out/name)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},algorithm='ordinary masked critic-LN SAC, not novel',gamma=1.,horizon=100,task='PullCubeTool-v1',entropy='stage0 one dimension, stage1 two',observation='54D current physical state/actual effect/known target and budget; no future label'),indent=2));env=ContactEnv(a.out/'episodes',list(range(a.scene_start,a.scene_start+a.scenes)),a.correction);model=BoundarySAC(BoundaryPolicy,env,learning_rate=3e-4,buffer_size=50000,learning_starts=64,batch_size=64,train_freq=1,gradient_steps=2,ent_coef=.02,gamma=1.,tau=.005,policy_kwargs=dict(net_arch=[128,128]),device='cpu',seed=a.seed);model.save(a.out/'INITIAL');start=time.monotonic()
 try:
  for k in range(128,a.decisions+1,128):
   model.learn(128,reset_num_timesteps=False);model.save(a.out/f'SAC_{k}');model.save_replay_buffer(a.out/f'BUFFER_{k}.pkl');d=dict(decisions=model.num_timesteps,updates=model._n_updates,episodes=len(env.rows),physics_steps=sum(x['steps']for x in env.rows),wall_s=time.monotonic()-start);(a.out/f'TRAINING_{k}.json').write_text(json.dumps(d,indent=2));print('CHECKPOINT',d,flush=True)
 except BaseException as ex:
  model.save(a.out/'INTERRUPTED');model.save_replay_buffer(a.out/'INTERRUPTED_BUFFER.pkl');raise
 finally:
  if env.env.records and len(env.rows)<env.counter:
   with gzip.open(a.out/'PARTIAL.json.gz','wt')as f:json.dump(dict(seed=env.case,initial=env.env.initial,steps=env.env.records,plans=env.env.plans,actions=env.actions),f)
  env.close()
if __name__=='__main__':main()
