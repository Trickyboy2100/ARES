"""Ordinary critic-LN SAC on two actual physical skill boundaries; no novelty claim.
Gamma=1 finite native100 horizon with remaining budget observed. One scalar action
per option, actual duration recorded. No mid-episode state reset or simulated labels.
"""
import argparse,json,gzip,hashlib,time,random,shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch,gymnasium as gym
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'external/sb3-2.7.0'))
from stable_baselines3 import SAC
from program_sac_policy import CriticLayerNormPolicy
import program_tool_contract_control as control
from program_stap_learning import features
run=control.run;task=run.task
class SkillEnv(gym.Env):
 def __init__(self,out,seeds,mass=1.,mode='corrected_feedback',parameter='height'):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.seeds=seeds;self.mass=mass;self.mode=mode;self.parameter=parameter;self.counter=0;self.rows=[];self.flow=None
  self.action_space=gym.spaces.Box(-1.,1.,(1,),np.float32);self.observation_space=gym.spaces.Box(-np.inf,np.inf,(45,),np.float32)
  self.env=run.ContactRecorder(gym.make('ARES-ToolLoad-v0',density_scale=mass,num_envs=1,obs_mode='state',control_mode='pd_joint_pos',reward_mode='sparse',render_mode='rgb_array',sim_backend='physx_cuda',render_backend='gpu',reconfiguration_freq=1));e=self.env;e.allow_unstable=False;e.stability_max=10;e.acceleration=6.;e.target_y=None;e.target_feedback=False;e.parameter_selector=SimpleNamespace(method='ordinary_smdp_sac');e.effect_mode=mode;e.expand_geometric_yaw=mode=='corrected_wide';e.recompute_pull=mode in ['corrected_feedback','corrected_wide'];e.pull_limit=.36 if e.recompute_pull else .29
  task.TracePlanner=control.ContractPlanner
 def advance(self,value=None,first=False):
  outer=(random.getstate(),np.random.get_state(),torch.get_rng_state())
  try:
   if first:random.seed(self.case);np.random.seed(self.case);torch.manual_seed(self.case)
   else:random.setstate(self.physics_rng[0]);np.random.set_state(self.physics_rng[1]);torch.set_rng_state(self.physics_rng[2])
   return next(self.flow) if first else self.flow.send(value)
  finally:
   self.physics_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state())
   random.setstate(outer[0]);np.random.set_state(outer[1]);torch.set_rng_state(outer[2])
 def obs(self):return features(self.env.measure(),100-len(self.env.records),self.mass,0,self.stage)
 def reset(self,seed=None,options=None):
  super().reset(seed=seed)
  if self.flow is not None:self.flow.close()
  self.case=self.seeds[self.counter%len(self.seeds)];self.counter+=1;self.actions=[];self.start=time.monotonic();self.env.hook_height=.026;self.env.successor_accel=None;self.env.stability=None
  self.flow=task.execute_chain_steps(self.env,self.case,grasp_x=None,hook_y=None,transfer_stride=1,pull_policy='goal');self.stage=self.advance(first=True);assert self.stage==0
  return self.obs(),{}
 def step(self,action):
  value=float(np.clip(action[0],-1,1));before=len(self.env.records);self.actions.append(dict(stage=self.stage,step=before,normalized=value))
  arg=.035+.035*(value+1) if self.stage==0 else None
  if self.stage==1:
   if self.parameter=='accel':self.env.successor_accel=3.+1.5*(value+1)
   else:self.env.hook_height=.026+.006*(value+1)
  error=None;done=False
  try:self.stage=self.advance(arg)
  except StopIteration:done=True
  except (run.Failure,run.Horizon) as exc:error=repr(exc);done=True
  if not done:return self.obs(),0.,False,False,dict(duration_steps=len(self.env.records)-before)
  self.env.phase='TERMINAL_HOLD';q=np.asarray(run.serial(self.env.unwrapped.agent.robot.get_qpos()[0,:7]))
  while len(self.env.records)<100:self.env.step(np.r_[q,-1.])
  final=self.env.records[-1];self.stage=1;success=bool(final['native_success']);path=self.out/f'{self.counter:05d}_{self.case}.json.gz';r=dict(seed=self.case,mass=self.mass,success=success,error=error,steps=100,initial_sha256=self.env.initial_sha,grasp=.035+.035*(self.actions[0]['normalized']+1),mode=self.mode,hook_height=self.env.hook_height,successor_acceleration=getattr(self.env,'successor_accel',None),parameter=self.parameter,wall_s=time.monotonic()-self.start,rl_decisions=len(self.actions))
  with gzip.open(path,'wt')as f:json.dump(dict(summary=r,initial=self.env.initial,initial_measure=self.env.initial_measure,steps=self.env.records,plans=self.env.plans,decisions=self.env.decisions,skill_results=self.env.skill_results,policy_actions=self.actions),f)
  r.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest());self.rows.append(r);(self.out/'RESULTS.json').write_text(json.dumps(self.rows,indent=2));print('EP',self.counter,self.case,success,error,r['grasp'],r['hook_height'],flush=True)
  return self.obs(),float(success),True,False,dict(duration_steps=100-before,success=success)
 def close(self):
  if self.flow is not None:self.flow.close()
  self.env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--decisions',type=int,default=1024);p.add_argument('--seed',type=int,default=31141);p.add_argument('--mode',default='corrected_feedback');p.add_argument('--grid',action='store_true');p.add_argument('--parameter',choices=['height','accel'],default='height');p.add_argument('--warm',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--resume-buffer',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 sources=['program_tool_smdp_sac.py','program_tool_contract_control.py','program_stap_tool.py','program_tool_all_contacts.py','program_tool_effect_probe.py','program_sac_policy.py','program_stap_learning.py','pull_tool_physics.py','program_multiskill_contact_probe.py','program_multiskill_pull.py']
 for n in sources:shutil.copy2(Path(__file__).with_name(n),a.out/n)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(args={k:str(v) if isinstance(v,Path)else v for k,v in vars(a).items()},task='ARES-ToolLoad-v0 native100',actions='stage0 grasp35..105mm; stage1 '+('successor acceleration3..6 rad/s2' if a.parameter=='accel' else 'hook height26..38mm')+'; common corrected control',observation='45D same actual state, known density and remaining budget; no future success',gamma=1.,reward='terminal native success only',sources={n:hashlib.sha256((a.out/n).read_bytes()).hexdigest()for n in sources}),indent=2))
 env=SkillEnv(a.out/'episodes',list(range(746000,746064)) if not a.grid else [s for s in range(746000,746004) for _ in range(9)],mode=a.mode,parameter=a.parameter)
 try:
  if a.grid:
   for i in range(36):
    o,_=env.reset();o,r,done,_,_=env.step(np.array([[-1,0,1][i//3%3]]))
    if not done:env.step(np.array([[-1,0,1][i%3]]))
   return
  model=SAC(CriticLayerNormPolicy,env,policy_kwargs=dict(net_arch=[128,128]),learning_rate=3e-4,buffer_size=50000,learning_starts=64,batch_size=64,train_freq=1,gradient_steps=2,ent_coef=.02,gamma=1.,tau=.005,seed=a.seed,device='cpu',verbose=0);
  if a.resume:
   model=SAC.load(a.resume,env=env,device='cpu',force_reset=True);model.load_replay_buffer(a.resume_buffer)
  model.save(a.out/'INITIAL');start=time.monotonic()
  if a.warm:
   for r in json.loads((a.warm/'RESULTS.json').read_text()):
    path=Path(r['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(path,'rt'));actions=d['policy_actions'];decs=d['decisions'];x=features(decs[0]['observation'],100,r['mass'],0,0);terminal=features(d['steps'][-1],0,r['mass'],0,1)
    if len(actions)>1:
     z=features(decs[1]['observation'],100-decs[1]['boundary_step'],r['mass'],0,1);model.replay_buffer.add(x[None],z[None],np.array([[actions[0]['normalized']]]),np.array([0.]),np.array([False]),[{}]);model.replay_buffer.add(z[None],terminal[None],np.array([[actions[1]['normalized']]]),np.array([float(r['success'])]),np.array([True]),[{}])
    else:model.replay_buffer.add(x[None],terminal[None],np.array([[actions[0]['normalized']]]),np.array([float(r['success'])]),np.array([True]),[{}])
   (a.out/'WARMSTART.json').write_text(json.dumps(dict(source=str(a.warm),transitions=model.replay_buffer.size(),scope='actual completed prior physical episodes; recovered replay, not recovered lost actor weights'),indent=2))
  for checkpoint in range(model.num_timesteps+256,a.decisions+1,256):
   model.learn(total_timesteps=256,reset_num_timesteps=False);model.save(a.out/f'SAC_{checkpoint}');model.save_replay_buffer(a.out/f'BUFFER_{checkpoint}.pkl');record=dict(decisions=model.num_timesteps,optimizer_updates=model._n_updates,physical_steps=sum(r['steps']for r in env.rows),episodes=len(env.rows),wall_s=time.monotonic()-start,logger={k:float(v) for k,v in model.logger.name_to_value.items() if isinstance(v,(float,int,np.floating))})
   (a.out/f'TRAINING_{checkpoint}.json').write_text(json.dumps(record,indent=2));(a.out/'TRAINING.json').write_text(json.dumps(record,indent=2))
 except BaseException as exc:
  if 'model' in locals():model.save(a.out/'INTERRUPTED');model.save_replay_buffer(a.out/'INTERRUPTED_BUFFER.pkl')
  with gzip.open(a.out/'PARTIAL.json.gz','wt')as f:json.dump(dict(error=repr(exc),steps=env.env.records,actions=getattr(env,'actions',[]),case=getattr(env,'case',None)),f)
  raise
 finally:
  if len(env.rows)<env.counter and env.env.records:
   with gzip.open(a.out/'UNFINISHED_PHYSICAL_EPISODE.json.gz','wt')as f:json.dump(dict(case=env.case,initial=env.env.initial,steps=env.env.records,actions=env.actions,plans=env.env.plans,skill_results=env.env.skill_results,status='PARTIAL_NOT_TERMINAL_SUCCESS_OR_FAILURE'),f)
  env.close()
if __name__=='__main__':main()
