"""Ordinary two-boundary SMDP SAC: native grasp pitch and contact pace.
Inherited masked actor/critic-LN. A baseline, NOT an independent method.
"""
import argparse,json,time,gzip,random,shutil,hashlib
from pathlib import Path
import numpy as np,torch,gymnasium as gym
from program_contact_sac import BoundarySAC,BoundaryPolicy
from program_stap_learning import features
import program_native_charger_pace as physical
from pull_tool_physics import sp,serial
from transforms3d.quaternions import quat2axangle

class NativeEnv(gym.Env):
 def __init__(self,out,seeds,reward_mode='native_sparse'):
  self.reward_mode=reward_mode;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True);self.ep=physical.make_native();self.seeds=seeds;self.counter=0;self.rows=[];self.stage=0;self.action_space=gym.spaces.Box(-1.,1.,(2,),np.float32);self.observation_space=gym.spaces.Box(-np.inf,np.inf,(54,),np.float32)
 def obs(self):
  e=self.ep.env;u=e.unwrapped;local=u.goal_pose.sp.inv()*sp(u.charger);axis,angle=quat2axangle(local.q)
  if angle>np.pi:angle-=2*np.pi
  force=np.array(serial(u.scene.get_pairwise_contact_forces(u.charger,u.receptacle)[0]))if self.stage else np.zeros(3)
  return np.r_[features(e.measure(),(200-len(e.records))/2,1.,0.,self.stage),local.p/.025,axis*angle/.2,force/10].astype(np.float32)
 def reset(self,seed=None,options=None):
  super().reset(seed=seed);self.case=self.seeds[self.counter%len(self.seeds)];self.counter+=1;self.stage=0;self.actions=[];self.start=time.monotonic();self.ep.env.reset(seed=self.case);return self.obs(),{}
 def step(self,action):
  a=np.clip(np.asarray(action,dtype=float),-1,1);before=len(self.ep.env.records)
  if self.stage==0:
   a[1]=0.;self.angle=15.+15.*(float(a[0])+1);outer=(random.getstate(),np.random.get_state(),torch.get_rng_state());random.seed(31031);np.random.seed(31031);torch.manual_seed(31031)
   self.ep.reset(self.case,(0.,0.),angle=self.angle);self.physics_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state());random.setstate(outer[0]);np.random.set_state(outer[1]);torch.set_rng_state(outer[2]);self.stage=1
   reward=0.;done=False
   if self.ep.error is not None or len(self.ep.env.records)>=200:
    # Failed predecessor has no legal successor decision. Complete native horizon
    # without allowing a fictitious action/entropy term in an absorbing state.
    e=self.ep.env;q=np.asarray(serial(e.unwrapped.agent.robot.get_qpos()[0,:7]));e.phase='TERMINAL_HOLD'
    while len(e.records)<200:e.step(np.r_[q,-1.])
    reward=5. if e.records[-1]['native_success']else -1.;done=True;self.ep.done=True;self.ep.planner.close()
  else:
   outer=(random.getstate(),np.random.get_state(),torch.get_rng_state());random.setstate(self.physics_rng[0]);np.random.set_state(self.physics_rng[1]);torch.set_rng_state(self.physics_rng[2]);_,reward,done=self.ep.step(a);self.physics_rng=(random.getstate(),np.random.get_state(),torch.get_rng_state());random.setstate(outer[0]);np.random.set_state(outer[1]);torch.set_rng_state(outer[2])
  if self.reward_mode=='native_sparse':reward=float(self.ep.env.records[-1]['native_success'])if done else 0.
  self.actions.append(dict(stage=0 if before==0 else 1,step=before,action=a.tolist(),reward=reward,duration=len(self.ep.env.records)-before))
  if done:
   row=physical.save(self.ep,self.out,self.counter,self.angle,'ordinary_sac');path=Path(row['path']);d=json.load(gzip.open(path,'rt'));d['policy_actions']=self.actions
   with gzip.open(path,'wt')as f:json.dump(d,f)
   row['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();row['rl_decisions']=len(self.actions);self.rows.append(row);(self.out/'RESULTS.json').write_text(json.dumps(self.rows,indent=2));print('EP',self.counter,self.case,round(self.angle,2),row['success'],row['error'],flush=True)
  return self.obs(),reward,done,False,dict(duration_steps=len(self.ep.env.records)-before)
 def close(self):self.ep.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--reward-mode',choices=['native_sparse','legacy_contact_progress'],default='native_sparse');p.add_argument('--out',type=Path,required=True);p.add_argument('--decisions',type=int,default=1024);p.add_argument('--scene-start',type=int,default=749120);p.add_argument('--scenes',type=int,default=16);p.add_argument('--seed',type=int,default=31181);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 for n in ['program_native_charger_sac.py','program_native_charger_pace.py','program_contact_sac.py','program_sac_policy.py','program_skill_baselines.py','program_adaptive_contact.py','program_completion_contract.py','program_charger.py','pull_tool_physics.py']:shutil.copy2(Path(__file__).with_name(n),a.out/n)
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},task='PlugCharger-v1',horizon=200,stage0='grasp pitch15..45deg; unused action dimension masked',stage1='advance0.5..4mm and angular increment0.01..0.08rad, four physics steps; same strong servo/dwell',reward=a.reward_mode+'; native_sparse:1 only if full native task success at200, otherwise0; no partial contact-only potential',gamma=1,obs='same54D actual state/current task error/actual contact/budget, no future outcomes',source_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in a.out.glob('*.py')},claim='ORDINARY_RL_BASELINE_NOT_NEW_METHOD'),indent=2));env=NativeEnv(a.out/'episodes',list(range(a.scene_start,a.scene_start+a.scenes)),a.reward_mode);model=BoundarySAC(BoundaryPolicy,env,learning_rate=3e-4,buffer_size=50000,learning_starts=128,batch_size=64,train_freq=1,gradient_steps=2,ent_coef=.02,gamma=1.,tau=.005,policy_kwargs=dict(net_arch=[128,128]),device='cpu',seed=a.seed);model.save(a.out/'INITIAL');t=time.monotonic()
 try:
  for k in range(256,a.decisions+1,256):
   model.learn(256,reset_num_timesteps=False);model.save(a.out/f'SAC_{k}');model.save_replay_buffer(a.out/f'BUFFER_{k}.pkl');budget=dict(decisions=model.num_timesteps,updates=model._n_updates,complete_episodes=len(env.rows),complete_steps=sum(x['steps']for x in env.rows),current_partial_steps=0 if env.ep.done else len(env.ep.env.records),wall_s=time.monotonic()-t);(a.out/f'TRAINING_{k}.json').write_text(json.dumps(budget,indent=2));print('CHECKPOINT',budget,flush=True)
 except BaseException:
  model.save(a.out/'INTERRUPTED');model.save_replay_buffer(a.out/'INTERRUPTED_BUFFER.pkl');(a.out/'INTERRUPTED_BUDGET.json').write_text(json.dumps(dict(decisions=model.num_timesteps,updates=model._n_updates,episodes=len(env.rows),partial_steps=0 if env.ep.done else len(env.ep.env.records)),indent=2));raise
 finally:
  if env.ep.env.records and not env.ep.done:
   with gzip.open(a.out/'PARTIAL.json.gz','wt')as f:json.dump(dict(seed=env.case,angle=env.angle,initial=env.ep.env.initial,initial_measure=env.ep.env.initial_measure,steps=env.ep.env.records,plans=env.ep.env.plans,calls=env.ep.env.calls,actions=env.actions),f)
  env.close()
if __name__=='__main__':main()
