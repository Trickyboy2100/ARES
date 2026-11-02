"""Testable effect-aligned SAC representation, not an established novel method.
Physical square authority is identical; change of variables includes exact Jacobian BEFORE the shared no-op deadzone.
Frozen ridge learns measured paired terminal cube displacement, not success labels.
"""
import argparse,json,gzip,time,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
import program_contact_sac as sac
from program_stap_learning import features
from scipy.spatial.transform import Rotation

def observation(m,stage,remaining,progress,goal):
 return np.r_[features(m,remaining,1.,0.,stage),m['cube_linear_velocity'],np.log1p(m['tool_cube_force_N'])if stage else 0.,progress,goal].astype(np.float32)
def transitions(d):
 r=d['summary'];event=d['contact_event'];x=observation(d['initial_measure'],0,100,[0,0],[0,0,0]);a0=[(r['grasp']-.07)/.035,0.];goal=np.zeros(3)
 if event:
  plan=next(v for v in d['plans']if v['phase']=='PULL'and 'attempts'in v);before=d['steps'][plan['step']-1];q=plan['target'][3:];goal=Rotation.from_quat([*q[1:],q[0]]).apply(before['tcp_T_tool'][:3])+plan['target'][:3]
  start=np.array(d['skill_results'][0]['measurement']['cube'][:2]);z=observation(event['measurement'],1,event['remaining'],np.array(event['measurement']['cube'][:2])-start,goal);last=observation(d['steps'][-1],1,0,np.array(d['steps'][-1]['cube'][:2])-start,goal)
  a1=d.get('policy_actions',[{},{}])[-1].get('physical_action',r['action'])
  return [(x,z,a0,0.,False),(z,last,a1,float(r['success']),True)]
 return [(x,observation(d['steps'][-1],1,0,[0,0],[0,0,0]),a0,float(r['success']),True)]

def physical_features(x):
 v=x[...,21:24]-x[...,14:17];q=x[...,17:21];u=q[...,1:];local=v-2*q[...,:1]*np.cross(u,v)+2*np.cross(u,np.cross(u,v))
 return np.concatenate([local/.1,x[...,28:29]/.1,x[...,41:42],np.linalg.norm(x[...,51:53]-x[...,14:16],axis=-1,keepdims=True)/.36],axis=-1)

def fit(root,out):
 out.mkdir(parents=True,exist_ok=False);rows=[];dataset=[]
 for folder in ['relative','near','design']:
  rs=json.loads((root/folder/'RESULTS.json').read_text());groups={}
  for r in rs:groups.setdefault((r['seed'],r['grasp']),[]).append(r)
  for key,rs in groups.items():
   ds=[json.load(gzip.open(r['path'],'rt'))for r in rs]
   for r,d in zip(rs,ds):
    assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'];dataset.append(dict(path=r['path'],sha256=r['sha256']))
   if ds[0]['contact_event'] is None:continue
   n=ds[0]['contact_event']['step'];assert all(d['steps'][:n]==ds[0]['steps'][:n]for d in ds)
   z=transitions(ds[0])[1][0];by={tuple(d['summary']['action']):d for d in ds};j=np.stack([(np.array(by[tuple(a)]['steps'][-1]['cube'][:2])-np.array(by[tuple(-np.array(a))]['steps'][-1]['cube'][:2]))/2 for a in [[1,0],[0,1]]],axis=1)
   rows.append(dict(seed=key[0],grasp=key[1],state=z.tolist(),jacobian=j.tolist()))
 X=physical_features(np.array([r['state']for r in rows]));Y=np.array([r['jacobian']for r in rows]).reshape(-1,4);seeds=np.array([r['seed']for r in rows]);mu=X.mean(0);sd=np.ones(X.shape[1])
 def solve(x,y,alpha):return np.linalg.solve(x.T@x+alpha*np.diag([0]+[1]*(x.shape[1]-1)),x.T@y)
 cv={};global_errors=[]
 for seed in sorted(set(seeds)):global_errors.extend(np.sum((Y[seeds!=seed].mean(0)-Y[seeds==seed])**2,axis=1).tolist())
 for alpha in [.01,.1,1.,10.,100.]:
  errors=[]
  for seed in sorted(set(seeds)):
   train=seeds!=seed;mean=X[train].mean(0);std=np.ones(X.shape[1]);tx=np.c_[np.ones(train.sum()),(X[train]-mean)/std];vx=np.c_[np.ones((~train).sum()),(X[~train]-mean)/std];pred=vx@solve(tx,Y[train],alpha);errors.extend(np.sum((pred-Y[~train])**2,axis=1).tolist())
  cv[alpha]=float(np.mean(errors))
 alpha=min(cv,key=cv.get);B=solve(np.c_[np.ones(len(X)),(X-mu)/sd],Y,alpha)
 payload=dict(mean=mu.tolist(),std=sd.tolist(),coef=B.tolist(),alpha=alpha,leave_scene_out_mse=cv,leave_scene_out_global_mse=float(np.mean(global_errors)),contexts=len(rows),scene_clusters=len(set(seeds)),global_jacobian=Y.mean(0).reshape(2,2).tolist(),features='cube in actual tool frame xyz/.1, attachment x/.1, budget/100, remaining tool goal XY length/.36; fixed physical units',target='paired finite difference of terminal cube worldXY, per normalized legal parameter; future effects used only as training labels',rows=rows,dataset=dataset)
 (out/'EFFECT_MODEL.json').write_text(json.dumps(payload,indent=2));print({k:payload[k]for k in ['contexts','scene_clusters','leave_scene_out_mse','alpha']})

class DirectionActor(sac.MaskedActor):
 def __init__(self,*args,effect=None,direction='conditional',**kwargs):
  super().__init__(*args,**kwargs);self.direction=direction
  self.register_buffer('effect_mean',torch.tensor(effect['mean'],dtype=torch.float32));self.register_buffer('effect_std',torch.tensor(effect['std'],dtype=torch.float32));self.register_buffer('effect_coef',torch.tensor(effect['coef'],dtype=torch.float32));self.register_buffer('global_j',torch.tensor(effect['global_jacobian'],dtype=torch.float32))
 def mapped(self,obs,a):
  v=obs[:,21:24]-obs[:,14:17];q=obs[:,17:21];u=q[:,1:];local=v-2*q[:,:1]*torch.cross(u,v,dim=1)+2*torch.cross(u,torch.cross(u,v,dim=1),dim=1);f=torch.cat([local/.1,obs[:,28:29]/.1,obs[:,41:42],(obs[:,51:53]-obs[:,14:16]).norm(dim=1,keepdim=True)/.36],1);X=torch.cat([torch.ones_like(f[:,:1]),(f-self.effect_mean)/self.effect_std],1)
  J=(X@self.effect_coef).reshape(-1,2,2)if self.direction=='conditional'else self.global_j.expand(len(obs),-1,-1)
  pull=obs[:,51:53]-obs[:,14:16];pull=pull/pull.norm(dim=1,keepdim=True).clamp(min=1e-6);g=torch.einsum('bij,bi->bj',J,pull);norm=g.norm(dim=1,keepdim=True);u=g/norm.clamp(min=1e-6);u=torch.where((norm<.001).expand_as(u),torch.tensor([1.,0.],device=obs.device),u)
  rot=torch.stack([u[:,0]*a[:,0]-u[:,1]*a[:,1],u[:,1]*a[:,0]+u[:,0]*a[:,1]],1);factor=a.abs().amax(1,keepdim=True)/rot.abs().amax(1,keepdim=True).clamp(min=1e-8);factor=torch.where(a.abs().amax(1,keepdim=True)<1e-8,torch.ones_like(factor),factor);mapped=rot*factor;active=obs[:,44:45]>.5
  return torch.where(active,mapped,a),torch.where(active[:,0],2*factor[:,0].log(),torch.zeros_like(factor[:,0]))
 def forward(self,obs,deterministic=False):return self.mapped(obs,super().forward(obs,deterministic))[0]
 def action_log_prob(self,obs):
  a,lp=super().action_log_prob(obs);mapped,ld=self.mapped(obs,a);return mapped,lp-ld
class DirectionPolicy(sac.BoundaryPolicy):
 def __init__(self,*args,effect=None,direction='conditional',**kwargs):self.effect=effect;self.direction=direction;super().__init__(*args,**kwargs)
 def make_actor(self,features_extractor=None):return DirectionActor(**self._update_features_extractor(self.actor_kwargs,features_extractor),effect=self.effect,direction=self.direction).to(self.device)

def train(a):
 from stable_baselines3.common.logger import configure
 torch.set_num_threads(4);effect=json.loads(a.effect.read_text());
 if a.offline_only:
  import gymnasium as gym
  env=gym.Env();env.observation_space=gym.spaces.Box(-np.inf,np.inf,(54,),np.float32);env.action_space=gym.spaces.Box(-1.,1.,(2,),np.float32)
 else:env=sac.ContactEnv(a.out/'episodes',list(range(748010,748042)),'relative_yaw')
 policy=sac.BoundaryPolicy if a.method in ['ordinary','generic_aux']else DirectionPolicy;kw=dict(net_arch=[128,128])
 if a.method not in ['ordinary','generic_aux']:kw.update(effect=effect,direction=a.method)
 from program_contact_auxiliary import GenericAuxSAC
 model_class=GenericAuxSAC if a.method=='generic_aux'else sac.BoundarySAC
 model=model_class(policy,env,learning_rate=3e-4,buffer_size=50000,learning_starts=0,batch_size=64,train_freq=1,gradient_steps=2,ent_coef=.02,gamma=1.,tau=.005,policy_kwargs=kw,device='cpu',seed=a.seed);model.set_logger(configure(str(a.out),['csv']));dataset=effect['dataset'];count=0
 # Exact same physical data, physical actions and sparse native reward for both.
 for r in dataset:
  d=json.load(gzip.open(r['path'],'rt'))
  for x,z,u,reward,done in transitions(d):model.replay_buffer.add(x[None],z[None],np.array(u)[None],np.array([reward]),np.array([done]),[{}]);count+=1
 model.save(a.out/'INITIAL');model._current_progress_remaining=1.;start=time.monotonic();model.train(a.offline_updates,64);model.save(a.out/'OFFLINE');model.save_replay_buffer(a.out/'OFFLINE_BUFFER.pkl')
 if a.offline_only:
  (a.out/'TRAINING_OFFLINE.json').write_text(json.dumps(dict(offline_updates=a.offline_updates,shared_pretraining_transitions=count,shared_pretraining_episodes=len(dataset),new_physics_steps=0,wall_s=time.monotonic()-start),indent=2));return
 for k in range(128,a.decisions+1,128):
  model.learn(128,reset_num_timesteps=False);model.save(a.out/f'SAC_{k}');model.save_replay_buffer(a.out/f'BUFFER_{k}.pkl');stats=dict(decisions=model.num_timesteps,updates=model._n_updates,offline_updates=a.offline_updates,shared_pretraining_transitions=count,shared_pretraining_episodes=len(dataset),episodes=len(env.rows),physics_steps=sum(x['steps']for x in env.rows),wall_s=time.monotonic()-start);(a.out/f'TRAINING_{k}.json').write_text(json.dumps(stats,indent=2));print(stats,flush=True)
 if env.env.records and len(env.rows)<env.counter:
  with gzip.open(a.out/'PARTIAL.json.gz','wt')as f:json.dump(dict(seed=env.case,initial=env.env.initial,initial_measure=env.env.initial_measure,steps=env.env.records,plans=env.env.plans,actions=env.actions,decisions=env.env.decisions,skill_results=env.env.skill_results,contact_event=env.env.contact_event),f)
 env.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','train']);p.add_argument('--root',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--effect',type=Path);p.add_argument('--method',choices=['ordinary','conditional','global','generic_aux'],default='ordinary');p.add_argument('--decisions',type=int,default=256);p.add_argument('--offline-updates',type=int,default=500);p.add_argument('--offline-only',action='store_true');p.add_argument('--seed',type=int,default=31171);a=p.parse_args()
 if a.mode=='fit':fit(a.root,a.out)
 else:
  a.out.mkdir(parents=True,exist_ok=False)
  for n in ['program_contact_direction.py','program_contact_auxiliary.py','program_contact_sac.py','program_post_contact.py','program_tool_contract_control.py']:shutil.copy2(Path(__file__).with_name(n),a.out/n)
  (a.out/'PROTOCOL.json').write_text(json.dumps(dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},task='PullCubeTool-v1',horizon=100,gamma=1.,same_physical_action_box=True,entropy='exact nominal-parameter change of variables before common no-op projection; not differential entropy over executed no-op atoms',novelty='UNVERIFIED',extra_effect_fit='same shared trajectories; frozen ridge, no hidden physical-state labels'),indent=2));train(a)
if __name__=='__main__':main()
