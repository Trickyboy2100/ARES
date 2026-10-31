import sys,json,hashlib
from pathlib import Path
sys.path[:0]=['research/skill_handoff_lab/tools','/home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0']
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
class ShapeEnv(gym.Env):
 def __init__(self):
  self.observation_space=gym.spaces.Box(-10,10,(90,),dtype=np.float32);self.action_space=gym.spaces.Box(-1,1,(3,),dtype=np.float32)
 def reset(self,*,seed=None,options=None):super().reset(seed=seed);return np.zeros(90,dtype=np.float32),{}
 def step(self,a):return np.zeros(90,dtype=np.float32),0.,False,False,{}
def vector(m):return np.concatenate([p.detach().cpu().numpy().ravel() for p in m.actor.parameters()])
initial=SAC('MlpPolicy',ShapeEnv(),policy_kwargs=dict(net_arch=[64,64]),learning_rate=3e-4,buffer_size=10000,learning_starts=128,batch_size=64,train_freq=1,gradient_steps=1,ent_coef=.02,gamma=.99,tau=.005,seed=31031,device='cpu',verbose=0)
x=vector(initial);rows=[]
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
for method,run in [('sac','sac_pilot_1024_retry'),('effect_sac','effect_sac_pilot_1024_retry'),('ln_sac_alpha01','ln_sac_pilot_1024'),('ln_sac_alpha002','ln_sac_alpha002'),('effect_ln_sac_alpha002','effect_ln_sac_alpha002')]:
 p=r/run/'MODEL.zip'
 if not p.exists():continue
 m=SAC.load(p,device='cpu');v=vector(m);rows.append(dict(method=method,model=str(p),model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),actor_parameters=len(v),changed_parameters=int(np.count_nonzero(v!=x)),actor_l2_change=float(np.linalg.norm(v-x)),initial_actor_sha256=hashlib.sha256(x.tobytes()).hexdigest(),final_actor_sha256=hashlib.sha256(v.tobytes()).hexdigest()))
(r/'WEIGHT_QA.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
