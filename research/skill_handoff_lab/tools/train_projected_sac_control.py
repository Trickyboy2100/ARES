"""Matched normalization/data/width continuous projected SAC control (standard)."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch,gymnasium as gym
from program_discrete_options import prior,normalizer
import program_contact_sac as sac
from stable_baselines3.common.logger import configure

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False);torch.set_num_threads(2);data,_=prior(a.prior);mu,sd=normalizer(data);np.savez(a.out/'NORMALIZATION.npz',mu=mu,sd=sd);sac.mask=lambda obs:torch.stack([torch.ones_like(obs[:,44]),torch.zeros_like(obs[:,44])],1)
 logs=[]
 for seed in [31301,31302]:
  e=gym.Env();e.observation_space=gym.spaces.Box(-np.inf,np.inf,(54,),np.float32);e.action_space=gym.spaces.Box(-1,1,(2,),np.float32)
  m=sac.BoundarySAC(sac.BoundaryPolicy,e,learning_rate=3e-4,buffer_size=10000,learning_starts=0,batch_size=64,ent_coef=.02,gamma=1.,tau=.005,policy_kwargs=dict(net_arch=[64,64]),device='cpu',seed=seed);m.set_logger(configure(str(a.out/f'log{seed}'),['csv']));m._current_progress_remaining=1.
  for t in data:
   action=t['a']-1 if t['stage']==0 else 2*t['a']-1;m.replay_buffer.add(((np.array(t['s'])-mu)/sd).astype('float32')[None],((np.array(t['next'])-mu)/sd).astype('float32')[None],np.array([[action,0]],dtype='float32'),np.array([t['r']]),np.array([t['done']]),[{}])
  m.save(a.out/f'INITIAL_{seed}');start=time.monotonic()
  for k in [200,800,2000]:
   m.train(gradient_steps=k-m._n_updates,batch_size=64);m.save(a.out/f'SAC_{seed}_{k}');logs.append(dict(seed=seed,updates=k,wall_s=time.monotonic()-start,transitions=len(data)));(a.out/'TRAINING.json').write_text(json.dumps(logs,indent=2))
  m.save_replay_buffer(a.out/f'BUFFER_{seed}.pkl')
if __name__=='__main__':main()
