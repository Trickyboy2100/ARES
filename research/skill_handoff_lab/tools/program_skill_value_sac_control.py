"""Inherited ordinary masked LN-SAC, same scene folds/data as paired-value study.
Actor projected onto the same logged candidates: closed-set diagnostic, not online score.
"""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np,torch,gymnasium as gym
from program_paired_skill_value import load,load_peg
from program_contact_sac import BoundarySAC,BoundaryPolicy
from stable_baselines3.common.logger import configure

def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['tool','peg'],default='tool');p.add_argument('--full-train',action='store_true');p.add_argument('--aux',choices=['none','paired','absolute','shuffled'],default='none');p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/'SOURCE.py');torch.set_num_threads(2);groups=load_peg(a.root)if a.task=='peg'else load(a.root)
 if a.task=='peg':
  import program_contact_sac as sac
  sac.mask=lambda obs:torch.stack([torch.ones_like(obs[:,44]),torch.zeros_like(obs[:,44])],dim=1)
  for g in groups:
   for r,d,ts in g['items']:
    r['action']=[2*r['action'][0]-1,0.]
    if len(ts)==2:
     t=ts[1];ts[1]=(t[0],t[1],r['action'],t[3],t[4])
 scenes=sorted({g['seed']for g in groups});logs=[];results=[]
 for fold in ([-1]if a.full_train else range(3)):
  train=[g for g in groups if scenes.index(g['seed'])%3!=fold];test=[g for g in groups if scenes.index(g['seed'])%3==fold]
  for seed in ([31201]if a.full_train else [31201,31202,31203]):
   env=gym.Env();env.observation_space=gym.spaces.Box(-np.inf,np.inf,(54,),np.float32);env.action_space=gym.spaces.Box(-1.,1.,(2,),np.float32)
   model=BoundarySAC(BoundaryPolicy,env,learning_rate=3e-4,buffer_size=10000,learning_starts=0,batch_size=64,ent_coef=.02,gamma=1.,tau=.005,policy_kwargs=dict(net_arch=[64,64]),device='cpu',seed=seed);model.set_logger(configure(str(a.out/f'f{fold}s{seed}'),['csv']));count=0
   for g in train:
    for _,_,ts in g['items']:
     for s,z,u,r,done in ts:model.replay_buffer.add(s[None],z[None],np.array(u,dtype=np.float32)[None],np.array([r]),np.array([done]),[{}]);count+=1
   if a.aux!='none':
    xs=[];acts=[];ys=[];pairs=[]
    for g in train:
     ids=[]
     for rr,dd,tt in g['items']:
      if len(tt)==2:ids.append(len(xs));xs.append(tt[1][0]);acts.append(tt[1][2]);ys.append(tt[1][3])
     pairs.extend((i,j)for i in ids for j in ids if i<j)
    xx=torch.tensor(np.array(xs),dtype=torch.float32);aa=torch.tensor(np.array(acts),dtype=torch.float32);yy=torch.tensor(ys,dtype=torch.float32);ij=torch.tensor(pairs,dtype=torch.long)
    if a.aux=='shuffled':ij[:,1]=torch.randperm(len(xs))[ij[:,1]]
    original_step=model.critic.optimizer.step;aux_log=[]
    def augmented_step(*args,**kwargs):
     qs=model.critic(xx,aa)
     if a.aux=='absolute':loss=sum(((q[:,0]-yy)**2).mean()for q in qs)/len(qs)
     else:loss=sum(((q[ij[:,0],0]-q[ij[:,1],0]-(yy[ij[:,0]]-yy[ij[:,1]]))**2).mean()for q in qs)/len(qs)
     loss.backward();aux_log.append(float(loss.detach()));return original_step(*args,**kwargs)
    model.critic.optimizer.step=augmented_step
   model._current_progress_remaining=1.;model.save(a.out/f'INITIAL_f{fold}s{seed}');start=time.monotonic()
   for upto in [200,600,1200]:
    before=model._n_updates;model.train(gradient_steps=upto-before,batch_size=64);model.save(a.out/f'SAC{upto}_f{fold}s{seed}');model.save_replay_buffer(a.out/f'BUFFER_f{fold}s{seed}.pkl');chosen=[]
    for g in test:
     rr=g['items'];tt=rr[0][2];prediction=model.predict(tt[1][0]if len(tt)==2 else tt[0][0],deterministic=True)[0];actions=np.array([r['action']for r,_,_ in rr]);idx=int(np.linalg.norm(actions-prediction,axis=1).argmin())if len(tt)==2 else 0
     y=float(rr[idx][0]['success']);nom=next(float(r['success'])for r,_,_ in rr if r['action']==([-1.,0.]if a.task=='peg'else [0.,0.]));r=dict(fold=fold,seed=seed,checkpoint=upto,scene=g['seed'],grasp=g['grasp'],success=y,nominal=nom,actor=prediction.tolist(),selected_action=actions[idx].tolist());results.append(r);chosen.append(r)
    joint=[]
    for scene in {g['seed']for g in test}:
     candidates=[g for g in test if g['seed']==scene];u=model.predict(candidates[0]['items'][0][2][0][0],deterministic=True)[0];grasp=.01*float(u[0])if a.task=='peg'else .07+.035*float(u[0]);g=min(candidates,key=lambda g:abs(g['grasp']-grasp));joint.append(next(r['success']for r in chosen if r['scene']==scene and r['grasp']==g['grasp']))
    logs.append(dict(fold=fold,seed=seed,checkpoint=upto,transitions=count,actor_updates=upto,critic_updates=upto,wall_s=time.monotonic()-start,context_success=sum(x['success']for x in chosen),context_n=len(chosen),joint_success=sum(joint),joint_n=len(joint),new_physics=0,task=a.task,auxiliary=a.aux,auxiliary_mean=float(np.mean(aux_log[-100:]))if a.aux!='none'else None));(a.out/'TRAINING.json').write_text(json.dumps(logs,indent=2));(a.out/'PREDICTIONS.json').write_text(json.dumps(results,indent=2));print(logs[-1],flush=True)
if __name__=='__main__':main()
