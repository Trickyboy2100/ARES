"""Revision: state-conditioned target-chart Actor coordinates, unchanged SAC.

Targets are derived from already observed reference differences, not planner
queries or future outcomes. Target proximity is only an inductive bias: it does
not imply equal trajectories, contact outcomes or atomic continuation semantics.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from train_full_option import Learner as Base
from program_full_option import FullJoint
from program_rule_option import Assets
from program_atomic_feedback import AtomicRuntime
from train_joint_continuation import sha
from train_option_online import atomic_json,atomic_torch

class Learner(Base):
 def __init__(self,assets,seed,coordinate='state'):
  super().__init__(assets,seed,'sac');self.arm='chart_'+coordinate;self.coordinate=coordinate
 def matrix(self,ss):
  s=torch.as_tensor(np.asarray(ss),dtype=torch.float32);n=len(s)
  # Indexed by successor selectors: refresh, four active offsets, four
  # atomic refreshed offsets. Rule, continue, withdraw and hold stay isolated.
  ids=[2,3,4,5,6,9,10,11,12];feat=torch.zeros(n,9,7)
  shifts=torch.tensor([[0,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],dtype=torch.float32)
  feat[:,:,:3]=shifts # one unit = existing3mm action grid
  refreshed=[0,5,6,7,8]
  if self.coordinate=='state':
   feat[:,refreshed,:3]+=s[:,None,77:80]*(.05/.003)
   feat[:,refreshed,3:6]+=s[:,None,80:83]*(180./5.) # observed rotvec/pi;5degree scale
  feat[:,5:,6]=1 # atomic refreshed continuation differs even at equal targets
  distance=(feat[:,:,None]-feat[:,None,:]).square().sum(-1)
  kernel=torch.exp(-.5*distance)
  # A repeat of the active correction with a nonempty queue does not replan.
  # Isolate it instead of equating its target coordinates with a new plan.
  last=(s[:,63]*7).round().long();physical=torch.tensor(ids)-1
  repeat=(last[:,None]==physical[None,:])&(s[:,61:62]>0)
  eye=torch.eye(9)[None,:,:]
  kernel=torch.where(repeat[:,:,None]|repeat[:,None,:],eye,kernel)
  block=.75*eye+.25*kernel # pointwise invertible; fixed state processing is an inductive bias
  mat=torch.eye(13).repeat(n,1,1)
  for i,ii in enumerate(ids):
   for j,jj in enumerate(ids):mat[:,ii,jj]=block[:,i,j]
  return mat/mat.norm(dim=2,keepdim=True)
 def logits(self,x,b,mask,ss):
  z=self.actor(x);rear=torch.bmm(self.matrix(ss),z[:,3:16,None]).squeeze(-1)
  front=torch.cat([z[:,:3],torch.zeros(len(z),10)],1)
  delta=torch.where(torch.tensor(np.asarray(ss)[:,44]<.5)[:,None],front,rear)
  return(b+delta).masked_fill(~mask,-1e9)
 def state(self):
  d=super().state();d['coordinate']=self.coordinate;return d

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--shared',type=Path,required=True);p.add_argument('--coordinate',choices=['state','static'],default='state');p.add_argument('--seed',type=int,default=32201);p.add_argument('--episodes',type=int,default=128);p.add_argument('--train-start',type=int,default=765100);p.add_argument('--dev-start',type=int,default=765400);p.add_argument('--dev-count',type=int,default=24);p.add_argument('--updates',type=int,default=40);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
 manifest={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()if k not in ('resume','stop_after')}
 manifest.update(source={f:sha(Path(__file__).with_name(f))for f in ['train_target_chart_option.py','train_full_option.py','program_full_option.py','program_atomic_feedback.py','peg_atomic_macro.py']},shared_sha=sha(a.shared),native100=True,gamma=1,train=list(range(a.train_start,a.train_start+a.episodes)),dev=list(range(a.dev_start,a.dev_start+a.dev_count)),selection='fixed final;0/32/64/128 checkpoints',candidate='fixed RBF target-chart Actor coordinates; state-dependent versus static reference shift ablation; no additional information or action authority',basis=dict(translation_m=.003,rotation_deg=5,kernel_weight=.25,atomic_family_feature=1),shared_training='same Base.train, optimizer, critic, exploration and common256 native design episodes as full SAC')
 assert not set(manifest['train'])&set(manifest['dev']);assets=Assets();m=Learner(assets,a.seed,a.coordinate);history=[];cursor=0
 if a.resume:
  assert json.loads((a.out/'MANIFEST.json').read_text())==manifest;d=torch.load(a.out/'CURRENT.pt',weights_only=False);m.load(d['model']);data=d['data'];history=d['history'];cursor=d['cursor']
 else:
  assert not(a.out/'MANIFEST.json').exists();data=m.decorate(copy.deepcopy(torch.load(a.shared,weights_only=False)['data']),-1000);atomic_json(a.out/'MANIFEST.json',manifest);shutil.copy2(__file__,a.out/Path(__file__).name);atomic_torch(a.out/'INITIAL.pt',m.state());atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=0))
 rt=AtomicRuntime(a.out/'physics');rt.counter=max([0]+[int(f.name.split('_')[0])for f in rt.out.glob('*.json.gz')]);rt.rows=[h['row']for h in history];jobs=[]
 for i in range(a.episodes+1):
  if i in [0,32,64,128,a.episodes]:jobs.extend([('eval',i,s)for s in manifest['dev']])
  if i<a.episodes:jobs.append(('train',i,a.train_start+i))
 try:
  for ji in range(cursor,len(jobs)):
   kind,i,seed=jobs[ji];m.episode=i;pending=a.out/f'PENDING_{ji:04d}.pt'
   if pending.exists():item=torch.load(pending,weights_only=False);assert item['job']==jobs[ji];m.rng.bit_generator.state=item['rng']
   else:
    start=a.out/f'START_{ji:04d}.json';assert not start.exists(),'Inspect incomplete physics before resume';atomic_json(start,dict(job=jobs[ji]));pol=FullJoint(assets,lambda s:m.choose(s,kind=='train'),lambda:len(rt.env.records));row,ts,_=rt.run(seed,pol,f'{kind}_{i}');ts=m.decorate(pol.transitions(ts),i);item=dict(row=row,ts=ts,events=pol.events,rng=m.rng.bit_generator.state,job=jobs[ji]);atomic_torch(pending,item)
   log=None
   if kind=='train':data.extend(item['ts']);log=m.train(data,a.updates,i)
   history.append(dict(kind=kind,checkpoint=i,row=item['row'],events=item['events'],effective_decisions=len(item['ts']),log=log));atomic_json(a.out/'HISTORY.json',history);atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=ji+1))
   if kind=='eval'and seed==manifest['dev'][-1]:atomic_torch(a.out/f'CHECKPOINT_{i}.pt',m.state())
   print(m.arm,a.seed,kind,i,seed,item['row']['success'],len(item['ts']),flush=True)
   if kind=='eval'and seed==manifest['dev'][-1]and a.stop_after==i:break
 finally:rt.close()
if __name__=='__main__':main()
