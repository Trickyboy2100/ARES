"""Full-catalogue joint SMDP SAC and executable-reference actor coordinates.
Standard foundations; candidate coordinates are a falsifiable inductive bias,
not a claimed novel RL algorithm. Both arms see identical physical catalogue.
"""
import argparse,copy,json,time,shutil
from pathlib import Path
import numpy as np,torch
from torch import nn
from program_full_option import FullJoint,Warm,legal
from program_rule_option import Assets,rule
from program_atomic_feedback import AtomicRuntime
from train_joint_continuation import sha
from train_option_online import atomic_json,atomic_torch
class Learner:
 def __init__(self,assets,seed,arm):
  self.assets=assets;self.warm=Warm(assets);self.arm=arm;self.rng=np.random.default_rng(seed)
  # Warm Gate constructor resets global torch RNG. Set fresh-network seed AFTER it.
  torch.manual_seed(seed)
  self.mu=self.warm.model.mu;self.sd=self.warm.model.sd
  self.actor=nn.Sequential(nn.Linear(83,96),nn.Tanh(),nn.Linear(96,17));nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
  self.q=nn.ModuleList([nn.Sequential(nn.Linear(83,96),nn.Tanh(),nn.Linear(96,13))for _ in range(2)]);self.target=copy.deepcopy(self.q)
  self.ao=torch.optim.Adam(self.actor.parameters(),lr=1e-4);self.qo=torch.optim.Adam(self.q.parameters(),lr=3e-4);self.updates=0;self.episode=0
  # Same parameter count and full logit expressivity. Row-normalized fixed
  # coordinates share offset gradients across references; interactions permit
  # rejecting that sharing. Compare flat coordinates, not changed authority.
  self.map=torch.zeros(13,14);self.map[0,0]=1;self.map[1,1]=1;self.map[7,2]=1;self.map[8,3]=1;self.map[2,5]=1
  for k in range(4):self.map[3+k,4]=1;self.map[3+k,6+k]=1;self.map[9+k,5]=1;self.map[9+k,6+k]=1;self.map[9+k,10+k]=1
  self.map/=self.map.norm(dim=1,keepdim=True)
 def norm(self,s):return(torch.tensor(np.asarray(s),dtype=torch.float32)-self.mu)/self.sd
 def context(self,ss):
  rows=[];masks=[]
  for s in ss:
   b,m=self.warm.model.context([s]);b=b[0].numpy();m=m[0].numpy();out=np.full(13,-3.,np.float32);mask=legal(s)
   if s[44]<.5:out[:3]=b
   else:
    out[0]=b[0]
    for i,p in enumerate(self.assets.proposals(s)):
     if m[i+1]:
      a=p+1
      if a==1 and not mask[a]:a=0
      if mask[a]:out[a]=max(out[a],b[i+1])
   rows.append(out);masks.append(mask)
  return torch.tensor(np.array(rows)),torch.tensor(np.array(masks))
 def logits(self,x,b,mask,ss):
  z=self.actor(x);rear=z[:,3:16]if self.arm!='factor'else z[:,3:]@self.map.T
  front=torch.cat([z[:,:3],torch.zeros(len(z),10)],1);delta=torch.where(torch.tensor(np.asarray(ss)[:,44]<.5)[:,None],front,rear)
  return(b+delta).masked_fill(~mask,-1e9)
 def probs(self,x,b,mask,ss):return self.logits(x,b,mask,ss).softmax(1)
 def choose(self,s,train=False):
  b,m=self.context([s])
  with torch.no_grad():p=self.probs(self.norm([s]),b,m,[s])[0].numpy()
  if train:
   ratio=min((self.episode+1)/64.,1.);prior=np.eye(13)[self.warm.choose(s)]
   p=(1-ratio)*prior+ratio*p;p=.9*p+.1*m[0].numpy()/m[0].sum();return int(self.rng.choice(13,p=p/p.sum()))
  return int(p.argmax())
 def decorate(self,ts,episode):
  for t in ts:
   for key,s in [('base',t['s']),('next_base',t['next'])]:
    b,m=self.context([s]);t[key]=b[0].tolist();t[key+'_mask']=m[0].tolist()
   t['episode']=episode
  return ts
 def train(self,data,steps,episode):
  front=[t for t in data if t['stage']==0];rear=[t for t in data if t['stage']==1];tic=time.monotonic();logs=[]
  for _ in range(steps):
   rows=[front[i]for i in self.rng.integers(len(front),size=16)]+[(rear or front)[i]for i in self.rng.integers(len(rear or front),size=48)]
   ss=[t['s']for t in rows];ns=[t['next']for t in rows];x=self.norm(ss);z=self.norm(ns);a=torch.tensor([t['a']for t in rows]);idx=torch.arange(64)
   b=torch.tensor([t['base']for t in rows]);mask=torch.tensor([t['base_mask']for t in rows]);nb=torch.tensor([t['next_base']for t in rows]);nm=torch.tensor([t['next_base_mask']for t in rows])
   with torch.no_grad():
    p=self.probs(z,nb,nm,ns);q=torch.stack([v(z)for v in self.target]).mean(0);v=(p*(q-.01*p.clamp_min(1e-9).log())).sum(1)
    y=torch.tensor([t['r']for t in rows])+(~torch.tensor([t['done']for t in rows]))*v
   qloss=sum((q(x)[idx,a]-y).square().mean()for q in self.q)/2
   self.qo.zero_grad();qloss.backward();nn.utils.clip_grad_norm_(self.q.parameters(),10);self.qo.step()
   p=self.probs(x,b,mask,ss)
   with torch.no_grad():q=torch.stack([v(x)for v in self.q]).mean(0)
   if self.arm=='supervised':
    # Same data/state/action authority: direct terminal-outcome scorer.
    # Both successes and failures are used; no bootstrapped value enters actor.
    outcome=torch.tensor([t['outcome']for t in rows]);aloss=nn.functional.binary_cross_entropy_with_logits(self.logits(x,b,mask,ss)[idx,a],outcome)
   else:
    aloss=(p*(.01*p.clamp_min(1e-9).log()-q)).sum(1).mean();w=torch.tensor([t['outcome']for t in rows]);aloss-=.1*(w*p[idx,a].clamp_min(1e-9).log()).sum()/w.sum().clamp_min(1)
   self.ao.zero_grad();aloss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1);self.ao.step()
   with torch.no_grad():
    for q,t in zip(self.q.parameters(),self.target.parameters()):t.lerp_(q,.01)
   self.updates+=1;logs.append(dict(q=float(qloss),actor=float(aloss)))
  return dict(updates=steps,wall_s=time.monotonic()-tic,last=logs[-1])
 def state(self):return dict(arm=self.arm,actor=self.actor.state_dict(),q=self.q.state_dict(),target=self.target.state_dict(),ao=self.ao.state_dict(),qo=self.qo.state_dict(),rng=self.rng.bit_generator.state,updates=self.updates)
 def load(self,d):
  assert d['arm']==self.arm
  for k in ['actor','q','target','ao','qo']:getattr(self,k).load_state_dict(d[k])
  self.rng.bit_generator.state=d['rng'];self.updates=d['updates']
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--shared',type=Path,required=True);p.add_argument('--arm',choices=['sac','factor','supervised'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,default=128);p.add_argument('--train-start',type=int,default=765100);p.add_argument('--dev-start',type=int,default=765400);p.add_argument('--dev-count',type=int,default=24);p.add_argument('--updates',type=int,default=40);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
 manifest={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()if k not in ('resume','stop_after')};manifest.update(source={f:sha(Path(__file__).with_name(f))for f in ['train_full_option.py','program_full_option.py','program_atomic_feedback.py','peg_atomic_macro.py']},shared_sha=sha(a.shared),native100=True,gamma=1,exploration='shared 64-episode ramp from warm behavior to learned stochastic policy, plus10% legal uniform; new action prior logit -3',selection='fixed final; checkpoints0/32/64/128 when reached',train=list(range(a.train_start,a.train_start+a.episodes)),dev=list(range(a.dev_start,a.dev_start+a.dev_count)),seed_scope='torch reseeded AFTER loading shared Gate; distinct new actor hidden and critic weights',candidate='reference/offset gradient sharing with equal information/authority and full logit expressivity; standard parameterized-action foundations, no a priori originality')
 assert not set(manifest['train'])&set(manifest['dev']);assets=Assets();m=Learner(assets,a.seed,a.arm);data=[];history=[];cursor=0
 if a.resume:
  assert json.loads((a.out/'MANIFEST.json').read_text())==manifest;d=torch.load(a.out/'CURRENT.pt',weights_only=False);m.load(d['model']);data=d['data'];history=d['history'];cursor=d['cursor']
 else:
  assert not(a.out/'MANIFEST.json').exists();shared=torch.load(a.shared,weights_only=False);data=m.decorate(copy.deepcopy(shared['data']),-1000);atomic_json(a.out/'MANIFEST.json',manifest);shutil.copy2(__file__,a.out/Path(__file__).name);atomic_torch(a.out/'INITIAL.pt',m.state());atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=0))
 rt=AtomicRuntime(a.out/'physics');rt.counter=max([0]+[int(f.name.split('_')[0])for f in rt.out.glob('*.json.gz')]);rt.rows=[h['row']for h in history];jobs=[]
 for i in range(a.episodes+1):
  if i in [0,32,64,128,a.episodes]:jobs.extend([('eval',i,s)for s in manifest['dev']])
  if i<a.episodes:jobs.append(('train',i,a.train_start+i))
 try:
  for ji in range(cursor,len(jobs)):
   kind,i,seed=jobs[ji];m.episode=i;pending=a.out/f'PENDING_{ji:04d}.pt'
   if pending.exists():item=torch.load(pending,weights_only=False);assert item['job']==jobs[ji];m.rng.bit_generator.state=item['rng']
   else:
    start=a.out/f'START_{ji:04d}.json';assert not start.exists(),'Inspect incomplete physics before resume';atomic_json(start,dict(job=jobs[ji]));pol=FullJoint(assets,lambda s:m.choose(s,kind=='train'),lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,f'{kind}_{i}');ts=m.decorate(pol.transitions(ts),i);item=dict(row=row,ts=ts,events=pol.events,rng=m.rng.bit_generator.state,job=jobs[ji]);atomic_torch(pending,item)
   log=None
   if kind=='train':data.extend(item['ts']);log=m.train(data,a.updates,i)
   history.append(dict(kind=kind,checkpoint=i,row=item['row'],events=item['events'],effective_decisions=len(item['ts']),log=log));atomic_json(a.out/'HISTORY.json',history);atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=ji+1))
   if kind=='eval'and seed==manifest['dev'][-1]:atomic_torch(a.out/f'CHECKPOINT_{i}.pt',m.state())
   print(a.arm,a.seed,kind,i,seed,item['row']['success'],len(item['ts']),flush=True)
   if kind=='eval'and seed==manifest['dev'][-1]and a.stop_after==i:break
 finally:rt.close()
if __name__=='__main__':main()
