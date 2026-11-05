"""Execution-identified joint Option SAC: flat versus action-shared critics.
Both see exactly the same deterministic proposal/execution identities. Shared
Q enforces equality for executable aliases; parameterized Q is prior art.
This is a new bounded representation experiment, not an originality assertion.
"""
import argparse, copy, hashlib, json, time, shutil, gzip
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_rule_option import Assets, rule, gate_features
from train_rule_gate_v2 import Gate
from peg_atomic_macro import AtomicMacro
from program_atomic_feedback import AtomicRuntime
from train_option_online import atomic_json, atomic_torch

ROOT=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004')
WARM=ROOT/'rule_option_20261105/v2_stopping_rl_31801/INITIAL.pt'
SHARED=ROOT/'joint_continuation_20261106/SHARED_DESIGN_REPLAY.pt'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Joint:
 def __init__(self,assets,actor,step):
  self.assets,self.actor,self.step=assets,actor,step;self.option=None;self.events=[]
 def __call__(self,s):
  if self.option is not None:
   physical=self.option(s)
   if not self.option.events[-1]['boundary']:
    self.events.append(dict(boundary=False,physical=physical,step=self.step()));return physical
   self.option=None
  action=int(self.actor(s))
  if s[44]<.5:physical=action
  elif action==0:physical=rule(s)
  else:
   physical=self.assets.proposals(s)[action-1]
   if physical:
    first=[True]
    def internal(x):
     if first[0]:first[0]=False;return physical
     return rule(x)
    self.option=AtomicMacro(internal,self.assets.reference);physical=self.option(s)
  self.events.append(dict(boundary=True,a=action,physical=physical,step=self.step(),stage=int(s[44])));return physical
 def transitions(self,ts):
  assert len(ts)==len(self.events)
  out=[]
  for t,e in zip(ts,self.events):
   if e['boundary']:out.append(dict(t,a=e['a'],physical_a=e['physical']))
   else:
    out[-1]['r']+=t['r'];out[-1]['duration']+=t['duration'];out[-1]['next']=t['next'];out[-1]['done']=t['done']
  ret=0.
  for t in reversed(out):ret=t['r']+ret;t['return']=ret
  assert sum(t['duration']for t in out)==100
  return out

class Learner:
 def __init__(self,assets,seed,layout="flat"):
  self.layout=layout
  torch.manual_seed(seed);self.assets=assets;self.rng=np.random.default_rng(seed)
  d=torch.load(WARM,weights_only=False);w=d['model'];self.gate=Gate(w['mu'].numpy(),w['sd'].numpy(),31801);self.gate.load_state_dict(w)
  self.mu=torch.tensor(assets.models[1].mu);self.sd=torch.tensor(assets.models[1].sd)
  def net():return nn.Sequential(nn.Linear(83,64),nn.Tanh(),nn.Linear(64,3))
  self.actor=net();nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
  self.q=nn.ModuleList([nn.Sequential(nn.Linear(139 if layout=='semantic' else 125,64 if layout=='semantic' else 72),nn.Tanh(),nn.Linear(64 if layout=='semantic' else 72,1 if layout=='semantic' else 3))for _ in range(2)]);self.target=copy.deepcopy(self.q)
  self.ao=torch.optim.Adam(self.actor.parameters(),lr=1e-4);self.qo=torch.optim.Adam(self.q.parameters(),lr=3e-4);self.updates=0
 def norm(self,s):return(torch.tensor(np.asarray(s),dtype=torch.float32)-self.mu)/self.sd
 def context(self,s):
  s=np.asarray(s);front=s[:,44]<.5;base=np.zeros((len(s),3),np.float32);mask=np.ones((len(s),3),bool)
  with torch.no_grad():
   if front.any():base[front]=self.assets.models[1].probs(self.assets.models[1].x(s[front]))[:,:3].clamp_min(1e-6).log().numpy()
   for i in np.flatnonzero(~front):
    ab=rule(s[i]);cs=self.assets.proposals(s[i]);mask[i]=[True,cs[0]!=ab,cs[1]!=ab]
    xx=torch.tensor(np.stack([gate_features(s[i],ab,c)for c in cs]));base[i,1:]=self.gate(xx).numpy()
  return torch.tensor(base),torch.tensor(mask)
 def probs(self,x,base,mask):return(base+self.actor(x)).masked_fill(~mask,-1e9).softmax(1)
 def choose(self,s,train=False):
  b,m=self.context([s]);x=self.norm([s])
  with torch.no_grad():p=self.probs(x,b,m)[0].numpy()
  if train:
   p=.9*p+.1*m[0].numpy()/float(m[0].sum());return int(self.rng.choice(3,p=p/p.sum()))
  return int(p.argmax())
 def identity(self,s):
  if s[44]<.5:physical=[0,1,2];types=[[0.,0.]]*3
  else:
   physical=[rule(s)]+self.assets.proposals(s);types=[[1.,0.]]+[[1.,0.]if a==0 else[0.,1.]for a in physical[1:]]
  return np.concatenate([np.eye(12,dtype=np.float32)[physical],np.array(types,dtype=np.float32)],axis=1).reshape(-1).tolist()
 def values(self,q,x,identity):
  full=torch.cat([x,identity],1)
  if self.layout=='flat':return q(full)
  # Same all-proposal information as flat. Explicit executed-action identity
  # selects a shared scalar function; identical executable Options have equal Q.
  ids=identity.reshape(-1,3,14);return q(torch.cat([full[:,None,:].expand(-1,3,-1),ids],2)).squeeze(-1)
 def decorate(self,ts,episode):
  for t in ts:
   for key,obs in [('base',t['s']),('next_base',t['next'])]:
    b,m=self.context([obs]);t[key]=b[0].tolist();t[key+'_mask']=m[0].tolist()
   t['identity']=self.identity(t['s']);t['next_identity']=self.identity(t['next']);t['episode']=episode
  return ts
 def train(self,data,steps,arm,episode):
  start=time.monotonic();front=[t for t in data if t['stage']==0];rear=[t for t in data if t['stage']==1];logs=[]
  for _ in range(steps):
   rows=[front[i]for i in self.rng.integers(len(front),size=16)]+[(rear or front)[i]for i in self.rng.integers(len(rear or front),size=48)]
   x=self.norm([t['s']for t in rows]);z=self.norm([t['next']for t in rows]);a=torch.tensor([t['a']for t in rows]);idx=torch.arange(64)
   identity=torch.tensor([t['identity']for t in rows]);next_identity=torch.tensor([t['next_identity']for t in rows]);b=torch.tensor([t['base']for t in rows]);mask=torch.tensor([t['base_mask']for t in rows]);nb=torch.tensor([t['next_base']for t in rows]);nm=torch.tensor([t['next_base_mask']for t in rows])
   with torch.no_grad():
    p=self.probs(z,nb,nm);v=(p*(torch.stack([self.values(q,z,next_identity)for q in self.target]).mean(0)-.01*p.clamp_min(1e-9).log())).sum(1)
    r=torch.tensor([t['r']for t in rows]);done=torch.tensor([t['done']for t in rows]);y=r+(~done)*v
    # gamma=1 finite native horizon. Recent actual suffix returns change FRONT
    # backup only; they are behavior-policy returns, not exact current Q^pi.
    recent=torch.tensor([t['stage']==0 and episode-t['episode']<16 for t in rows])
    if arm=='return_mix':y=torch.where(recent,.5*y+.5*torch.tensor([t['return']for t in rows]),y)
   qloss=sum((self.values(q,x,identity)[idx,a]-y).square().mean()for q in self.q)/2
   self.qo.zero_grad();qloss.backward();nn.utils.clip_grad_norm_(self.q.parameters(),10);self.qo.step()
   p=self.probs(x,b,mask)
   with torch.no_grad():q=torch.stack([self.values(q,x,identity)for q in self.q]).mean(0)
   aloss=(p*(.01*p.clamp_min(1e-9).log()-q)).sum(1).mean()
   successful=torch.tensor([t['outcome']for t in rows]);aloss-=.1*(successful*p[idx,a].clamp_min(1e-9).log()).sum()/successful.sum().clamp_min(1)
   self.ao.zero_grad();aloss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1);self.ao.step()
   with torch.no_grad():
    for q,tq in zip(self.q.parameters(),self.target.parameters()):tq.lerp_(q,.01)
   self.updates+=1;logs.append(dict(q=float(qloss),actor=float(aloss),front_target=float(y[:16].mean()),recent_front=int(recent.sum())))
  return dict(updates=steps,wall_s=time.monotonic()-start,last=logs[-1])
 def state(self):return dict(layout=self.layout,actor=self.actor.state_dict(),q=self.q.state_dict(),target=self.target.state_dict(),ao=self.ao.state_dict(),qo=self.qo.state_dict(),rng=self.rng.bit_generator.state,updates=self.updates)
 def load(self,d):
  assert d['layout']==self.layout
  for k in ['actor','q','target','ao','qo']:getattr(self,k).load_state_dict(d[k])
  self.rng.bit_generator.state=d['rng'];self.updates=d['updates']

def verify_design(out,assets):
 refs=torch.load(SHARED,weights_only=False)['episodes'];chosen=[next(v for v in refs if v['row']['seed']==764008 and v['events'][0]['a']==2 and 'design_warm_g2' in v['row']['path']),next(v for v in refs if v['row']['seed']==764004 and v['events'][0]['a']==1 and 'design_warm_g1' in v['row']['path'])]
 if (out/'COMPATIBILITY.json').exists():return
 rt=AtomicRuntime(out/'compatibility_physics');checks=[]
 try:
  for i,ref in enumerate(chosen):
   result=out/f'COMPATIBILITY_{i}.json'
   if result.exists():checks.append(json.loads(result.read_text()));continue
   start=out/f'COMPATIBILITY_START_{i}.json';assert not start.exists(),'Inspect incomplete compatibility physics';atomic_json(start,dict(source=ref['row']['path']))
   schedule={e['step']:e['a']for e in ref['events']if e['boundary']}
   pol=Joint(assets,lambda s:schedule[len(rt.env.records)],lambda:len(rt.env.records));row,ts,d=rt.run(ref['row']['seed'],pol,'compatibility')
   old=json.load(gzip.open(ref['row']['path'],'rt'));keys=['q','qvel','tcp','peg','action'];equal=all(all(x[k]==y[k]for k in keys)for x,y in zip(old['steps'],d['steps']))
   check=dict(row=row,source=ref['row']['path'],all100physical_steps_equal=equal);atomic_json(result,check);assert equal,'Not a compatible physical execution';checks.append(check)
  atomic_json(out/'COMPATIBILITY.json',checks)
 finally:rt.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['sac','return_mix'],default='sac');p.add_argument('--seed',type=int,default=31901);p.add_argument('--episodes',type=int,default=64);p.add_argument('--train-start',type=int,default=764100);p.add_argument('--dev-start',type=int,default=764300);p.add_argument('--dev-count',type=int,default=16);p.add_argument('--updates',type=int,default=40);p.add_argument('--resume',action='store_true');p.add_argument('--layout',choices=['flat','semantic'],required=True);a=p.parse_args();assert a.arm=='sac';torch.set_num_threads(2)
 a.out.mkdir(parents=True,exist_ok=True);manifest={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()if k!='resume'}
 manifest.update(source_sha=sha(__file__),dependencies={p.name:sha(p)for p in Path(__file__).parent.glob('program_*.py')},macro_sha=sha(Path(__file__).with_name('peg_atomic_macro.py')),warm_sha=sha(WARM),shared_replay_sha=sha(SHARED),shared_design_episodes=96,native100=True,gamma=1,scope='three real grasps and recurrent completed Options; nominal rule is action0 at rear',checkpoint_selection='fixed final; no best dev',candidate='shared executable-action Q versus same-information flat Q; ordinary SAC only, standard parameterized-Q foundation',train=list(range(a.train_start,a.train_start+a.episodes)),dev=list(range(a.dev_start,a.dev_start+a.dev_count)))
 assert not set(manifest['train'])&set(manifest['dev'])
 assets=Assets();m=Learner(assets,a.seed,a.layout);data=torch.load(SHARED,weights_only=False)['data'];history=[];cursor=0
 if a.resume:
  assert json.loads((a.out/'MANIFEST.json').read_text())==manifest
  saved=torch.load(a.out/'CURRENT.pt',weights_only=False);m.load(saved['model']);data=saved['data'];history=saved['history'];cursor=saved['cursor']
 else:
  assert not(a.out/'MANIFEST.json').exists();atomic_json(a.out/'MANIFEST.json',manifest);shutil.copy2(__file__,a.out/Path(__file__).name)
  atomic_torch(a.out/'INITIAL.pt',m.state());atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=0))
 verify_design(a.out,assets)
 rt=AtomicRuntime(a.out/'physics');rt.counter=max([0]+[int(f.name.split('_')[0])for f in rt.out.glob('*.json.gz')]);rt.rows=[h['row']for h in history]
 jobs=[]
 for i in range(a.episodes+1):
  if i in [0,a.episodes//2,a.episodes]:jobs.extend([('eval',i,s)for s in manifest['dev']])
  if i<a.episodes:jobs.append(('train',i,a.train_start+i))
 try:
  for ji in range(cursor,len(jobs)):
   kind,i,seed=jobs[ji];pending=a.out/f'PENDING_{ji:04d}.pt'
   if pending.exists():
    item=torch.load(pending,weights_only=False);assert item['job']==jobs[ji];m.rng.bit_generator.state=item['rng']
   else:
    started=a.out/f'START_{ji:04d}.json';assert not started.exists(),'Inspect incomplete native physics attempt';atomic_json(started,dict(job=jobs[ji]))
    pol=Joint(assets,lambda s:m.choose(s,kind=='train'),lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,f'{kind}_{i}');ts=m.decorate(pol.transitions(ts),i)
    item=dict(row=row,ts=ts,events=pol.events,rng=m.rng.bit_generator.state,job=jobs[ji]);atomic_torch(pending,item)
   log=None
   if kind=='train':data.extend(item['ts']);log=m.train(data,a.updates,a.arm,i)
   history.append(dict(kind=kind,checkpoint=i,row=item['row'],events=item['events'],effective_decisions=len(item['ts']),log=log));atomic_json(a.out/'HISTORY.json',history)
   atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=ji+1))
   if kind=='eval'and seed==manifest['dev'][-1]:atomic_torch(a.out/f'CHECKPOINT_{i}.pt',m.state())
   print(a.arm,a.seed,kind,i,seed,item['row']['success'],len(item['ts']),flush=True)
 finally:rt.close()
if __name__=='__main__':main()
