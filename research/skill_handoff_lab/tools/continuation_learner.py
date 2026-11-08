"""Shared flat SMDP SAC with version-bound empirical preference/continuation study.
Standard ranking and empirical KL are not claimed as independent novelty.
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
class ContinuationLearner:
 def __init__(self,assets,seed,arm):
  self.assets=assets;self.warm=Warm(assets);self.arm=arm;self.rng=np.random.default_rng(seed)
  # Warm Gate constructor resets global torch RNG. Set fresh-network seed AFTER it.
  torch.manual_seed(seed)
  self.mu=self.warm.model.mu;self.sd=self.warm.model.sd
  self.actor=nn.Sequential(nn.Linear(83,96),nn.Tanh(),nn.Linear(96,17));nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)
  self.q=nn.ModuleList([nn.Sequential(nn.Linear(83,96),nn.Tanh(),nn.Linear(96,13))for _ in range(2)]);self.target=copy.deepcopy(self.q)
  self.ao=torch.optim.Adam(self.actor.parameters(),lr=1e-4);self.qo=torch.optim.Adam(self.q.parameters(),lr=3e-4);self.updates=0;self.episode=0;self.pairs=[];self.anchors=[];self.reference_version=None
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
  z=self.actor(x);rear=z[:,3:16]
  front=torch.cat([z[:,:3],torch.zeros(len(z),10)],1);delta=torch.where(torch.tensor(np.asarray(ss)[:,44]<.5)[:,None],front,rear)
  return(b+delta).masked_fill(~mask,-1e9)
 def probs(self,x,b,mask,ss):return self.logits(x,b,mask,ss).softmax(1)
 def choose(self,s,train=False):
  b,m=self.context([s])
  with torch.no_grad():p=self.probs(self.norm([s]),b,m,[s])[0].numpy()
  if train:
   ratio=min((self.episode+1)/64.,1.);prior=np.eye(13)[self.warm.choose(s)]
   p=(1-ratio)*prior+ratio*p;p=.9*p+.1*m[0].numpy()/float(m[0].sum());return int(self.rng.choice(13,p=p/p.sum()))
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
   if self.arm=='ranking':
    # Same data/state/action authority: direct terminal-outcome scorer.
    # Both successes and failures are used; no bootstrapped value enters actor.
    outcome=torch.tensor([t['outcome']for t in rows]);aloss=nn.functional.binary_cross_entropy_with_logits(self.logits(x,b,mask,ss)[idx,a],outcome)
   else:
    aloss=(p*(.01*p.clamp_min(1e-9).log()-q)).sum(1).mean();w=torch.tensor([t['outcome']for t in rows]);aloss-=.1*(w*p[idx,a].clamp_min(1e-9).log()).sum()/w.sum().clamp_min(1)
   extra, detail=self.evidence_loss()
   aloss=aloss+extra
   self.ao.zero_grad();aloss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1);self.ao.step()
   with torch.no_grad():
    for q,t in zip(self.q.parameters(),self.target.parameters()):t.lerp_(q,.01)
   self.updates+=1;logs.append(dict(q=float(qloss),actor=float(aloss),**detail))
  return dict(updates=steps,wall_s=time.monotonic()-tic,last=logs[-1])
 def state(self):return dict(arm=self.arm,actor=self.actor.state_dict(),q=self.q.state_dict(),target=self.target.state_dict(),ao=self.ao.state_dict(),qo=self.qo.state_dict(),rng=self.rng.bit_generator.state,updates=self.updates,reference_version=self.reference_version)
 def load(self,d):
  assert d['arm']==self.arm
  for k in ['actor','q','target','ao','qo']:getattr(self,k).load_state_dict(d[k])
  self.rng.bit_generator.state=d['rng'];self.updates=d['updates']

 def evidence_loss(self):
  zero=self.actor[-1].weight.sum()*0
  if self.arm=='sac' or not self.pairs:return zero,{}
  rows=[self.pairs[i] for i in self.rng.integers(len(self.pairs),size=32)]
  ss=[t['s']for t in rows];b,m=self.context(ss);logits=self.logits(self.norm(ss),b,m,ss)
  winner=torch.tensor([t['action'] if t['delta']>0 else t['baseline']for t in rows]);loser=torch.tensor([t['baseline']if t['delta']>0 else t['action']for t in rows]);idx=torch.arange(len(rows))
  ranking=torch.nn.functional.softplus(-(logits[idx,winner]-logits[idx,loser])).mean()
  if self.arm=='ranking':return .1*ranking,dict(pair_rank=float(ranking))
  # Empirical continuation trust region, NOT a state-wise safety guarantee.
  # Anchors come from successful full suffixes under the identified reference.
  rr=[self.anchors[i]for i in self.rng.integers(len(self.anchors),size=32)] if self.anchors else []
  kl=zero
  if rr:
   ss=[r['s']for r in rr];b,m=self.context(ss);p=self.probs(self.norm(ss),b,m,ss);q=torch.tensor([r['reference_p']for r in rr]);kl=(q*(q.clamp_min(1e-9).log()-p.clamp_min(1e-9).log())).sum(1).mean()
  # Suppress transfer of old measured preferences when their suffix has drifted.
  # KL has its own restoring gradient; the gate is detached and not a learned head.
  trust=(kl.detach()<.05).float()
  return .1*trust*ranking+kl,dict(pair_rank=float(ranking),continuation_kl=float(kl),pair_active=float(trust))
 def set_evidence(self,pairs,anchors,version):
  self.pairs=[p for p in pairs if p['delta']!=0];self.anchors=anchors;self.reference_version=version
