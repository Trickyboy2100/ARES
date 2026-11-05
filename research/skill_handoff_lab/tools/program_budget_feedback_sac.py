"""Finite feedback SAC and experimental budget-resolved handoff policy iteration.
Common inputs/actions/dense shaping. Auxiliary estimates affect predecessor
Bellman targets only in explicit comparator/candidate modes; no novelty claim.
"""
import copy,time
import numpy as np
import torch
from torch import nn
from program_peg_feedback import NOBS,ACTIONS,features,potential

class Net(nn.Module):
 def __init__(self,out,nobs=NOBS):
  super().__init__();self.net=nn.Sequential(nn.Linear(nobs,96),nn.LayerNorm(96),nn.ReLU(),nn.Linear(96,96),nn.LayerNorm(96),nn.ReLU(),nn.Linear(96,out))
 def forward(self,x):return self.net(x)

class FeedbackSAC:
 def __init__(self,mode,seed,mu,sd,recipe="original",nobs=NOBS,nactions=8):
  self.nobs=nobs;self.nactions=nactions
  self.recipe=recipe
  assert mode in ['ordinary','suffix','budget','budget_blind'];self.mode=mode;self.seed=seed;torch.manual_seed(seed);self.rng=np.random.default_rng(seed)
  self.mu=mu;self.sd=sd;self.q=nn.ModuleList([Net(nactions,nobs),Net(nactions,nobs)]);self.target=copy.deepcopy(self.q);self.actor=Net(nactions,nobs);self.completion=Net(25,nobs);self.scalar=Net(1,nobs)
  self.qopt=torch.optim.Adam(self.q.parameters(),lr=3e-4);self.aopt=torch.optim.Adam(self.actor.parameters(),lr=3e-4);self.hopt=torch.optim.Adam(list(self.completion.parameters())+list(self.scalar.parameters()),lr=3e-4)
  self.updates=0;self.history=[];self.alpha=.01
 def x(self,rows):return(torch.tensor(np.asarray(rows),dtype=torch.float32)-torch.tensor(self.mu))/torch.tensor(self.sd)
 def mask(self,x):
  m=torch.ones((len(x),self.nactions),dtype=torch.bool);m[x[:,44]<.5,3:]=False;return m
 def probs(self,x):return self.actor(x).masked_fill(~self.mask(x),-1e9).softmax(-1)
 def choose(self,s,stochastic=False):
  with torch.no_grad():p=self.probs(self.x([s]))[0].numpy()
  return int(self.rng.choice(self.nactions,p=p/p.sum()))if stochastic else int(p.argmax())
 def probability(self,x,budget):
  z=x.clone()
  if self.mode=='budget_blind':z[:,41]=0
  hazards=self.completion(z).sigmoid();n=torch.ceil(budget/4).long().clamp(0,25);j=torch.arange(25)[None];survival=torch.where(j<n[:,None],1-hazards,torch.ones_like(hazards)).prod(1)
  return 1-survival
 def train(self,data,labels,steps,bc=False,*,success_bc=None,cql=None,successor_only=False,bc_anchor=None,intervention_pairs=None,actor_rule="sac"):
  assert actor_rule in ("sac", "awac", "measured_gap", "measured_preference")
  # Independent controls retain the historical bc=True/False behavior by default.
  bc_weight=(.2 if bc else 0.) if success_bc is None else float(success_bc)
  cql_weight=(.1 if bc and self.recipe=='cql_warm' else 0.) if cql is None else float(cql)
  tic=time.monotonic();hist={}
  for _ in range(steps):
   # Equal front/rear sampling prevents long suffixes drowning predecessor RL.
   front=[t for t in data if t['stage']==0];rear=[t for t in data if t['stage']==1]
   rows=[] if successor_only else [front[i]for i in self.rng.integers(len(front),size=16)]
   pool=rear or front;rows +=[pool[i]for i in self.rng.integers(len(pool),size=64 if successor_only else 48)]
   x=self.x([t['s']for t in rows]);z=self.x([t['next']for t in rows]);a=torch.tensor([t['a']for t in rows]);r=torch.tensor([t['r']for t in rows]);done=torch.tensor([t['done']for t in rows]);idx=torch.arange(len(rows));frontmask=(x[:,44]<.5)&~done
   # Both maturity controls also fit the same auxiliary networks: same capacity,
   # labels and updates; ordinary never uses them to select/deploy actions.
   hrows=[labels[i]for i in self.rng.integers(len(labels),size=64)]
   hx=self.x([t['s']for t in hrows]);hz=hx.clone()
   if self.mode=='budget_blind':hz[:,41]=0
   logits=self.completion(hz);duration=torch.tensor([t['duration']for t in hrows]);event=torch.tensor([t['event']for t in hrows]);at=((duration-1)//4).clamp(0,24).long();grid=torch.arange(25)[None]
   observed=grid<=at[:,None];target=event[:,None]&(grid==at[:,None]);loss=(nn.functional.binary_cross_entropy_with_logits(logits,target.float(),reduction='none')*observed).sum(1).mean()/10
   sl=nn.functional.binary_cross_entropy_with_logits(self.scalar(hx)[:,0],event.float());hl=loss+sl
   self.hopt.zero_grad();hl.backward();self.hopt.step()
   with torch.no_grad():
    tq=torch.stack([q(z)for q in self.target]).mean(0);p=self.probs(z);nv=(p*(tq-self.alpha*p.clamp_min(1e-9).log())).sum(1)
    if self.mode!='ordinary':
     prob=self.scalar(z)[:,0].sigmoid() if self.mode=='suffix' else self.probability(z,torch.tensor([t['next'][41]*100 for t in rows]))
     # Undo shared potential to produce a terminal-success-consistent target.
     phi=torch.tensor([-.2*(min(np.linalg.norm(t['next'][45:48])/.15,3)+.2*t['next'][48])for t in rows])
     nv=torch.where(frontmask,prob-phi,nv)
    targetq=r+(~done).float()*nv
   qs=[q(x)for q in self.q];ql=sum((q[idx,a]-targetq).square().mean()for q in qs)/2
   if cql_weight:
    mask=self.mask(x);ql=ql+cql_weight*sum((q.masked_fill(~mask,-1e9).logsumexp(1)-q[idx,a]).mean()for q in qs)/2
   self.qopt.zero_grad();ql.backward();nn.utils.clip_grad_norm_(self.q.parameters(),10);self.qopt.step()
   with torch.no_grad():av=torch.stack([q(x)for q in self.q]).mean(0)
   p=self.probs(x);al=(p*(self.alpha*p.clamp_min(1e-9).log()-av)).sum(1).mean()
   if bc_weight:
    if bc_anchor is None:
     successful=torch.tensor([t['outcome']for t in rows],dtype=torch.float32)
     al=al-bc_weight*(successful*p[idx,a].clamp_min(1e-9).log()).sum()/successful.sum().clamp_min(1)
    else:
     assert bc_anchor and all(t['outcome'] and t['stage']==1 for t in bc_anchor)
     anchors=[bc_anchor[i] for i in self.rng.integers(len(bc_anchor),size=64)]
     ap=self.probs(self.x([t['s'] for t in anchors]));aa=torch.tensor([t['a'] for t in anchors])
     al=al-bc_weight*ap[torch.arange(64),aa].clamp_min(1e-9).log().mean()
   if intervention_pairs:
    # Same extra decision-state batch for all comparators. Outcomes are training
    # supervision only, never consulted by choose(). Prefixes are reexecuted.
    pr=[intervention_pairs[i] for i in self.rng.integers(len(intervention_pairs),size=64)]
    px=self.x([v['s'] for v in pr]);pa=torch.tensor([v['a'] for v in pr]);ii=torch.arange(64)
    pp=self.probs(px)
    with torch.no_grad():
     pq=torch.stack([q(px) for q in self.q]).mean(0)
     if actor_rule=='measured_gap':
      # Local policy-improvement target under the declared common continuation.
      # No claim that this rollout advantage equals Q of the current actor.
      pq[ii,pa]=pq[:,0]+torch.tensor([v['delta'] for v in pr],dtype=torch.float32)
    if actor_rule=='measured_preference':
     delta=torch.tensor([v['delta'] for v in pr],dtype=torch.float32)
     # Zero differences provide no direction: do not label every successful
     # intervention desirable. Direct conditional log-odds avoids p(a) scaling.
     logits=self.actor(px);odds=logits[ii,pa]-logits[:,0]
     extra=(delta.abs()*nn.functional.softplus(-delta.sign()*odds)).mean()
    elif actor_rule=='awac':
     with torch.no_grad():
      observed_a=torch.where(ii%2==0,torch.zeros_like(pa),pa)
      adv=pq[ii,observed_a]-(pp.detach()*pq).sum(1);weight=(adv/.1).exp().clamp(max=20)
     extra=-(weight*pp[ii,observed_a].clamp_min(1e-9).log()).mean()
    else:extra=(pp*(self.alpha*pp.clamp_min(1e-9).log()-pq)).sum(1).mean()
    al=(al+extra)/2
   self.aopt.zero_grad();al.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),10);self.aopt.step()
   with torch.no_grad():
    for param,t in zip(self.q.parameters(),self.target.parameters()):t.lerp_(param,.005)
   self.updates+=1;hist=dict(update=self.updates,q_loss=float(ql),actor_loss=float(al),hazard_loss=float(loss),suffix_loss=float(sl),front_target=float(targetq[:16].mean()),n=len(data),labels=len(labels))
  hist['wall_s']=time.monotonic()-tic;self.history.append(hist);return hist
 def save(self,path):
  keys=['q','target','actor','completion','scalar','qopt','aopt','hopt'];torch.save(dict(nobs=self.nobs,nactions=self.nactions,mode=self.mode,recipe=self.recipe,seed=self.seed,mu=self.mu,sd=self.sd,updates=self.updates,history=self.history,rng=self.rng.bit_generator.state,weights={k:getattr(self,k).state_dict()for k in keys}),path)
 @classmethod
 def load(cls,path):
  d=torch.load(path,map_location='cpu',weights_only=False);m=cls(d['mode'],d['seed'],d['mu'],d['sd'],d.get('recipe','original'),d.get('nobs',NOBS),d.get('nactions',8))
  for k,v in d['weights'].items():getattr(m,k).load_state_dict(v)
  m.updates=d['updates'];m.history=d['history'];m.rng.bit_generator.state=d['rng'];return m

def completion_labels(d,ts):
 # Event: first success that persists to the actual native terminal state.
 # Failures right-censored at remaining budget, not labeled impossible forever.
 success=[bool(s['evaluation']['success'][0])for s in d['steps']];event=bool(success[-1]);stable=next((i+1 for i in range(100)if all(success[i:])),None)if event else None
 return [dict(s=t['s'],duration=max(1,(stable-t['step'])if event else 100-t['step']),event=event,scene=t['scene'],source=t['source'])for t in ts if t['stage']==1]

def normalization(data):
 x=np.array([t['s']for t in data]);mu=x.mean(0).astype(np.float32);sd=np.maximum(x.std(0),.05).astype(np.float32);mu[[41,44]]=0;sd[[41,44]]=1;return mu,sd
