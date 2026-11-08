"""Executed-witness policy updates: soft set imitation versus projected Adam.
Gradient projection is established (e.g. GEM); no generic novelty/safety claim.
"""
import copy,time
import numpy as np
import torch
from continuation_learner import ContinuationLearner
from torch.nn.utils import parameters_to_vector,vector_to_parameters
class WitnessLearner(ContinuationLearner):
 def __init__(self,assets,seed,arm):
  assert arm in ('witness_soft','witness_project')
  super().__init__(assets,seed,arm);self.graph=None;self.projections=[]
  if arm=='witness_project':
   self.ao.register_step_pre_hook(self._before)
   self.ao.register_step_post_hook(self._after)
 def set_graph(self,g):
  self.graph=g;self.gs=[n['s']for n in g['nodes']];self.gx=self.norm(self.gs);self.gb,self.gm=self.context(self.gs)
  self.ga=torch.zeros(len(self.gs),13,dtype=torch.bool)
  for i,n in enumerate(g['nodes']):self.ga[i,n['allowed']]=True
  assert (self.ga&~self.gm).sum()==0
  if self.arm=='witness_project':assert self._valid(), 'Reference witness constraints must be feasible before projection'
 def _logits(self):return self.logits(self.gx,self.gb,self.gm,self.gs)
 def _valid(self):
  with torch.no_grad():return bool(self.ga[torch.arange(len(self.gs)),self._logits().argmax(1)].all())
 def _before(self,opt,args,kwargs):self.old=parameters_to_vector(self.actor.parameters()).detach().clone()
 def _after(self,opt,args,kwargs):
  if self.graph is None:return
  tic=time.monotonic();proposal=parameters_to_vector(self.actor.parameters()).detach().clone();direction=proposal-self.old;original_norm=float(direction.norm());cuts=0
  # Sequential projection of the actual Adam displacement onto local margin
  # halfspaces, followed by an exact nonlinear action-agreement check.
  for _ in range(3):
   with torch.no_grad():
    logits=self._logits();margin=logits.masked_fill(~self.ga,-1e9).max(1).values-logits.masked_fill(self.ga,-1e9).max(1).values
    bad=torch.where(~self.ga[torch.arange(len(self.gs)),logits.argmax(1)])[0]
   if not len(bad):break
   worst=bad[margin[bad].argsort()[:8]];vector_to_parameters(self.old.clone(),self.actor.parameters())
   with torch.enable_grad():
    logits=self._logits();margins=logits.masked_fill(~self.ga,-1e9).max(1).values-logits.masked_fill(self.ga,-1e9).max(1).values
    for i in worst:
     grad=torch.cat([g.reshape(-1)for g in torch.autograd.grad(margins[i],tuple(self.actor.parameters()),retain_graph=True)])
     deficit=1e-7-float(margins[i].detach()+grad@direction)
     if deficit>0:direction=direction+deficit*grad/(grad.square().sum()+1e-12);cuts+=1
   vector_to_parameters((self.old+direction).detach(),self.actor.parameters())
  scale=1.
  for _ in range(14):
   if self._valid():break
   scale*=.5;vector_to_parameters(self.old+scale*direction,self.actor.parameters())
  if not self._valid():scale=0.;vector_to_parameters(self.old.clone(),self.actor.parameters())
  assert self._valid()
  self.projections.append(dict(cuts=cuts,scale=scale,proposed_norm=original_norm,accepted_norm=float((parameters_to_vector(self.actor.parameters()).detach()-self.old).norm()),wall_s=time.monotonic()-tic))
 def evidence_loss(self):
  zero=self.actor[-1].weight.sum()*0;rank=zero
  if self.pairs:
   rows=[self.pairs[i]for i in self.rng.integers(len(self.pairs),size=32)];ss=[p['s']for p in rows];b,m=self.context(ss);z=self.logits(self.norm(ss),b,m,ss);idx=torch.arange(len(rows));win=torch.tensor([p['action']if p['delta']>0 else p['baseline']for p in rows]);lose=torch.tensor([p['baseline']if p['delta']>0 else p['action']for p in rows]);rank=torch.nn.functional.softplus(-(z[idx,win]-z[idx,lose])).mean()
  support=zero
  if self.graph is not None and self.arm=='witness_soft':
   logp=self._logits().log_softmax(1);support=-logp.masked_fill(~self.ga,-1e9).logsumexp(1).mean()
  return .1*rank+support,dict(pair_rank=float(rank),support_loss=float(support))
 def train(self,data,steps,episode):
  start=len(self.projections);out=super().train(data,steps,episode);logs=self.projections[start:]
  if logs:out['projection']=dict(attempts=len(logs),cuts=sum(x['cuts']for x in logs),zero_steps=sum(x['scale']==0 for x in logs),mean_scale=float(np.mean([x['scale']for x in logs])),mean_accepted_norm=float(np.mean([x['accepted_norm']for x in logs])),wall_s=sum(x['wall_s']for x in logs),all_stored_actions_admissible=self._valid())
  return out
 def state(self):
  d=super().state();d['projections']=self.projections;d['witness_graph']=self.graph;return d
 def load(self,d):
  super().load(d);self.projections=copy.deepcopy(d.get('projections',[]))
  if d.get('witness_graph')is not None:self.set_graph(d['witness_graph'])
