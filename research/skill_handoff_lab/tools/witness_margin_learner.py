"""Measured-reference half-margin retention: one fixed robustness ablation.
Standard convex dual optimization; no new RL objective or safety theorem.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from torch.nn.utils import parameters_to_vector,vector_to_parameters
from witness_learner import WitnessLearner
class MarginWitnessLearner(WitnessLearner):
 def __init__(self,assets,seed,arm='witness_margin'):
  assert arm=='witness_margin';self.ref_margins={};super().__init__(assets,seed,'witness_project');self.arm=arm
 def set_graph(self,g):
  super().set_graph(g)
  with torch.no_grad():z=self._logits();margin=z.masked_fill(~self.ga,-1e9).max(1).values-z.masked_fill(self.ga,-1e9).max(1).values
  for i,n in enumerate(g['nodes']):
   if n['history_sha']not in self.ref_margins:self.ref_margins[n['history_sha']]=.5*max(0.,float(margin[i]))if bool((self.gm[i]&~self.ga[i]).any())else 0.
  self.targets=torch.tensor([self.ref_margins[n['history_sha']]for n in g['nodes']]);assert self._valid()
 def _valid(self):
  with torch.no_grad():
   z=self._logits();margin=z.masked_fill(~self.ga,-1e9).max(1).values-z.masked_fill(self.ga,-1e9).max(1).values
   return bool(self.ga[torch.arange(len(self.gs)),z.argmax(1)].all()and (margin>=self.targets-1e-6).all())
 def state(self):
  d=super().state();d['reference_margins']=self.ref_margins;d['margin_fraction']=.5;return d
 def load(self,d):
  self.ref_margins=dict(d.get('reference_margins',{}));super().load(d)
 def _after(self,opt,args,kwargs):
  if self.graph is None:return
  tic=time.monotonic();proposal=parameters_to_vector(self.actor.parameters()).detach().clone();u=proposal-self.old;direction=u.clone();rows={};qp_ok=True
  for _ in range(5):
   with torch.no_grad():
    z=self._logits();margin=z.masked_fill(~self.ga,-1e9).max(1).values-z.masked_fill(self.ga,-1e9).max(1).values;bad=torch.where((margin<self.targets-1e-6)|~self.ga[torch.arange(len(self.gs)),z.argmax(1)])[0]
    if not len(bad):break
    violations=[(int(i),int(z[i].masked_fill(self.ga[i],-1e9).argmax()))for i in bad]
   vector_to_parameters(self.old.clone(),self.actor.parameters())
   with torch.enable_grad():
    z=self._logits()
    for i,b in violations:
     good=int(z[i].masked_fill(~self.ga[i],-1e9).argmax());key=(i,good,b)
     if key in rows:continue
     margin=z[i,good]-z[i,b]
     grad=torch.cat([g.reshape(-1)for g in torch.autograd.grad(margin,tuple(self.actor.parameters()),retain_graph=True)]).detach().double().numpy()
     scale=max(float(np.linalg.norm(grad)),1e-12)
     rows[key]=(grad/scale,(float(margin.detach())-float(self.targets[i])-1e-5)/scale)
   G=np.stack([x[0]for x in rows.values()]);c=G@u.double().numpy()+np.array([x[1]for x in rows.values()]);H=G@G.T
   def dual(v):return .5*v@H@v+c@v,H@v+c
   fit=minimize(dual,np.zeros(len(c)),jac=True,bounds=[(0,None)]*len(c),method='L-BFGS-B',options=dict(maxiter=300,ftol=1e-15,gtol=1e-10))
   qp_ok=qp_ok and bool(fit.success);direction=u+torch.tensor(G.T@fit.x,dtype=u.dtype);vector_to_parameters(self.old+direction,self.actor.parameters())
  scale=1.
  for _ in range(14):
   if self._valid():break
   scale*=.5;vector_to_parameters(self.old+scale*direction,self.actor.parameters())
  if not self._valid():scale=0.;vector_to_parameters(self.old.clone(),self.actor.parameters())
  assert self._valid()
  self.projections.append(dict(cuts=len(rows),scale=scale,proposed_norm=float(u.norm()),accepted_norm=float((parameters_to_vector(self.actor.parameters()).detach()-self.old).norm()),wall_s=time.monotonic()-tic,dual_converged=qp_ok))
