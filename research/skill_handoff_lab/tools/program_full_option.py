"""Shared full executable catalogue. Standard interface, no novelty claim.
Rear selector0 is rule CHUNK. Selectors1..12 execute physical0..11;
nonzero physical options retain AtomicMacro and reopen on its actual release.
"""
import numpy as np
from train_joint_continuation import Joint as ProposalJoint, Learner as WarmLearner
from peg_atomic_macro import AtomicMacro
from program_rule_option import rule
class FullJoint(ProposalJoint):
 def __call__(self,s):
  if self.option is not None:
   physical=self.option(s)
   if not self.option.events[-1]['boundary']:
    self.events.append(dict(boundary=False,physical=physical,step=self.step()));return physical
   self.option=None
  a=int(self.actor(s));mask=legal(s);assert mask[a],(a,mask)
  if s[44]<.5:physical=a
  elif a==0:physical=rule(s)
  else:
   physical=a-1
   if physical:
    first=[True]
    def internal(x):
     if first[0]:first[0]=False;return physical
     return rule(x)
    self.option=AtomicMacro(internal,self.assets.reference);physical=self.option(s)
  self.events.append(dict(boundary=True,a=a,physical=physical,step=self.step(),stage=int(s[44])));return physical

def legal(s):
 m=np.zeros(13,bool)
 if s[44]<.5:m[:3]=True
 else:
  m[:]=True
  if rule(s)==0:m[1]=False # same physical-zero chunk as rule0: exact alias
  if s[42]<.5 or s[41]*100<4:m[2:]=False # no new correction after grasp loss / <one feedback chunk
 return m
class Warm:
 def __init__(self,assets):self.assets=assets;self.model=WarmLearner(assets,31901)
 def choose(self,s):
  a=self.model.choose(s)
  if s[44]<.5 or a==0:return a
  physical=self.assets.proposals(s)[a-1];b=physical+1
  return b if legal(s)[b] else 0
