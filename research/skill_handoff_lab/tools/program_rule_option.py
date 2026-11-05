"""One complete deviation Option, then frozen strong rule. Native executor unchanged."""
import json,hashlib
from pathlib import Path
import numpy as np
from program_budget_feedback_sac import FeedbackSAC
from peg_atomic_macro import AtomicMacro
from train_option_online import PARENT
ROOT=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/atomic_handoff_20261105')
MODELS=[ROOT/'macro_bc64/CHECKPOINT_32.pt',ROOT/'macro_seed31702/CHECKPOINT_64.pt']
def rule(s):return 10 if s[44]>=.5 and s[49]<=6/7+1e-5 and s[41]<=.28 else 0
class Assets:
 def __init__(self):self.models=[FeedbackSAC.load(x)for x in MODELS];self.parent=FeedbackSAC.load(PARENT)
 def front(self,s):return self.models[1].choose(s)
 def reference(self,s):
  a=self.parent.choose(s[:64]);return a+6 if a in [2,3,4,5]else a
 def proposals(self,s):return [m.choose(s)for m in self.models]
def gate_features(s,ab,ac):return np.r_[s,np.eye(12)[ab],np.eye(12)[ac]].astype(np.float32)
class SingleDeviation:
 def __init__(self,assets,selector,step_fn):self.assets=assets;self.selector=selector;self.step_fn=step_fn;self.used=False;self.option=None;self.events=[];self.entry=None;self.release=None
 def __call__(self,s):
  step=self.step_fn()
  if s[44]<.5:return self.assets.front(s)
  ab=rule(s)
  if self.option is not None:
   a=self.option(s);ev=self.option.events[-1]
   if ev['boundary']:
    self.release=step;self.option=None;origin='baseline_after_release'
   else:origin='candidate_continuation'
   self.events.append(dict(step=step,origin=origin,action=a));return a
  if self.used:self.events.append(dict(step=step,origin='baseline_after_deviation',action=ab));return ab
  proposals=self.assets.proposals(s);choice=self.selector(s,ab,proposals,step)
  if choice is None:self.events.append(dict(step=step,origin='baseline',action=ab));return ab
  ac=proposals[choice];self.used=True;self.entry=dict(step=step,s=s.tolist(),ab=ab,ac=ac,proposal=choice,features=gate_features(s,ab,ac).tolist())
  if ac==0:a=0;self.release=step # zero Option is one physical feedback chunk; fix measured duration on next call
  else:
   first=[True]
   def option_actor(x):
    if first[0]:first[0]=False;return ac
    return rule(x)
   self.option=AtomicMacro(option_actor,self.assets.reference);a=self.option(s)
  self.events.append(dict(step=step,origin='candidate_entry',action=a));return a
 def metadata(self):
  release=self.release
  if self.entry and self.entry['ac']==0:
   release=next((e['step']for e in self.events if e['step']>self.entry['step']),100)
  return dict(entry=self.entry,release=release,option_duration=(release if release is not None else 100)-self.entry['step']if self.entry else 0,events=self.events,continuation='frozen strong rule after complete Option; inherited frozen reference inside non-atomic Option',at_most_one=True)
def prefix_hash(d,n):
 rows=[{k:x[k]for k in ['q','qvel','tcp','peg','action']}for x in d['steps'][:n]]
 return hashlib.sha256(json.dumps(dict(initial=d['initial'],steps=rows),sort_keys=True).encode()).hexdigest()
