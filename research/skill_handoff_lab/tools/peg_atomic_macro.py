"""Standard parameterized closed-loop Option: refresh the same correction at
refinement phase boundaries, then continue insertion. No future-success gate.
This changes temporal execution semantics; not a new SAC algorithm.
"""
import numpy as np
from program_atomic_feedback import SemanticOption
class AtomicMacro(SemanticOption):
 def __init__(self,actor,reference,termination='macro'):
  super().__init__(actor,reference,'insertion_queue');self.parameter=None
 def __call__(self,s):
  phase=int(np.argmax(s[64:71]))+1 if s[44]>=.5 else 0
  if self.active and self.parameter is not None and s[42]>=.5:
   # Release only on inherited INSERT queue completion. No length/budget
   # feasibility oracle and no additional physical time.
   if not(phase==7 and s[61]<=0):
    a=self.parameter if phase!=self.phase and phase<7 else 0
    self.events.append(dict(boundary=False,action=a,phase=phase,reason='atomic_reference_refresh'if a else 'macro_continue',remaining=float(s[41])*100));self.phase=phase;return a
  a=super().__call__(s)
  if self.events[-1]['boundary']:self.parameter=a if a>=8 else None
  return a
