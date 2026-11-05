"""Execution/return invariants; these tests are not new physical evidence."""
import numpy as np
from train_joint_continuation import Joint
class Assets:
 def proposals(self,s):return [2,0]
 def reference(self,s):return 0
s=np.zeros(83,np.float32);s[44]=1;s[42]=1;s[64+3]=1;s[61]=.5
choices=iter([1,2,0]);clock=[50];p=Joint(Assets(),lambda s:next(choices),lambda:clock[0])
assert p(s)==2
clock[0]=54;assert p(s)==0 and not p.events[-1]['boundary']
s[64:71]=0;s[70]=1;s[61]=0;clock[0]=58
assert p(s)==0 and p.events[-1]['boundary'] and p.events[-1]['a']==2
clock[0]=62;assert p(s)==10 and p.events[-1]['boundary']
ts=[dict(s=s.tolist(),next=s.tolist(),a=e['physical'],r=r,duration=d,done=i==3,stage=1,outcome=1)for i,(e,r,d)in enumerate(zip(p.events,[.1,.2,.3,.4],[50,4,4,42]))]
out=p.transitions(ts);assert len(out)==3 and out[0]['duration']==54 and abs(out[0]['return']-1)<1e-6
assert sum(t['duration']for t in out)==100 and out[-1]['done']
print('PASS: recurrent release; rule versus physical-zero; actual duration and MC aggregation')
