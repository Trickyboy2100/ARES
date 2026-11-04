"""Contract checks: actual queue features, reopen event, return conservation."""
import numpy as np
from program_continuation_option import ContinuationOption
s=np.zeros(64,dtype=np.float32);s[44]=1;s[42]=1;s[49]=4/7;s[61]=.12
p=ContinuationOption(lambda _:4,lambda _:0,'insertion_queue')
assert p(s)==4
s[49]=5/7;assert p(s)==0  # crossing refinement is NOT termination
s[49]=1;s[61]=.02;assert p(s)==0
s[61]=0;assert p(s)==4 and p.events[-1]['boundary']
s[61]=.1;s[42]=0;assert p(s)==4 and p.events[-1]['reason']=='grasp_lost'
ts=[dict(s=[],next=[],a=e['action'],r=float(i),done=i==4,duration=20,stage=1) for i,e in enumerate(p.events)]
a=p.transitions(ts)
assert len(a)==3 and sum(t['r'] for t in a)==10 and sum(t['duration'] for t in a)==100
assert [t['duration'] for t in a]==[60,20,20]
print('PASS: active continuation, event reopening, SMDP duration and reward conservation')
