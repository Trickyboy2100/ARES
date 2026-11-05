"""Focused regressions for rule fallback and complete Option handback semantics."""
import numpy as np
from program_rule_option import SingleDeviation

class Assets:
    def front(self,s):return 2
    def proposals(self,s):return [0,10]
    def reference(self,s):return 0

def state(phase=4):
    s=np.zeros(83,dtype=np.float32);s[44]=1;s[42]=1;s[41]=.2;s[49]=phase/7;s[61]=.5;s[64+phase-1]=1
    return s

def test_contracts():
    clock=[79];p=SingleDeviation(Assets(),lambda *args:None,lambda:clock[0])
    assert p(state())==10, 'Reject proposal must follow rule, not physical action zero'
    p=SingleDeviation(Assets(),lambda *args:0,lambda:clock[0])
    assert p(state())==0
    clock[0]=83;assert p(state())==10
    assert p.metadata()['option_duration']==4
    clock[0]=60;p=SingleDeviation(Assets(),lambda *args:1,lambda:clock[0])
    assert p(state(4))==10
    clock[0]=64;assert p(state(4))==0
    clock[0]=68;assert p(state(5))==10
    clock[0]=80;assert p(state(7))==0
    s=state(7);s[61]=0;clock[0]=84;assert p(s)==0
    assert p.metadata()['release']==84 and p.metadata()['option_duration']==24
    assert p.events[-1]['origin']=='baseline_after_release'
    print('PASS: rule fallback, zero-duration accounting, phase refresh, inherited queue release')

if __name__=='__main__':test_contracts()
