"""Non-physics regression checks for action mappings, boundary and legacy weights."""
import json,numpy as np,torch,tempfile
from pathlib import Path
from program_atomic_feedback import AtomicRuntime,SemanticOption
from program_budget_feedback_sac import FeedbackSAC
for a in range(12):assert AtomicRuntime.execution_action(None,a)==((a-6,True)if a>=8 else(a,False))
s=np.zeros(83,dtype=np.float32);s[44]=1;s[42]=1;s[41]=.2;s[61]=.1;s[67]=1
for mode,expected in [('phase',[True,False,True]),('insertion_queue',[True,False,False]),('chunk',[True,True,True])]:
 p=SemanticOption(lambda s:10,lambda s:0,mode);p(s);p(s);z=s.copy();z[67]=0;z[68]=1;p(z);assert [e['boundary']for e in p.events]==expected
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004');e=json.loads((r/'decoupled_interface_20261105/LATE_REPLAN_VALUES.json').read_text());m=FeedbackSAC.load(r/'decoupled_interface_20261105/relative_bc64/FINAL.pt')
with torch.no_grad():
 x=m.x([e['obs']]);p=m.probs(x)[0].numpy();q=torch.stack([v(x)for v in m.q]).mean(0)[0].numpy()
assert np.max(np.abs(p-e['actor_probs']))<1e-6 and np.max(np.abs(q-e['mean_q']))<1e-6
m=FeedbackSAC.load(r/'atomic_handoff_20261105/warm31701/INITIAL.pt');assert m.nobs==83 and m.nactions==12
for stage in [0,1]:
 z=np.zeros(83);z[44]=stage
 with torch.no_grad():p=m.probs(m.x([z]))[0].numpy()
 assert np.isclose(p.sum(),1) and (stage or (p[3:]==0).all())
print('PASS: legacy frozen outputs;12action mapping; semantic boundary;83D masks/checkpoint')
