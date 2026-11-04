import sys,json,importlib.util,argparse
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()/'research/skill_handoff_lab/tools'))
from program_budget_feedback_sac import FeedbackSAC
parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);root=parser.parse_args().parent
spec=importlib.util.spec_from_file_location('old',root/'source/program_budget_feedback_sac.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
d=json.loads((root/'TRANSITIONS.json').read_text());l=json.loads((root/'LABELS.json').read_text());torch.set_num_threads(2)
for bc in [False,True]:
 a=old.FeedbackSAC.load(root/'FINAL.pt');b=FeedbackSAC.load(root/'FINAL.pt');a.train(d,l,2,bc=bc);b.train(d,l,2,bc=bc)
 assert all(torch.equal(x,y) for k in ['q','actor','target','scalar','completion'] for x,y in zip(getattr(a,k).parameters(),getattr(b,k).parameters()))
print('PASS exact historical default updates, bc false and true')
a=FeedbackSAC.load(root/'FINAL.pt');b=FeedbackSAC.load(root/'FINAL.pt');a.train(d,l,1,success_bc=0,cql=0,successor_only=True);b.train(d,l,1,success_bc=.2,cql=0,successor_only=True)
assert all(torch.equal(x,y) for x,y in zip(a.q.parameters(),b.q.parameters()))
assert any(not torch.equal(x,y) for x,y in zip(a.actor.parameters(),b.actor.parameters()))
print('PASS single update: only actor changed by success BC; Q unchanged')
