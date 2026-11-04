"""CPU-only readout of frozen Q and Actor at an actual logged decision."""
import json,gzip,torch,hashlib
from pathlib import Path
from program_budget_feedback_sac import FeedbackSAC
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004/decoupled_interface_20261105');row=next(v for v in json.loads((r/'diag_relative_g1/RESULTS.json').read_text())if v['seed']==759205);d=json.load(gzip.open(row['path'],'rt'));e=next(v for v in d['summary']['decisions']if v['step']==94);m=FeedbackSAC.load(r/'relative_bc64/FINAL.pt');x=m.x([e['obs']])
with torch.no_grad():q=torch.stack([v(x)for v in m.q]).mean(0)[0].tolist();prob=m.probs(x)[0].tolist()
out=dict(seed=759205,grasp=1,step=94,model_sha=hashlib.sha256((r/'relative_bc64/FINAL.pt').read_bytes()).hexdigest(),obs=e['obs'],mean_q=q,actor_probs=prob,chosen=m.choose(e['obs']),scope='diagnostic prediction at true physical state; Q not calibrated probability; no update or physics')
(r/'LATE_REPLAN_VALUES.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items()if k!='obs'})
