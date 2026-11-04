import argparse,json,tempfile
from pathlib import Path
import numpy as np,torch
from program_budget_feedback_sac import FeedbackSAC,normalization,completion_labels
p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);a=p.parse_args();data=json.loads((a.pilot/'TRANSITIONS.json').read_text());import gzip
labels=[]
for r in json.loads((a.pilot/'feedback/RESULTS.json').read_text()):
 d=json.load(gzip.open(r['path'],'rt'));ts=[t for t in data if t['source']==r['path']];labels+=completion_labels(d,ts)
 for t in ts:
  assert t['duration']>0 and t['s'][41]>t['next'][41]
 assert abs(sum(t['r']for t in ts)-float(r['success'])-.2*(min(np.linalg.norm(ts[0]['s'][45:48])/.15,3)+.2*ts[0]['s'][48]))<1e-5
mu,sd=normalization(data);torch.set_num_threads(2);out={}
for mode in ['ordinary','suffix','budget','budget_blind']:
 m=FeedbackSAC(mode,31501,mu,sd);m.train(data,labels,4)
 x=m.x([t['s']for t in data]);p=m.probs(x);assert bool((p[x[:,44]<.5,3:]==0).all())
 vals=[m.probability(x,torch.ones(len(x))*b).detach().numpy()for b in [4,12,32,100]];assert all(np.all(v<=w+1e-7)for v,w in zip(vals,vals[1:]))
 with tempfile.TemporaryDirectory()as tmp:
  m.save(Path(tmp)/'m.pt');n=FeedbackSAC.load(Path(tmp)/'m.pt');m.train(data,labels,2);n.train(data,labels,2)
  assert all(torch.equal(v,n.q.state_dict()[k])for k,v in m.q.state_dict().items())
 out[mode]='PASS'
print(json.dumps(dict(fixtures=out,physical_transitions=len(data),native_duration_and_potential='PASS',legal_actions='PASS',budget_monotonicity='PASS',optimizer_resume='EXACT'),indent=2))
