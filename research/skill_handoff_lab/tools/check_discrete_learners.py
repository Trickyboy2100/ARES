"""Algorithm fixtures with analytically known finite-option task, not physics scores."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,normalizer

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);rows=[]
 # Two contexts, three real first options, two second options. Known deterministic
 # transition and terminal table includes genuine first/second dependence.
 for c in [0,1]:
  for first in range(3):
   for second in range(2):
    x=np.zeros(54,np.float32);x[0]=c;x[41]=1.;b=x.copy();b[1]=first;b[41]=.3;b[44]=1.;z=b.copy();z[41]=0
    reward=float(first==c and second==c)
    rows +=[dict(s=x.tolist(),next=b.tolist(),a=first,r=0.,done=False,duration=70,stage=0,outcome=reward),dict(s=b.tolist(),next=z.tolist(),a=second,r=reward,done=True,duration=30,stage=1,outcome=reward)]
 scores=[]
 for kind in ['dsac','fqi','iql','cql','supervised','stap']:
  m=Learner(kind,31301,normalizer(rows));m.train(rows,steps=2000);pred=[]
  for c in [0,1]:
   x=np.zeros(54,np.float32);x[0]=c;x[41]=1.;first=m.choose(x);b=x.copy();b[1]=first;b[41]=.3;b[44]=1.;second=m.choose(b);pred.append([first,second]);assert first==c and second==c,(kind,c,pred)
  with tempfile.TemporaryDirectory()as td:
   path=Path(td)/'m.pt';m.save(path);loaded=Learner.load(path);assert np.array_equal(m.values(b)[0],loaded.values(b)[0]);assert loaded.updates==m.updates
  scores.append(dict(method=kind,analytical_choices=pred,save_load_exact=True,updates=m.updates))
 a.out.write_text(json.dumps(dict(scope='analytic toy fixture ONLY; no physical or research-gain claim',checks=scores),indent=2));print(scores)
if __name__=='__main__':main()
