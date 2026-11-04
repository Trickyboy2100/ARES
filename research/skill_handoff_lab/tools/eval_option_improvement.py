"""Read frozen policies on exact newly measured finite physical branches.
Logged branch evaluation is explicitly NOT an additional simulation rollout.
"""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner
from program_option_improvement import Improved

def panel(folder):
 rows=json.loads((folder/'ROWS.json').read_text());ts=json.loads((folder/'TRANSITIONS.json').read_text());out={}
 for r in rows:
  assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
  aa,bb=map(int,r['tag'][-2:]);out[r['seed'],aa,bb]=(r,[t for t in ts if t['source']==r['path']])
 return out

def evaluate(m,p,rule='deployed'):
 scenes=sorted({s for s,a,b in p});nb=1+max(b for s,a,b in p);results=[]
 def choose(obs):
  if rule=='deployed':return m.choose(obs)
  if rule=='actor':
   with torch.no_grad():return int(m.probs(m.x(np.atleast_2d(obs)))[0,:3 if obs[44]<.5 else m.successor_actions].argmax())
  _,q=m.values(obs);q=q[:,0,:3 if obs[44]<.5 else m.successor_actions];return int((q.mean(0)if rule=='mean'else q.min(0)).argmax())
 for s in scenes:
  first=choose(p[s,0,0][1][0]['s']);ts=p[s,first,0][1];second=choose(ts[1]['s'])if len(ts)==2 else 0;r,_=p[s,first,second]
  union=any(p[s,a,b][0]['success']for a in range(3)for b in range(nb));first_ok=any(p[s,first,b][0]['success']for b in range(nb))
  results.append(dict(scene=s,actions=[first,second]if len(ts)==2 else[first],success=r['success'],source=r['path'],failure='success'if r['success']else 'no_measured_solution'if not union else'predecessor_selection'if not first_ok else'successor_selection'))
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False)
 files=sorted(a.root.glob('train_*/*.pt'));assert files
 manifest=[dict(path=str(f),sha=hashlib.sha256(f.read_bytes()).hexdigest())for f in files];(a.out/'FROZEN_MODELS.json').write_text(json.dumps(manifest,indent=2))
 pp=panel(a.panel);nb=1+max(b for s,aa,b in pp);scenes=sorted({s for s,aa,b in pp});scores=[];pred=[];errors=[]
 for f in files:
  kind,k=f.stem.rsplit('_',1);cls=Learner if kind in ('fqi','iql','supervised','stap')else Improved;m=cls.load(f)
  for rule in ['deployed']+(['mean','min']if kind in ('sac_min','sac_mean','usac','vi_expectile','iql')else ['actor']if kind in ('vi_mean','handoff','lower_bound')else[]):
   rr=evaluate(m,pp,rule);scores.append(dict(method=kind,seed=m.seed,checkpoint=int(k),rule=rule,success=sum(r['success']for r in rr),n=len(rr),model=str(f),evaluation='FROZEN_POLICY_ON_REAL_BRANCHES_NOT_ADDITIONAL_PHYSICS'));pred.append(dict(method=kind,seed=m.seed,checkpoint=int(k),rule=rule,rows=rr))
  if int(k)in (192,288) and kind!='stap':
   for s in scenes:
    for first in range(3):
     ts=pp[s,first,0][1]
     for stage in range(len(ts)):
      if stage==0 and first!=0:continue
      obs=ts[stage]['s'];v,q=m.values(obs);n=3 if stage==0 else nb
      y=[float(any(pp[s,aa,b][0]['success']for b in range(nb)))for aa in range(3)]if stage==0 else[float(pp[s,first,b][0]['success'])for b in range(nb)]
      errors.append(dict(method=kind,seed=m.seed,scene=s,stage=stage,first=first,q=q[:,0,:n].tolist(),pred=v[0,:n].tolist(),measured_optimal=y))
 for aa in range(3):
  for bb in range(nb):scores.append(dict(method=f'fixed{aa}{bb}',seed=None,checkpoint=None,rule='fixed',success=sum(pp[s,aa,bb][0]['success']for s in scenes),n=len(scenes),evaluation='DIRECT_NATIVE_PHYSICS'))
 (a.out/'SCORES.json').write_text(json.dumps(scores,indent=2));(a.out/'PREDICTIONS.json').write_text(json.dumps(pred,indent=2));(a.out/'VALUE_ERRORS.json').write_text(json.dumps(errors,indent=2));(a.out/'OPPORTUNITY.json').write_text(json.dumps(dict(scenes=scenes,union=sum(any(pp[s,aa,bb][0]['success']for aa in range(3)for bb in range(nb))for s in scenes),best_fixed=max(x['success']for x in scores if x['rule']=='fixed'),all_success=sum(all(pp[s,aa,bb][0]['success']for aa in range(3)for bb in range(nb))for s in scenes),physical_episodes=len(pp)),indent=2))
 print(json.dumps([r for r in scores if r['checkpoint']==192 and r['rule']=='deployed'],indent=2))
if __name__=='__main__':main()
