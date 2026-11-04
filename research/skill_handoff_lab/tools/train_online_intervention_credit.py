"""Current-policy native-prefix intervention credit; both branches fully charged.
Revision of fixed zero-continuation labels: freeze actor during each real pair,
then replace actor gap only at freshly measured states, never future deployment.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np,torch
from program_budget_feedback_sac import FeedbackSAC,completion_labels
from program_peg_feedback import FeedbackRuntime
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--mode',choices=['sac','awac','measured_gap','measured_preference'],required=True);p.add_argument('--episodes',type=int,default=16);p.add_argument('--start',type=int,default=757300);p.add_argument('--dev-count',type=int,default=24)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
m=FeedbackSAC.load(a.parent/'FINAL.pt');data=json.loads((a.parent/'TRANSITIONS.json').read_text());labels=json.loads((a.parent/'LABELS.json').read_text());rt=FeedbackRuntime(a.out/'physics');allpairs=[]
(a.out/'PROTOCOL.json').write_text(json.dumps({**{k:str(v) for k,v in vars(a).items()},'revision':'current actor frozen for both suffixes; no historical-gap reuse','selection':'first nonzero actor decision; otherwise first rear with round-robin intervention1..7','cost':'3 full native episodes per round including acquisition rollout','updates_per_round':80,'final_only':True},indent=2))
source=a.out/'source';source.mkdir()
for f in [Path(__file__).name,'program_budget_feedback_sac.py','program_peg_feedback.py','program_peg_option_runtime.py']:shutil.copy2(Path(__file__).parent/f,source/f)
for i in range(a.episodes):
 seed=a.start+i
 def deploy(s):return 1 if s[44]<.5 else m.choose(s)
 r,t,d=rt.run(seed,deploy,f'acquire_{i}');data.extend(t);labels.extend(completion_labels(d,t));ds=d['summary']['decisions'];targets=[k for k,v in enumerate(ds) if v['stage']==1 and v['id']!=0]
 if not targets:targets=[k for k,v in enumerate(ds) if v['stage']==1]
 fresh=[]
 if targets:
  k=targets[0];act=ds[k]['id'] or (1+i%7);runs=[]
  for action in [0,act]:
   def policy(s):
    n=len(rt.actions)
    if n<k:return ds[n]['id']
    if n==k:return action
    return deploy(s)
   rr,tt,dd=rt.run(seed,policy,f'pair_{i}_a{action}');data.extend(tt);labels.extend(completion_labels(dd,tt));runs.append((rr,dd))
  (b,bd),(v,vd)=runs;bx=bd['summary']['decisions'][k];vx=vd['summary']['decisions'][k];step=bx['step']
  assert bx['queue_before']==vx['queue_before'] and bx['obs']==vx['obs'] and bd['initial']==vd['initial']
  for x,y in zip(bd['steps'][:step],vd['steps'][:step]):
   for key in ['action','q','qvel','tcp','peg']:assert x[key]==y[key]
  fresh=[dict(s=bx['obs'],a=act,delta=int(v['success'])-int(b['success']),scene=seed,base_path=b['path'],action_path=v['path'],prefix_steps=step,prefix_exact=True,policy_update=m.updates)]
  allpairs.extend(fresh)
 log=m.train(data,labels,80,successor_only=True,intervention_pairs=fresh or None,actor_rule=a.mode)
 print('ONLINE_PAIR',i,r['success'],[(x['a'],x['delta']) for x in fresh],log['actor_loss'],flush=True)
 if (i+1)%8==0:m.save(a.out/f'CHECKPOINT_{i+1}.pt')
 for key,obj in [('TRANSITIONS',data),('LABELS',labels),('PAIRS',allpairs)]: (a.out/(key+'.json')).write_text(json.dumps(obj))
m.save(a.out/'FINAL.pt');rows=[]
for seed in range(757200,757200+a.dev_count):
 r,t,d=rt.run(seed,deploy,'final_eval');rows.append(r);print('FINAL',seed,r['success'],flush=True)
(a.out/'final_eval.json').write_text(json.dumps(rows));rt.close()
