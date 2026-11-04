"""Native reexecution pairs, no state restoration or development-label mining."""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
from program_peg_feedback import FeedbackRuntime
from program_budget_feedback_sac import completion_labels
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=757000);p.add_argument('--count',type=int,default=8)
p.add_argument('--phases',type=int,nargs='+',default=[4,7])
p.add_argument('--continuation',choices=['zero','rule'],default='zero')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rt=FeedbackRuntime(a.out/'physics');pairs=[];trans=[];labels=[]
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(start=a.start,count=a.count,actions=[0,4,5,7],selection='first decision in specified phase, regardless of outcome',phases=a.phases,prefix='zero feedback, actual grasp1',continuation=a.continuation,native100=True,restores=0),indent=2))
for seed in range(a.start,a.start+a.count):
 for phase in a.phases:
  runs=[]
  for action in [0,4,5,7]:
   hit=[]
   def policy(s):
    if s[44]<.5:return 1
    if not hit and abs(s[49]*7-phase)<1e-4:
     hit.append(len(rt.actions));return action
    return 4 if hit and a.continuation=='rule' and s[49]<=6/7+1e-5 and s[41]<=.28 else 0
   row,ts,d=rt.run(seed,policy,f'phase{phase}_a{action}')
   trans.extend(ts);labels.extend(completion_labels(d,ts));runs.append((row,d,hit))
   print('PAIR',seed,phase,action,row['success'],hit,flush=True)
  base,bd,bhit=runs[0]
  for action,(row,d,hit) in zip([4,5,7],runs[1:]):
   if not bhit or not hit:continue
   assert hit==bhit
   idx=hit[0];b=bd['summary']['decisions'][idx];v=d['summary']['decisions'][idx];step=b['step']
   assert b['queue_before']==v['queue_before'] and b['obs']==v['obs'] and step==v['step']
   assert bd['initial']==d['initial']
   for x,y in zip(bd['steps'][:step],d['steps'][:step]):
    for key in ['action','q','qvel','tcp','peg']:assert x[key]==y[key],key
   pairs.append(dict(scene=seed,phase=phase,s=b['obs'],a=action,delta=int(row['success'])-int(base['success']),base_success=base['success'],action_success=row['success'],base_path=base['path'],action_path=row['path'],prefix_steps=step,queue_sha=hashlib.sha256(json.dumps(b['queue_before'],sort_keys=True).encode()).hexdigest(),prefix_exact=True))
  for name,obj in [('PAIRS',pairs),('TRANSITIONS',trans),('LABELS',labels)]: (a.out/(name+'.json')).write_text(json.dumps(obj))
rt.close();print('DONE',len(pairs),sum(x['delta']>0 for x in pairs),sum(x['delta']<0 for x in pairs),flush=True)
