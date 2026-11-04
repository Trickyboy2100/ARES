"""Native same-prefix target-reparameterization and real predecessor contrasts.
No state restore, learning, hindsight scenario selection, or task-limit change.
"""
import argparse,json
from pathlib import Path
import numpy as np
from program_peg_feedback import FeedbackRuntime
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=759500);p.add_argument('--count',type=int,default=4);p.add_argument('--refresh-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
(a.out/'MANIFEST.json').write_text(json.dumps(dict(seeds=list(range(a.start,a.start+a.count)),grasp=[0,1,2],actions=[0,1] if a.refresh_only else [0,4,5],frames=['measured','remaining_target'],suffix='continue current queue after first rear intervention',purpose='mechanism only; never training/development scores',native100=True),indent=2))
rows=[];comparisons=[];rt=FeedbackRuntime(a.out/'physics');base_id=rt.controller_id
try:
 for seed in range(a.start,a.start+a.count):
  for grasp in range(3):
   base=None
   for frame,action in ([('measured',0),('measured',1)] if a.refresh_only else [('measured',0),('measured',4),('measured',5),('remaining_target',4),('remaining_target',5)]):
    rt.correction_frame=frame;rt.controller_id=base_id+('__remaining_target_v1' if frame=='remaining_target' else '');used=[False]
    def policy(s):
     if s[44]<.5:return grasp
     if not used[0]:used[0]=True;return action
     return 0
    row,ts,d=rt.run(seed,policy,f'g{grasp}_{frame}_a{action}')
    rear=[x for x in d['summary']['decisions'] if x['stage']==1]
    if action==0:base=d
    if rear:
     first=rear[0];step=first['step'];b=[x for x in base['summary']['decisions'] if x['stage']==1][0]
     exact=(first['queue_before']==b['queue_before'] and first['obs']==b['obs'] and d['initial']==base['initial'])
     exact=exact and all(x[k]==y[k] for x,y in zip(d['steps'][:step],base['steps'][:step]) for k in ['q','qvel','tcp','peg','action'])
     assert exact
     n=min(4,100-step);change=max(np.linalg.norm(np.array(d['steps'][i]['action'][:7])-np.array(base['steps'][i]['action'][:7])) for i in range(step,step+n))
     comparisons.append(dict(seed=seed,grasp=grasp,frame=frame,action=action,prefix_steps=step,exact_prefix=exact,max_first4_command_delta_rad=float(change),target_delta_m=first.get('requested_target_delta_m'),success=row['success'],nominal_success=base['summary']['success'],first_rear_obs=first['obs']))
    rows.append(dict(grasp=grasp,frame=frame,intervention=action,**row));print(seed,grasp,frame,action,row['success'],flush=True)
    (a.out/'RESULTS.json').write_text(json.dumps(rows));(a.out/'COMPARISONS.json').write_text(json.dumps(comparisons))
finally:rt.close()
