import json,gzip,numpy as np
from pathlib import Path
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[5]/'runs/research_program/EXP-003/post_contact_20261102');args=parser.parse_args();root=args.root;rows=[]
for folder in ['near','design']:
 for r in json.load(open(root/folder/'RESULTS.json')):
  if r['action']!=[0,0]or r['contact_step']is None:continue
  d=json.load(gzip.open(r['path'],'rt'));states=[d['initial_measure'],*d['steps']];first=None
  for i,s in enumerate(states[1:],1):
   if s['phase']not in ['TRANSFER','HOOK','PULL']:continue
   force=max([z['force_norms_N']['held_tool']for z in s['physics_substep_contacts']]+[s['tool_cube_force_N']]);delta=float(np.linalg.norm(np.array(s['cube'][:2])-states[i-1]['cube'][:2]))
   if force>.5 and delta>.0001:first=dict(step=i,phase=s['phase'],force_N=force,interval_cube_motion_m=delta);break
  rows.append(dict(seed=r['seed'],grasp=r['grasp'],source=r['path'],sha256=r['sha256'],pull_boundary=r['contact_step'],earliest=first,delay=None if first is None else r['contact_step']-first['step']))
vals=[r['delay']for r in rows if r['delay']is not None];out=dict(definition='First actual TRANSFER/HOOK/PULL force>0.5N + same20Hz interval cubeXY displacement>0.1mm, compared with prior PULL-phase boundary',contexts=len(rows),earlier_than_pull=sum(v>0 for v in vals),median_delay_steps=float(np.median(vals)),max_delay_steps=max(vals),rows=rows);(root/'analysis/EVENT_TIMING.json').write_text(json.dumps(out,indent=2));print({k:v for k,v in out.items()if k!='rows'})
