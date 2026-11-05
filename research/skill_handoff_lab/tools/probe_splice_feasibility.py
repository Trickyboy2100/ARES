"""No new physics: inspect whether rejection is already present in the old queue.
This does not approve unsafe residuals or infer full-task success.
"""
import argparse,json,gzip,numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();rows=[]
for row in json.loads((a.root/'progress_splice/RESULTS.json').read_text()):
 d=json.load(gzip.open(row['path'],'rt'))
 for x in row['decisions']:
  if 'splice'not in x:continue
  sp=x['splice'];step=x['step'];before=d['initial']if step==0 else d['steps'][step-1]
  # Actual commanded next step is original continuation after rejection.
  nxt=d['steps'][step]if step<len(d['steps'])else None
  if nxt is None:continue
  q=np.asarray(before['q'])[:7];v=np.asarray(before['qvel'])[:7];command=np.asarray(nxt['action'])[:7]
  acc=( (command-q)/.05-v)/.05
  rows.append(dict(seed=row['seed'],trigger=row['trigger'],step=step,rejection=sp['rejection'],old_next_command_implied_acceleration=acc.tolist(),old_next_exceeds_2=bool(np.any(np.abs(acc)>2)),interpretation='position-setpoint finite difference includes PD tracking lag; not measured motor acceleration'))
(a.root/'SPLICE_FEASIBILITY.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
