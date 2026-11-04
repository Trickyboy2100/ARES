"""Frozen controls: never retune on the new development group."""
import argparse,json
from pathlib import Path
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=759200);p.add_argument('--count',type=int,default=16);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
rt=FeedbackRuntime(a.out/'physics');rows=[]
try:
 for seed in range(a.start,a.start+a.count):
  for name in ['nominal','frozen_rule']:
   def actor(s):
    if s[44]<.5:return 1
    return 4 if name=='frozen_rule' and s[49]<=6/7+1e-5 and s[41]<=.28 else 0
   # Original strongest frozen rule retains its original repeated authority;
   # its rule is not re-tuned or claimed to be a learned Option comparator.
   r,t,d=rt.run(seed,actor,name);rows.append(dict(method=name,**r))
   (a.out/'RESULTS.json').write_text(json.dumps(rows));print(name,seed,r['success'],flush=True)
finally:rt.close()
