import argparse,json
from pathlib import Path
from peg_progress_splice import ProgressRuntime,source
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[759205,761100,761101,761102,761103]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);(a.out/'EXPANDED_EXECUTION.py').write_text(source)
rows=[]
for trigger in ['late','early']:
 rt=ProgressRuntime(a.out/trigger)
 try:
  for seed in a.seeds:
   used=[False]
   def actor(s):
    if s[44]<.5:return 1
    if not used[0] and (s[41]<=.060001 if trigger=='late' else s[49]<1):used[0]=True;return 2
    return 0
   r,t,d=rt.run(seed,actor,trigger);rows.append(dict(trigger=trigger,**r));(a.out/'RESULTS.json').write_text(json.dumps(rows));print(trigger,seed,r['success'],[x.get('splice') for x in r['decisions'] if 'splice'in x],flush=True)
 finally:rt.close()
