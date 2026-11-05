"""Native full-prefix H1/H2 probes. No state restoration or training reuse."""
import argparse,json,gzip,hashlib
from pathlib import Path
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[760213]);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
branches=['continue','refresh_once','local_once','atomic_once','sequential','atomic_rule','local_rule','atomic_option','atomic_phase']
(a.out/'MANIFEST.json').write_text(json.dumps(dict(branches=branches,seeds=a.seeds,scope='seen witness mechanism only; full native reexecution',trigger='first refinement rear boundary remaining<=28',suffix='continue except explicitly labeled repeated rule/Option',source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
rows=[];evidence={}
for branch in branches:
 rt=FeedbackRuntime(a.out/branch,correction_frame='remaining_target' if branch in ['local_once','local_rule','sequential'] else 'measured')
 try:
  for seed in a.seeds:
   triggered=[False];seq=[False]
   def actor(s):
    if s[44]<.5:return 1
    eligible=s[49]<=6/7+1e-5 and s[41]<=.28
    if branch=='continue':return 0
    if branch in ['atomic_rule','local_rule','atomic_option','atomic_phase']:return 4 if eligible else 0
    if seq[0]:seq[0]=False;return 4
    if eligible and not triggered[0]:
     triggered[0]=True
     if branch=='sequential':seq[0]=True
     return 1 if branch in ['refresh_once','sequential'] else 4
    return 0
   policy=ContinuationOption(actor,lambda s:0,'insertion_queue' if branch=='atomic_option' else 'phase') if branch in ['atomic_option','atomic_phase'] else actor
   r,t,d=rt.run(seed,policy,branch);rows.append(dict(branch=branch,**r));evidence[(seed,branch)]=d
   if hasattr(policy,'events'):(a.out/f'{branch}_{seed}_EVENTS.json').write_text(json.dumps(policy.events))
   (a.out/'RESULTS.json').write_text(json.dumps(rows));print(branch,seed,r['success'],flush=True)
 finally:rt.close()
checks=[]
for seed in a.seeds:
 base=evidence[(seed,'continue')]
 for branch in branches[1:]:
  d=evidence[(seed,branch)];ev=[x for x in d['summary']['decisions']if x['stage'] and x['id']];n=ev[0]['step'] if ev else 100
  exact=base['initial']==d['initial'] and all(x[k]==y[k] for x,y in zip(base['steps'][:n],d['steps'][:n]) for k in ['q','qvel','tcp','peg','action']);assert exact
  checks.append(dict(seed=seed,branch=branch,prefix=n,recorded_prefix_exact=exact,success=d['summary']['success'],events=[dict(step=x['step'],action=x['id'],phase=x.get('queue_before',{}).get('phase'),remaining=100-x['step'],replanned=x.get('replanned'),queue=x.get('queue_length'),target=x.get('target'))for x in d['summary']['decisions']]))
(a.out/'PREFIX_COMPARISONS.json').write_text(json.dumps(checks,indent=2))
