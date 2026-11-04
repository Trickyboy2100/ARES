"""Controlled physical probe: isolate zero-offset replanning from continuing.
Not a learned policy, not training replay, not a claimed new RL method.
Only two literal target-offset constants in inherited execute_feedback vary.
"""
import argparse,json,inspect,textwrap,hashlib,gzip
from pathlib import Path
import program_peg_feedback as base
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[759205]);p.add_argument('--grasp',type=int,default=1);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
original=textwrap.dedent(inspect.getsource(base.FeedbackRuntime.execute_feedback));assert original.count('=.003*')==2
(a.out/'BASE_EXECUTION.py').write_text(original)
(a.out/'MANIFEST.json').write_text(json.dumps(dict(seeds=a.seeds,grasp=a.grasp,trigger='first rear decision with remaining<=6; only once',branches=['continue','zero_offset_replan','3mm_replan'],source_sha=hashlib.sha256(original.encode()).hexdigest(),native100=True,scope='mechanism probe; no training; actual physical prefix reexecuted'),indent=2))
rows=[];comparisons=[]
for name,amplitude in [('continue',.003),('zero_offset_replan',0.),('3mm_replan',.003)]:
 source=original.replace('=.003*',f'={amplitude}*');ns={};exec(compile(source,str(a.out/f'{name}_EXECUTION.py'),'exec'),base.__dict__,ns);(a.out/f'{name}_EXECUTION.py').write_text(source)
 class Probe(base.FeedbackRuntime):execute_feedback=ns['execute_feedback']
 rt=Probe(a.out/name,correction_frame='remaining_target');rt.controller_id+='__clock_probe_'+name
 try:
  for seed in a.seeds:
   used=[False]
   def actor(s):
    if s[44]<.5:return a.grasp
    if name!='continue' and not used[0] and s[41]*100<=6.0001:used[0]=True;return 2
    return 0
   r,t,d=rt.run(seed,actor,name);rows.append(dict(method=name,**r));print(name,seed,r['success'],flush=True)
   (a.out/'RESULTS.json').write_text(json.dumps(rows))
 finally:rt.close()
for seed in a.seeds:
 ds={r['method']:json.load(gzip.open(r['path'],'rt'))for r in rows if r['seed']==seed}
 for name in ['zero_offset_replan','3mm_replan']:
  d=ds[name];events=[x for x in d['summary']['decisions']if x['stage']==1 and x['id']==2]
  if not events:comparisons.append(dict(seed=seed,method=name,triggered=False));continue
  event=events[0];n=event['step'];exact=d['initial']==ds['continue']['initial'] and all(x[k]==y[k]for x,y in zip(d['steps'][:n],ds['continue']['steps'][:n])for k in ['q','qvel','tcp','peg','action']);assert exact
  comparisons.append(dict(seed=seed,method=name,triggered=True,prefix=n,exact_prefix=exact,remaining=100-n,old_queue_remaining=len(event['queue_before']['queue'])-event['queue_before']['cursor'],new_queue_length=event['queue_length'],target_delta=event.get('requested_target_delta_m'),success=d['summary']['success'],continue_success=ds['continue']['summary']['success']))
(a.out/'COMPARISONS.json').write_text(json.dumps(comparisons,indent=2))
