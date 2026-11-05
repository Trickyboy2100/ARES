"""Post-confirmation causal explanation; never model selection or extra test scores."""
import argparse,json,gzip,hashlib
from pathlib import Path
from program_atomic_feedback import AtomicRuntime
from peg_atomic_macro import AtomicMacro
from program_budget_feedback_sac import FeedbackSAC
from train_option_online import PARENT
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'confirmation_effect';out.mkdir(exist_ok=False);ref=FeedbackSAC.load(PARENT);rt=AtomicRuntime(out/'physics');rows=[];checks=[]
try:
 for name,seed,step,action,cp in [('macro_bc64',762214,58,5,32),('macro_seed31702',762213,74,2,64)]:
  m=FeedbackSAC.load(a.root/name/f'CHECKPOINT_{cp}.pt');original=next(x for x in json.loads((a.root/('confirm_'+name)/'RESULTS.json').read_text())if x['seed']==seed);old=json.load(gzip.open(original['path'],'rt'))
  def reference(s):
   b=ref.choose(s[:64]);return b+6 if b in [2,3,4,5]else b
  for mode in ['learned','suppress_first','only_first']:
   def actor(s):
    if s[44]<.5:return m.choose(s)
    n=len(rt.env.records)
    if mode=='only_first':return action if n==step else 0
    if mode=='suppress_first'and n==step:return 0
    return m.choose(s)
   policy=AtomicMacro(actor,reference);r,t,d=rt.run(seed,policy,mode);rows.append(dict(mode=mode,**r));assert d['initial']==old['initial'];same=all(x[k]==y[k]for x,y in zip(d['steps'][:step],old['steps'][:step])for k in ['q','qvel','tcp','peg','action']);assert same
   checks.append(dict(seed=seed,mode=mode,prefix=step,exact_recorded_prefix=same,original_sha=original['sha256']));(out/'RESULTS.json').write_text(json.dumps(rows));(out/'PREFIX.json').write_text(json.dumps(checks,indent=2));print(seed,mode,r['success'],flush=True)
finally:rt.close()
