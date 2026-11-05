"""Full-native actual learned-rescue falsification. Seen dev diagnostic only."""
import argparse,json,gzip,hashlib
from pathlib import Path
from program_atomic_feedback import AtomicRuntime
from peg_atomic_macro import AtomicMacro
from program_budget_feedback_sac import FeedbackSAC
from train_option_online import PARENT
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);m=FeedbackSAC.load(a.checkpoint);ref=FeedbackSAC.load(PARENT);rows=[];ds={}
def reference(s):
 b=ref.choose(s[:64]);return b+6 if b in [2,3,4,5]else b
rt=AtomicRuntime(a.out/'physics')
try:
 for mode in ['learned','suppress63','nominal','only63']:
  def actor(s):
   if s[44]<.5:return 1
   step=len(rt.env.records)
   if mode=='nominal':return 0
   if mode=='only63':return 5 if step==63 else 0
   if mode=='suppress63'and step==63:return 0
   return m.choose(s)
  policy=AtomicMacro(actor,reference);r,t,d=rt.run(761202,policy,mode);rows.append(dict(mode=mode,**r));ds[mode]=d;print(mode,r['success'],flush=True)
finally:rt.close()
for mode in ds:
 assert ds[mode]['initial']==ds['nominal']['initial']
 assert all(x[k]==y[k]for x,y in zip(ds[mode]['steps'][:63],ds['nominal']['steps'][:63])for k in ['q','qvel','tcp','peg','action'])
(a.out/'RESULTS.json').write_text(json.dumps(rows));(a.out/'MANIFEST.json').write_text(json.dumps(dict(scene=761202,prefix=63,exact_recorded_prefix=True,checkpoint=str(a.checkpoint),sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),scope='posthoc development intervention, not fresh confirmation'),indent=2))
