import argparse,json,gzip
from pathlib import Path
import numpy as np
from program_peg_feedback import FeedbackRuntime,rule
from program_peg_option_runtime import Runtime
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
base=Runtime(a.out/'legacy');refs={}
for seed in range(753000,753004):refs[seed]=base.run(seed,lambda o:1 if o[44]<.5 else 0,'nominal')[2]
base.close();rt=FeedbackRuntime(a.out/'feedback');checks=[];ts=[]
for seed in range(753000,753008):
 for mode in ['zero','feedback','refresh']:
  cb=(lambda o:1 if o[44]<.5 else 0) if mode=='zero' else rule if mode=='feedback' else (lambda o:1 if o[44]<.5 else (1 if o[-1]==0 else 0))
  row,t,d=rt.run(seed,cb,mode);ts+=t
  if mode=='zero' and seed in refs:
   errors={k:max(np.max(np.abs(np.array(x[k])-np.array(y[k])))for x,y in zip(d['steps'],refs[seed]['steps']))for k in ['q','peg','tcp','action']};checks.append(dict(scene=seed,errors=errors));assert max(errors.values())<1e-6 and d['summary']['success']==refs[seed]['summary']['success'],checks[-1]
rt.close();(a.out/'CHECKS.json').write_text(json.dumps(checks,indent=2));(a.out/'TRANSITIONS.json').write_text(json.dumps(ts));print('PILOT_DONE',flush=True)
