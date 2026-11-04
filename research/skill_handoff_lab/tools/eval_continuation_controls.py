import argparse,json
from pathlib import Path
import torch
from program_budget_feedback_sac import FeedbackSAC
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=16);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004')
m=FeedbackSAC.load(root/'success_retention_20261104/controlled/FINAL.pt');rt=FeedbackRuntime(a.out/'physics');rows=[]
for seed in range(758200,758200+a.count):
 for name in ['zero','frozen_rule','primitive_sac','option_parent','option_rule']:
  def f(s):
   if s[44]<.5:return 1
   if name=='zero':return 0
   if name in ['frozen_rule','option_rule']:return 4 if s[49]<=6/7+1e-5 and s[41]<=.28 else 0
   return m.choose(s)
  policy=ContinuationOption(f,m.choose,'insertion_queue') if name.startswith('option_') else f
  r,t,d=rt.run(seed,policy,name);rows.append(dict(method=name,**r));(a.out/'RESULTS.json').write_text(json.dumps(rows))
  print(seed,name,r['success'],flush=True)
rt.close()
