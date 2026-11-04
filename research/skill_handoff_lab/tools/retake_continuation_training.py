"""Selected training diagnoses only; never independent generalization evidence."""
import argparse,json
from pathlib import Path
import torch
from program_budget_feedback_sac import FeedbackSAC
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004');ref=FeedbackSAC.load(root/'success_retention_20261104/controlled/FINAL.pt');rt=FeedbackRuntime(a.out/'physics');rows=[]
for checkpoint in sorted(a.run.glob('*_FINAL.pt')):
 m=FeedbackSAC.load(checkpoint)
 for seed in [758000,758002,758004,758005,757300]:
  p=ContinuationOption(lambda s:1 if s[44]<.5 else m.choose(s),ref.choose,'insertion_queue')
  r,t,d=rt.run(seed,p,checkpoint.stem);rows.append(dict(model=checkpoint.name,events=p.events,**r));(a.out/'RESULTS.json').write_text(json.dumps(rows))
  print(checkpoint.name,seed,r['success'],flush=True)
rt.close()
