"""Native mechanism comparison; all cases are already-seen diagnostics."""
import argparse,json
from pathlib import Path
import torch
from program_budget_feedback_sac import FeedbackSAC
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--event-only',action='store_true');a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004')
reference=FeedbackSAC.load(root/'success_retention_20261104/controlled/FINAL.pt')
actor=FeedbackSAC.load(root/'intervention_credit_20261104/current_measured_preference/FINAL.pt')
rt=FeedbackRuntime(a.out/'physics');rows=[]
for seed in [757300,757302,753209]:
 for mode in (['insertion_queue'] if a.event_only else ['chunk','phase','episode']):
  for continuation in (['reference','zero'] if mode!='chunk' else ['reference']):
   policy=ContinuationOption(lambda s:1 if s[44]<.5 else actor.choose(s),lambda s:reference.choose(s) if continuation=='reference' else 0,mode)
   r,t,d=rt.run(seed,policy,mode+'_'+continuation)
   rows.append(dict(seed=seed,mode=mode,continuation=continuation,success=r['success'],path=r['path'],events=policy.events,transitions=policy.transitions(t)))
   (a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2))
   print(seed,mode,continuation,r['success'],flush=True)
rt.close()
