"""Posthoc full-native predecessor/successor decomposition, never new test."""
import argparse,json
from pathlib import Path
from program_peg_feedback import FeedbackRuntime
from program_budget_feedback_sac import FeedbackSAC
from program_continuation_option import ContinuationOption
from train_option_online import PARENT
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--frame',choices=['measured','remaining_target'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
b=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-004');models={'measured':b/'online_integrity_20261105/online_success_bc/FINAL.pt','remaining_target':b/'decoupled_interface_20261105/relative_bc64/FINAL.pt'}
rear=FeedbackSAC.load(models[a.frame]);other=FeedbackSAC.load(models['remaining_target' if a.frame=='measured' else 'measured']);ref=FeedbackSAC.load(PARENT);rt=FeedbackRuntime(a.out/'physics',correction_frame=a.frame);rows=[]
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(scope='POSTHOC previously seen32, no training',seeds=list(range(760200,760232)),rear=str(models[a.frame]),fronts=['fixed1','cross_other_model'],native100=True),indent=2))
try:
 for mode in ['fixed1','cross']:
  for seed in range(760200,760232):
   def actor(s):return (1 if mode=='fixed1' else other.choose(s))if s[44]<.5 else rear.choose(s)
   policy=ContinuationOption(actor,ref.choose,'insertion_queue');r,t,d=rt.run(seed,policy,mode);rows.append(dict(front=mode,**r));(a.out/'RESULTS.json').write_text(json.dumps(rows));print(mode,seed,r['success'],flush=True)
finally:rt.close()
