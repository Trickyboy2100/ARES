"""Frozen policy/control native evaluation; explicit interface, no training."""
import argparse,json,hashlib
from pathlib import Path
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption
from program_budget_feedback_sac import FeedbackSAC
from train_option_online import PARENT
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=759200);p.add_argument('--count',type=int,default=16);p.add_argument('--frame',choices=['measured','remaining_target'],required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--split',default='development');p.add_argument('--seeds',type=str);p.add_argument('--grasp',type=int,choices=[0,1,2]);p.add_argument('--control',choices=['nominal','frozen_rule'],default='frozen_rule');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seeds=[int(x)for x in a.seeds.split(',')] if a.seeds else list(range(a.start,a.start+a.count))
(a.out/'MANIFEST.json').write_text(json.dumps(dict(frame=a.frame,split=a.split,evaluator_sha=sha(Path(__file__)),runtime_sha=sha(Path(__file__).with_name('program_peg_feedback.py')),seeds=seeds,grasp_override=a.grasp,scope='diagnostic shared predecessor' if a.grasp is not None else 'full learned predecessor',checkpoint=str(a.checkpoint),checkpoint_sha=sha(a.checkpoint) if a.checkpoint else None,control=a.control,reference_sha=sha(PARENT),native_horizon=100,selection='frozen before evaluation',sealed=False),indent=2))
m=FeedbackSAC.load(a.checkpoint) if a.checkpoint else None
ref=FeedbackSAC.load(PARENT) if m else None
rt=FeedbackRuntime(a.out/'physics',correction_frame=a.frame);rows=[]
try:
 for seed in seeds:
  def actor(s):
   if s[44]<.5:return 1
   return 4 if a.control=='frozen_rule' and s[49]<=6/7+1e-5 and s[41]<=.28 else 0
  def learned(s):
   return a.grasp if a.grasp is not None and s[44]<.5 else m.choose(s)
  policy=ContinuationOption(learned,ref.choose,'insertion_queue') if m else actor
  r,t,d=rt.run(seed,policy,'frozen_model' if m else a.control);rows.append(r)
  (a.out/'RESULTS.json').write_text(json.dumps(rows));print(seed,r['success'],flush=True)
finally:rt.close()
