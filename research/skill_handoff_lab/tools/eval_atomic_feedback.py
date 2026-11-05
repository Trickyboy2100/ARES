"""Frozen full-native evaluation for the common83D/12-action interface."""
import argparse,json,hashlib
from pathlib import Path
from program_atomic_feedback import AtomicRuntime,SemanticOption
from program_budget_feedback_sac import FeedbackSAC
from train_option_online import PARENT
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=761200);p.add_argument('--count',type=int,default=16);p.add_argument('--checkpoint',type=Path);p.add_argument('--termination',choices=['phase','chunk','insertion_queue'],default='phase');p.add_argument('--control',choices=['nominal','rule','budget_guard'],default='nominal');p.add_argument('--grasp',type=int);p.add_argument('--front-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(a.out/'MANIFEST.json').write_text(json.dumps(dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},checkpoint_sha=sha(a.checkpoint)if a.checkpoint else None,seeds=list(range(a.start,a.start+a.count)),native100=True,source_sha=sha(Path(__file__))),indent=2))
m=FeedbackSAC.load(a.checkpoint)if a.checkpoint else None;ref=FeedbackSAC.load(PARENT)
def reference(s):
 b=ref.choose(s[:64]);return b+6 if b in (2,3,4,5)else b
rt=AtomicRuntime(a.out/'physics');rows=[]
try:
 for seed in range(a.start,a.start+a.count):
  def actor(s):
   if s[44]<.5:return a.grasp if a.grasp is not None else m.choose(s)if m else 1
   if a.front_only:return 0
   if m:return m.choose(s)
   if a.control=='nominal':return 0
   # Frozen prior rule mapped to same atomic physical primitive.
   if a.control=='budget_guard' and s[41]<.08:return 0
   return 10 if s[49]<=6/7+1e-5 and s[41]<=.28 else 0
  policy=SemanticOption(actor,reference if m else lambda s:0,a.termination)
  r,t,d=rt.run(seed,policy,'eval');rows.append(r);(a.out/'RESULTS.json').write_text(json.dumps(rows));(a.out/f'EVENTS_{seed}.json').write_text(json.dumps(policy.events));print(seed,r['success'],flush=True)
finally:rt.close()
