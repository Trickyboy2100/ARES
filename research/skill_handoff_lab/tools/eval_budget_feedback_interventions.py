"""Full native-prefix controlled deployment ablations of one frozen learner."""
import argparse,json,shutil,hashlib
from pathlib import Path
from program_budget_feedback_sac import FeedbackSAC
from program_peg_feedback import FeedbackRuntime,rule
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--start',type=int,default=753200);p.add_argument('--count',type=int,default=16);p.add_argument('--scenes',type=int,nargs='+');p.add_argument('--fixed-grasp',type=int,choices=[0,1,2]);p.add_argument('--modes',nargs='+',default=['zero_suffix','one_update','rule_suffix']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);m=FeedbackSAC.load(a.model);rt=FeedbackRuntime(a.out/'physics');rows=[]
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(model=str(a.model),sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),scenes=a.scenes or list(range(a.start,a.start+a.count)),modes=a.modes,fixed_grasp=a.fixed_grasp,targeted=a.scenes is not None),indent=2))
for seed in (a.scenes or range(a.start,a.start+a.count)):
 for mode in a.modes:
  seen=[False]
  def policy(s):
   if s[44]<.5:return m.choose(s) if a.fixed_grasp is None else a.fixed_grasp
   if mode=='zero_suffix':return 0
   if mode=='rule_suffix':return rule(s)
   if mode=='one_update':
    if seen[0]:return 0
    seen[0]=True;return m.choose(s)
   if mode=='budget_refine_zplus':
    return 4 if s[49]<=6/7+1e-5 and s[41]<=.28 else 0
   if mode=='phase3_zplus_budget':
    # Posthoc strong-rule comparator, derived from a seen development rescue.
    # Same legal observed phase/budget and same action4, no planner changes.
    if not seen[0] and abs(s[49]-6/7)<1e-5 and s[41]<=.15:
     seen[0]=True;return 4
    return 0
   if mode=='full':return m.choose(s)
   raise ValueError(mode)
  row,t,d=rt.run(seed,policy,mode);rows.append(row);print(mode,seed,row['success'],flush=True)
rt.close();(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2))
