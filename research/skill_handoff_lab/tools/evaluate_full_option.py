"""Frozen full native comparison; all permissions shared, no updates."""
import argparse,torch,json,shutil
from pathlib import Path
from program_full_option import FullJoint,Warm,legal
from program_rule_option import Assets,rule
from program_atomic_feedback import AtomicRuntime
from train_full_option import Learner
from train_option_online import atomic_json
from train_joint_continuation import sha
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--extraction',choices=['actor','meanq','actor_proposal_support','actor_old_gate_support','actor_front_meanq_rear','meanq_front_actor_rear'],default='actor');p.add_argument('--mode',choices=['warm','rule','learned','fixed5'],required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--count',type=int,default=24);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=False);assets=Assets();warm=Warm(assets);m=None
shutil.copy2(__file__,a.out/'evaluate_full_option.py')
if a.model:
 d=torch.load(a.model,weights_only=False)
 if d['arm'].startswith('chart_'):
  from train_target_chart_option import Learner as ChartLearner
  m=ChartLearner(assets,32201,d['coordinate'])
 else:m=Learner(assets,32201,d['arm'])
 m.load(d)
assert (a.mode=='learned')==(m is not None)
atomic_json(a.out/'MANIFEST.json',dict(mode=a.mode,extraction=a.extraction,model=str(a.model),model_sha=sha(a.model)if a.model else None,source=sha(__file__),seeds=list(range(a.start,a.start+a.count)),native100=True,updates=0))
rt=AtomicRuntime(a.out/'physics');rows=[]
try:
 for seed in range(a.start,a.start+a.count):
  used=[False]
  def fixed(s):
   if s[44]>=.5 and not used[0]:
    used[0]=True
    if legal(s)[6]:return 6
   return warm.choose(s)
  def learned(s):
   if a.extraction=='actor':return m.choose(s)
   if a.extraction=='actor_front_meanq_rear'and s[44]<.5:return m.choose(s)
   if a.extraction=='meanq_front_actor_rear'and s[44]>=.5:return m.choose(s)
   if a.extraction in ['actor_proposal_support','actor_old_gate_support']:
    b,mask=m.context([s])
    if s[44]>=.5:
     support=torch.zeros(13,dtype=torch.bool);support[0]=True
     for physical in assets.proposals(s):
      if a.extraction=='actor_proposal_support' or physical!=rule(s):support[physical+1]=True
     mask[0]&=support
    with torch.no_grad():return int(m.probs(m.norm([s]),b,mask,[s])[0].argmax())
   with torch.no_grad():q=torch.stack([v(m.norm([s]))for v in m.q]).mean(0)[0];q[~torch.tensor(legal(s))]=-1e9
   return int(q.argmax())
  actor=learned if m else (lambda s:assets.front(s)if s[44]<.5 else 0)if a.mode=='rule'else fixed if a.mode=='fixed5'else warm.choose
  pol=FullJoint(assets,actor,lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,'frozen');row['events']=pol.events;rows.append(row);atomic_json(a.out/'RESULTS.json',rows);print(a.mode,seed,row['success'],flush=True)
finally:rt.close()
