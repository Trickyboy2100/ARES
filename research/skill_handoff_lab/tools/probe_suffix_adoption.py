"""Selected post-training causal diagnosis; never counted as learned policy gain."""
import argparse,json,torch,gzip,hashlib
from pathlib import Path
from program_rule_option import Assets
from program_full_option import FullJoint,Warm,legal
from program_atomic_feedback import AtomicRuntime
from train_full_option import Learner
from train_option_online import atomic_json
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'mechanism';out.mkdir(exist_ok=False);torch.set_num_threads(2);assets=Assets();warm=Warm(assets)
models={}
for name,path in [('trained',a.root/'trace_32401/CHECKPOINT_64.pt'),('prior',a.root/'trace_32401/INITIAL.pt')]:
 m=Learner(assets,32401,'sac');m.load(torch.load(path,weights_only=False));models[name]=m
atomic_json(out/'MANIFEST.json',dict(seed=769401,selection='Known development regression; fixed first selector6 chosen from inherited successful witness, NOT a deployable learned method',modes=['trained','force6_trained_suffix','force6_prior_suffix'],native100=True,extra_training=0,source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
rt=AtomicRuntime(out/'physics');rows=[]
try:
 for mode in ['trained','force6_trained_suffix','force6_prior_suffix']:
  used=[False]
  def actor(s):
   if s[44]<.5:return warm.choose(s)
   if mode!='trained' and not used[0]:
    used[0]=True;assert legal(s)[6];return 6
   return models['prior'if mode=='force6_prior_suffix'else'trained'].choose(s)
  pol=FullJoint(assets,actor,lambda:len(rt.env.records));row,ts,d=rt.run(769401,pol,mode);row['events']=pol.events;row['mode']=mode;rows.append(row);atomic_json(out/'CASES.json',rows);print(mode,row['success'],flush=True)
finally:rt.close()
base=json.load(gzip.open(rows[0]['path'],'rt'));comparisons=[]
for row in rows[1:]:
 other=json.load(gzip.open(row['path'],'rt'));n=0
 for x,y in zip(base['steps'],other['steps']):
  if not all(x[k]==y[k]for k in ['action','q','qvel','tcp','peg']):break
  n+=1
 bs=next(d['obs']for d in rows[0]['decisions']if d['stage']==1);ss=next(d['obs']for d in row['decisions']if d['stage']==1)
 bdec=next(d for d in rows[0]['decisions']if d['stage']==1);odec=next(d for d in row['decisions']if d['stage']==1)
 comparisons.append(dict(mode=row['mode'],equal_recorded_prefix=n,equal_boundary_obs=bs==ss,equal_queue=bdec['queue_before']==odec['queue_before'],equal_active_target=bdec['active_target_before']==odec['active_target_before'],success=row['success']))
atomic_json(out/'PREFIX.json',comparisons)
