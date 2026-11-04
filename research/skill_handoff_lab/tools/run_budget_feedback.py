"""Actual native-prefix/online-feedback experiments; no handoff reset."""
import argparse,json,gzip,hashlib,shutil,subprocess,time
from pathlib import Path
import numpy as np,torch
from program_peg_feedback import FeedbackRuntime,rule
from program_budget_feedback_sac import FeedbackSAC,completion_labels,normalization
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--warm',type=Path);p.add_argument('--mode',choices=['collect','ordinary','suffix','budget','budget_blind','baselines'],required=True);p.add_argument('--seed',type=int,default=31501);p.add_argument('--episodes',type=int,default=96);p.add_argument('--start',type=int,default=753100);p.add_argument('--dev-start',type=int,default=753200);p.add_argument('--dev-count',type=int,default=16);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),native_horizon=100,full_prefix=True,dense_shaping='same potential telescope',history_current_window=48,updates_per_episode=40,pretraining_updates=1000),indent=2))
source=a.out/'source';source.mkdir()
for name in ['program_peg_feedback.py','program_budget_feedback_sac.py','program_peg_option_runtime.py',Path(__file__).name]:shutil.copy2(Path(__file__).parent/name,source/name)
rt=FeedbackRuntime(a.out/'physics');rng=np.random.default_rng(a.seed);data=[];labels=[];history=[];online=[];recent=[]
def store(row,t,d):
 data.extend(t);labels.extend(completion_labels(d,t));history.append(row)
if a.mode=='collect':
 for i in range(a.episodes):
  first=i%3;seed=a.start+i//2
  def policy(s):
   if s[44]<.5:return first
   if i%2==0:return rule(s)
   return int(rng.choice([0,1,2,3,4,5,6,7],p=[.6,.12,.05,.05,.05,.05,.04,.04]))
  row,t,d=rt.run(seed,policy,'warm');store(row,t,d);print('COLLECT',i,seed,row['success'],len(t),row['error'],flush=True)
elif a.mode=='baselines':
 for i in range(a.dev_count):
  for name in ['zero','refresh','feedback']:
   def policy(s):
    if s[44]<.5:return 1
    if name=='zero':return 0
    if name=='refresh':return 1 if s[-1]==0 else 0
    return rule(s)
   row,t,d=rt.run(a.dev_start+i,policy,name);store(row,t,d);print('BASELINE',i,name,row['success'],flush=True)
else:
 data=json.loads((a.warm/'TRANSITIONS.json').read_text());labels=json.loads((a.warm/'LABELS.json').read_text());initial_data=list(data);initial_labels=list(labels)
 m=FeedbackSAC(a.mode,a.seed,*normalization(data));m.save(a.out/'INITIAL.pt');m.train(data,labels,1000,bc=True);m.save(a.out/'PRETRAIN.pt')
 def evaluate(checkpoint):
  rows=[]
  for j in range(a.dev_count):
   row,t,d=rt.run(a.dev_start+j,lambda s:m.choose(s),f'dev_{checkpoint}');rows.append(row)
  (a.out/f'EVAL_{checkpoint}.json').write_text(json.dumps(rows,indent=2));print('EVAL',checkpoint,sum(r['success']for r in rows),'/',len(rows),flush=True)
 evaluate(0)
 for i in range(a.episodes):
  def policy(s):
   if rng.random()<.1:return int(rng.integers(3 if s[44]<.5 else 8))
   return m.choose(s,stochastic=True)
  row,t,d=rt.run(a.start+i,policy,f'online_{i}');online.extend(t);recent.append(completion_labels(d,t));store(row,t,d)
  # Recent real suffix outcomes follow evolving policy. Warm labels are used
  # only until enough online outcomes exist; all methods use identical rule.
  hl=[z for episode in recent[-48:]for z in episode]or initial_labels
  log=m.train(initial_data+online,hl,40);print('TRAIN',a.mode,i,row['success'],len(t),log['q_loss'],flush=True)
  if (i+1)%32==0:
   m.save(a.out/f'CHECKPOINT_{i+1}.pt');evaluate(i+1)
 m.save(a.out/'FINAL.pt');(a.out/'TRAINING.json').write_text(json.dumps(m.history,indent=2))
rt.close();(a.out/'TRANSITIONS.json').write_text(json.dumps(data));(a.out/'LABELS.json').write_text(json.dumps(labels));(a.out/'ONLINE_TRANSITIONS.json').write_text(json.dumps(online));print('DONE',a.mode,flush=True)
