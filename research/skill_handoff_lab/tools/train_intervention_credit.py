"""Matched real-data SAC/AWAC-style/measured-gap policy improvement prototype."""
import argparse,json,shutil,sys
from pathlib import Path
import torch,numpy as np
from program_budget_feedback_sac import FeedbackSAC,completion_labels
from program_peg_feedback import FeedbackRuntime
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--mode',choices=['sac','awac','measured_gap'],required=True);p.add_argument('--updates',type=int,default=1000);p.add_argument('--episodes',type=int,default=16);p.add_argument('--dev-start',type=int,default=757200);p.add_argument('--dev-count',type=int,default=24)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
# Persist verbose native planner diagnostics, including failures.
logfile=open(a.out/'RUN.log','w',buffering=1);sys.stdout=logfile;sys.stderr=logfile
m=FeedbackSAC.load(a.parent/'FINAL.pt');data=json.loads((a.parent/'TRANSITIONS.json').read_text())+json.loads((a.pairs/'TRANSITIONS.json').read_text());labels=json.loads((a.parent/'LABELS.json').read_text())+json.loads((a.pairs/'LABELS.json').read_text());pairs=json.loads((a.pairs/'PAIRS.json').read_text());rng=np.random.default_rng(31701)
(a.out/'PROTOCOL.json').write_text(json.dumps({**{k:str(v) for k,v in vars(a).items()},'predecessor':1,'online_start':757100,'seed':31701,'interpretation':'rollout-policy gap, not current-policy Q; prototype originality unverified','checkpoint_selection':'final only; intermediate diagnostic'},indent=2))
source=a.out/'source';source.mkdir()
for f in [Path(__file__).name,'program_budget_feedback_sac.py','program_peg_feedback.py','program_peg_option_runtime.py']:shutil.copy2(Path(__file__).parent/f,source/f)
for k in range(a.updates//100):
 log=m.train(data,labels,100,successor_only=True,intervention_pairs=pairs,actor_rule=a.mode);print('UPDATE',m.updates,log['actor_loss'],flush=True)
m.save(a.out/'OFFLINE.pt');rt=FeedbackRuntime(a.out/'physics')
def evaluate(tag):
 rows=[]
 for seed in range(a.dev_start,a.dev_start+a.dev_count):
  r,t,d=rt.run(seed,lambda s:1 if s[44]<.5 else m.choose(s),tag);rows.append(r);print('EVAL',tag,seed,r['success'],flush=True)
 (a.out/(tag+'.json')).write_text(json.dumps(rows))
evaluate('offline_eval')
for i in range(a.episodes):
 def policy(s):
  if s[44]<.5:return 1
  return int(rng.integers(8)) if rng.random()<.1 else m.choose(s)
 r,t,d=rt.run(757100+i,policy,f'online_{i}');data.extend(t);labels.extend(completion_labels(d,t))
 m.train(data,labels,40,successor_only=True,intervention_pairs=pairs,actor_rule=a.mode)
 print('ONLINE',i,r['success'],flush=True)
 if (i+1)%8==0:m.save(a.out/f'CHECKPOINT_{i+1}.pt')
m.save(a.out/'FINAL.pt');evaluate('final_eval');rt.close()
(a.out/'TRANSITIONS.json').write_text(json.dumps(data));(a.out/'LABELS.json').write_text(json.dumps(labels))
