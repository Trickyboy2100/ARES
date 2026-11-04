"""Fixed real predecessor, native100 successor SAC single-factor retention study.
Action zero continues the current queue; this is not a bounded motor residual.
"""
import argparse,json,hashlib,shutil,time,subprocess
from pathlib import Path
import numpy as np
import torch
from program_budget_feedback_sac import FeedbackSAC,completion_labels
from program_peg_feedback import FeedbackRuntime
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
p.add_argument('--episodes',type=int,default=32);p.add_argument('--start',type=int,default=756000)
p.add_argument('--success-bc',type=float,default=0);p.add_argument('--exploration',choices=['original','controlled'],default='original')
p.add_argument('--dev-count',type=int,default=16);p.add_argument('--seed',type=int,default=31601)
p.add_argument('--baseline',action='store_true')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
m=FeedbackSAC.load(a.parent/'FINAL.pt');assert m.mode=='ordinary'
data=json.loads((a.parent/'TRANSITIONS.json').read_text());labels=json.loads((a.parent/'LABELS.json').read_text())
rng=np.random.default_rng(a.seed);rt=FeedbackRuntime(a.out/'physics');rows=[];online=[]
assert all(t['controller']==rt.controller_id for t in data)
protocol=dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},parent_sha=hashlib.sha256((a.parent/'FINAL.pt').read_bytes()).hexdigest(),base=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),predecessor=1,horizon=100,updates_per_episode=40,successor_only=True,cql_online=0,controlled='epsilon .1 uniform otherwise deterministic actor; original epsilon .1 otherwise stochastic actor',claim='mature retention/exploration controls; not novel',data_sha=hashlib.sha256((a.parent/'TRANSITIONS.json').read_bytes()).hexdigest())
(a.out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));src=a.out/'source';src.mkdir()
for f in [Path(__file__).name,'program_budget_feedback_sac.py','program_peg_feedback.py','program_peg_option_runtime.py']:shutil.copy2(Path(__file__).parent/f,src/f)
def evaluate(n):
 result=[]
 for seed in range(753200,753200+a.dev_count):
  row,t,d=rt.run(seed,lambda s:1 if s[44]<.5 else (0 if a.baseline else m.choose(s)),f'dev_{n}')
  result.append(row)
 (a.out/f'EVAL_{n}.json').write_text(json.dumps(result,indent=2));print('EVAL',n,sum(r['success']for r in result),len(result),flush=True)
evaluate(0);m.save(a.out/'INITIAL.pt')
for i in range(0 if a.baseline else a.episodes):
 def policy(s):
  if s[44]<.5:return 1
  if rng.random()<.1:return int(rng.integers(8))
  return m.choose(s,stochastic=a.exploration=='original')
 row,t,d=rt.run(a.start+i,policy,f'train_{i}');online.extend(t);data.extend(t);labels.extend(completion_labels(d,t));rows.append(row)
 log=m.train(data,labels,40,success_bc=a.success_bc,cql=0,successor_only=True)
 print('TRAIN',i,row['success'],len(t),log['q_loss'],flush=True)
 if (i+1)%16==0 or i+1==a.episodes:
  m.save(a.out/f'CHECKPOINT_{i+1}.pt');evaluate(i+1)
  (a.out/'TRANSITIONS.json').write_text(json.dumps(data));(a.out/'LABELS.json').write_text(json.dumps(labels));(a.out/'ONLINE.json').write_text(json.dumps(online));(a.out/'TRAIN.json').write_text(json.dumps(rows))
m.save(a.out/'FINAL.pt');rt.close();print('DONE',flush=True)
