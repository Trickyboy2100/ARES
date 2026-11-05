"""Shared fresh-native nominal warmup for expanded observation/action baseline."""
import argparse,json,torch,numpy as np,hashlib
from pathlib import Path
from program_atomic_feedback import AtomicRuntime
from program_budget_feedback_sac import FeedbackSAC,completion_labels,normalization
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=31701);p.add_argument('--count',type=int,default=16);p.add_argument('--start',type=int,default=761000);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
rt=AtomicRuntime(a.out/'physics');data=[];labels=[];rows=[]
try:
 for seed in range(a.start,a.start+a.count):
  r,ts,d=rt.run(seed,lambda s:1 if s[44]<.5 else 0,'shared_nominal');data+=ts;labels+=completion_labels(d,ts);rows.append(r);print(seed,r['success'],flush=True)
finally:rt.close()
mu,sd=normalization(data);m=FeedbackSAC('ordinary',a.seed,mu,sd,nobs=83,nactions=12);m.save(a.out/'RANDOM.pt')
# Standard supervised nominal-policy initialization, all successes/failures kept.
x=m.x([t['s']for t in data]);actions=torch.tensor([t['a']for t in data]);idx=torch.arange(len(data))
for k in range(512):
 prob=m.probs(x);loss=-prob[idx,actions].clamp_min(1e-9).log().mean();m.aopt.zero_grad();loss.backward();m.aopt.step()
m.save(a.out/'INITIAL.pt');torch.save(dict(data=data,labels=labels),a.out/'WARM_DATA.pt')
(a.out/'MANIFEST.json').write_text(json.dumps(dict(seed=a.seed,seeds=list(range(a.start,a.start+a.count)),episodes=len(rows),physical_steps=sum(x['steps']for x in rows),actor_supervised_updates=512,critic_updates=0,training_data_reuse='actor initialization only; each online arm starts empty replay',normalization='shared observed nominal training data',actions=12,nobs=83,sha=hashlib.sha256((a.out/'INITIAL.pt').read_bytes()).hexdigest()),indent=2))
