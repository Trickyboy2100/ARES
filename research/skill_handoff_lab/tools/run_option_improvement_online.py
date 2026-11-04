"""Own-policy native100 interaction after equal compatible physical pretraining."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np,torch
from program_discrete_options import prior
from program_option_improvement import Improved
from program_peg_option_runtime import Runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--design',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--episodes',type=int,default=32);p.add_argument('--start',type=int,default=752100);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
 m=Improved.load(a.checkpoint);old,_=prior(a.prior);base=old+json.loads(a.design.read_text());rng=np.random.default_rng(m.seed+100);online=[];history=[];rt=Runtime(a.out/'physics')
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),method=m.kind,seed=m.seed,train_scenes=list(range(a.start,a.start+a.episodes)),episodes=a.episodes,epsilon=.15,updates_per_episode=25,pretraining_episodes=252,prior_replay='60 historical plus192 new compatible shared physical episodes',claim='UNVERIFIED',evaluation='separate frozen real branch evaluation'),indent=2))
 try:
  m.save(a.out/'MODEL_0.pt')
  for k in range(a.episodes):
   def act(o):
    n=3 if o[44]<.5 else 2
    return int(rng.integers(n))if rng.random()<.15 else m.choose(o,stochastic=True)
   r,t,d=rt.run(a.start+k,act,f'online{k+1}');online+=t;metrics=m.train(base,online,steps=25);history.append(dict(episode=k+1,physics_steps=100*(k+1),decisions=len(online),updates=m.updates,source=r['path'],success=r['success'],metrics=metrics))
   (a.out/'ONLINE.json').write_text(json.dumps(online));(a.out/'HISTORY.json').write_text(json.dumps(history,indent=2))
   if (k+1)%16==0 or k+1==a.episodes:m.save(a.out/f'MODEL_{k+1}.pt')
  m.save(a.out/'FINAL.pt')
 finally:rt.close()
if __name__=='__main__':main()
