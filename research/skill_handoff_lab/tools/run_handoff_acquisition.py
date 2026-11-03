"""Equal-budget adaptive pair acquisition; every pair reexecutes both full prefixes."""
import argparse,hashlib,json,gzip,shutil
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,prior,normalizer
from program_peg_option_runtime import Runtime
from program_handoff_acquisition import choose_record

def write(p,d):p.write_text(json.dumps(d,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',required=True);p.add_argument('--seed',type=int,default=31301);p.add_argument('--pairs',type=int,default=8);a=p.parse_args();a.out.mkdir(exist_ok=False);torch.set_num_threads(2);data,_=prior(a.prior);model=Learner('dsac',a.seed,normalizer(data));model.train(data,steps=800);rng=np.random.default_rng(a.seed);online=[];pool=[];history={};curves=[];rt=Runtime(a.out/'physics')
 for name in ['program_discrete_options.py','program_peg_option_runtime.py','program_handoff_acquisition.py','run_handoff_acquisition.py']:shutil.copy2(Path(__file__).with_name(name),a.out/name)
 write(a.out/'PROTOCOL.json',dict(mode=a.mode,seed=a.seed,warmup_scenes=list(range(751000,751008)),dev_scenes=list(range(751200,751206)),warmup_episodes=8,paired_episodes=a.pairs*2,physical_cost_per_pair=200,optimizer_updates_per_episode=50,prior_transitions=len(data),candidate='virtual matched-label successor TD step then predecessor Bellman step; expected change of training-anchor greedy decisions; no deployment critic modification',posterior_limitation='plug-in independent Bernoulli branch weights, not calibrated posterior',privacy='public simulation only',sealed=False))
 def evaluate(k):
  model.save(a.out/f'MODEL_{k}.pt');rows=[rt.run(s,model.choose,f'dev{k}')[0]for s in range(751200,751206)];curves.append(dict(episodes=k,updates=model.updates,success=sum(r['success']for r in rows),n=len(rows),sources=[r['path']for r in rows]));write(a.out/'CURVE.json',curves)
 try:
  evaluate(0)
  for i,s in enumerate(range(751000,751008)):
   act0=int(rng.integers(3));act1=int(rng.integers(2));row,ts,d=rt.run(s,lambda o:act0 if o[44]<.5 else act1,f'warmup{i}');online+=ts;model.train(data,online,steps=50)
   if len(ts)==2:
    pool.append(dict(scene=s,grasp=act0,prefix=ts[0],suffix=ts[1]));history[len(pool)-1]=[int(row['success'])]
  assert pool;evaluate(8);write(a.out/'POOL.json',pool)
  anchors=[t['s']for t in data+online if t['stage']==0]
  for j in range(a.pairs):
   idx,acq=choose_record(model,pool,a.mode,rng,anchors,history);r=pool[idx];ds=[]
   for b in [0,1]:
    row,ts,d=rt.run(r['scene'],lambda o:r['grasp'] if o[44]<.5 else b,f'pair{j}_{b}');online+=ts;ds.append(d);history[idx].append(int(row['success']))
   n=r['prefix']['duration'];assert ds[0]['initial']==ds[1]['initial'] and ds[0]['steps'][:n]==ds[1]['steps'][:n]
   before_actions=[model.choose(o)for o in anchors];before_suf=model.choose(r['suffix']['s']);before_first=model.choose(r['prefix']['s'])
   acq.update(scene=r['scene'],grasp=r['grasp'],prefix_steps=n,exact_prefix=True,outcomes=[int(d['summary']['success'])for d in ds]);model.train(data,online,steps=100)
   after_actions=[model.choose(o)for o in anchors];acq.update(actual_anchor_switch_fraction=float(np.mean(np.array(before_actions)!=after_actions)),before_suffix=before_suf,after_suffix=model.choose(r['suffix']['s']),before_first=before_first,after_first=model.choose(r['prefix']['s']),before_anchor_actions=before_actions,after_anchor_actions=after_actions)
   write(a.out/f'ACQUIRE_{j:03d}.json',acq);write(a.out/'ONLINE.json',online)
   if (j+1)%4==0:evaluate(8+2*(j+1))
  model.save(a.out/'FINAL.pt');write(a.out/'HISTORY.json',model.history)
 finally:rt.close()
if __name__=='__main__':main()
