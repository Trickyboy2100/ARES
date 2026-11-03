"""Revision: spend pair budget on unmeasured front options, not repeated labels.
Same full native physics and SAC; coverage constraint shared by all allocators.
"""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,prior,normalizer
from program_peg_option_runtime import Runtime
from program_handoff_acquisition import update_score,frontier_score,JointOutcomeKernel

def write(p,d):p.write_text(json.dumps(d,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',required=True,choices=['uniform','ees','handoff','local','policy','joint','independent_kernel']);p.add_argument('--seed',type=int,default=31301);p.add_argument('--anchors',choices=['prior_and_current','current_only'],default='prior_and_current');a=p.parse_args();a.out.mkdir(exist_ok=False);torch.set_num_threads(2);data,groups=prior(a.prior);kernel=JointOutcomeKernel(groups);model=Learner('dsac',a.seed,normalizer(data));model.train(data,steps=800);rng=np.random.default_rng(a.seed);online=[];pool={};history={};curves=[];rt=Runtime(a.out/'physics')
 for name in ['program_discrete_options.py','program_peg_option_runtime.py','program_handoff_acquisition.py','run_frontier_acquisition.py']:shutil.copy2(Path(__file__).with_name(name),a.out/name)
 write(a.out/'PROTOCOL.json',dict(mode=a.mode,seed=a.seed,anchor_distribution=a.anchors,revision='coverage shared; expand actual first-option frontier',training_scenes=list(range(751000,751008)),dev_scenes=list(range(751200,751206)),warmup=8,pairs=8,cost_per_pair=200,steps_per_episode=100,prior_source=str(a.prior),optimizer_steps_per_pair=100,claim='UNVERIFIED; active expected-update heuristic, no new value loss'))
 def evaluate(k):
  model.save(a.out/f'MODEL_{k}.pt');rows=[rt.run(s,model.choose,f'dev{k}')[0]for s in range(751200,751206)];curves.append(dict(episodes=k,updates=model.updates,success=sum(r['success']for r in rows),n=len(rows),sources=[r['path']for r in rows]));write(a.out/'CURVE.json',curves)
 try:
  evaluate(0)
  for i,s in enumerate(range(751000,751008)):
   act0=int(rng.integers(3));act1=int(rng.integers(2));row,ts,d=rt.run(s,lambda o:act0 if o[44]<.5 else act1,f'warmup{i}');online+=ts;model.train(data,online,steps=50)
   for aa in range(3):pool[(s,aa)]=dict(scene=s,grasp=aa,prefix=dict(ts[0],a=aa),suffix=None,known={},complete=False)
   entry=pool[(s,act0)];entry['known'][act1]=int(row['success']);entry['suffix']=ts[1]if len(ts)==2 else None
   if len(ts)==1:entry['complete']=True
  evaluate(8);anchors=[t['s']for t in (data+online if a.anchors=='prior_and_current'else online) if t['stage']==0]
  if a.mode=='policy':
   for j in range(16):
    def collect(obs):
     if rng.random()<.15:return int(rng.integers(3 if obs[44]<.5 else 2))
     return model.choose(obs,stochastic=True)
    row,ts,d=rt.run(751000+j%8,collect,f'train{j+9}');online+=ts;model.train(data,online,steps=50);write(a.out/'ONLINE.json',online)
    if (j+1)%8==0:evaluate(j+9)
   model.save(a.out/'FINAL.pt');write(a.out/'HISTORY.json',model.history);return
  for j in range(8):
   scores=[];details=[];keys=[k for k,r in pool.items()if not r['complete']];start=time.monotonic()
   for key in keys:
    r=pool[key];_,q=model.values(r['prefix']['s']);means=q[:,0,:3].mean(0);act=r['grasp'];hist=history.get(key,list(r['known'].values()))
    if a.mode=='uniform':score=float(rng.random());detail=None
    elif a.mode=='ees':
     comp=(1+sum(hist))/(2+len(hist));gain=.1 if len(hist)<3 else float(np.ptp([np.mean(hist[k:k+2])for k in range(0,len(hist),2)]));ext=min(1.,comp+gain);score=max(0.,means[act]+ext-comp-max(means))+.01/(1+len(hist));detail=dict(competence=comp,extrapolated=ext)
    elif r['suffix']is None:score,detail=frontier_score(model,r,anchors)
    else:
     weights=kernel.predict(r['suffix']['s'],a.mode=='independent_kernel')if a.mode in ['joint','independent_kernel']else None
     score,detail=update_score(model,r,anchors,a.mode!='local',joint=weights)
    scores.append(float(score));details.append(detail)
   ii=int(np.argmax(scores));key=keys[ii];r=pool[key];ds=[];before=[model.choose(o)for o in anchors];acq=dict(keys=keys,scores=scores,chosen=ii,details=details,virtual_wall_s=time.monotonic()-start,new_frontier=r['suffix']is None,known_before=r['known'].copy())
   for b in [0,1]:
    row,ts,d=rt.run(r['scene'],lambda o:r['grasp']if o[44]<.5 else b,f'pair{j}_{b}');online+=ts;ds.append(d);r['known'][b]=int(row['success']);r['suffix']=ts[1]if len(ts)==2 else None;r['prefix']=ts[0];history.setdefault(key,[]).append(int(row['success']))
   n=r['prefix']['duration'];assert ds[0]['initial']==ds[1]['initial']and ds[0]['steps'][:n]==ds[1]['steps'][:n];r['complete']=True;model.train(data,online,steps=100)
   acq.update(scene=r['scene'],grasp=r['grasp'],prefix_steps=n,exact_prefix=True,outcomes=[int(d['summary']['success'])for d in ds],actual_anchor_switch_fraction=float(np.mean(np.array(before)!=[model.choose(o)for o in anchors])),complete_prefixes=sum(x['complete']for x in pool.values()))
   write(a.out/f'ACQUIRE_{j:03d}.json',acq);write(a.out/'ONLINE.json',online)
   if (j+1)%4==0:evaluate(8+2*(j+1))
  model.save(a.out/'FINAL.pt');write(a.out/'HISTORY.json',model.history);write(a.out/'POOL.json',list(pool.values()))
 finally:rt.close()
if __name__=='__main__':main()
