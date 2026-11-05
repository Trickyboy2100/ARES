"""Full joint-policy paired return update versus equal-data ordinary SAC.
Standard episodic policy gradient with an action-independent baseline. It is
not the old single-deviation bandit and is not claimed an original estimator.
"""
import argparse,copy,hashlib,json,time,shutil,gzip
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_execution_joint import Assets,Learner,Joint,AtomicRuntime,ROOT,WARM,SHARED,sha,atomic_json,atomic_torch
from program_rule_option import prefix_hash

def paired_update(m,ts,delta,steps):
 start=time.monotonic()
 if not delta:return dict(updates=0,wall_s=0.,delta=0)
 x=m.norm([t['s']for t in ts]);b=torch.tensor([t['base']for t in ts]);mask=torch.tensor([t['base_mask']for t in ts]);act=torch.tensor([t['a']for t in ts]);idx=torch.arange(len(ts))
 def logp():
  p=m.probs(x,b,mask);p=.9*p+.1*mask.float()/mask.sum(1,keepdim=True);return p[idx,act].clamp_min(1e-9).log().sum()
 with torch.no_grad():old=logp()
 for _ in range(steps):
  ratio=(logp()-old).clamp(-10,10).exp();loss=-torch.minimum(ratio*delta,ratio.clamp(.8,1.2)*delta)
  m.ao.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.actor.parameters(),1.);m.ao.step();m.updates+=1
 return dict(updates=steps,wall_s=time.monotonic()-start,delta=delta,ratio=float(ratio.detach()),loss=float(loss.detach()),critic_updates=0)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['sac','paired_pg'],default='sac');p.add_argument('--seed',type=int,default=31901);p.add_argument('--episodes',type=int,default=64);p.add_argument('--train-start',type=int,default=764100);p.add_argument('--dev-start',type=int,default=764300);p.add_argument('--dev-count',type=int,default=16);p.add_argument('--updates',type=int,default=40);p.add_argument('--resume',action='store_true');p.add_argument('--references',type=Path,required=True);p.add_argument('--layout',choices=['flat','semantic'],required=True);a=p.parse_args();torch.set_num_threads(2)
 a.out.mkdir(parents=True,exist_ok=True);manifest={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()if k!='resume'}
 manifest.update(source_sha=sha(__file__),execution_learner_sha=sha(Path(__file__).with_name('train_execution_joint.py')),dependencies={p.name:sha(p)for p in Path(__file__).parent.glob('program_*.py')},macro_sha=sha(Path(__file__).with_name('peg_atomic_macro.py')),warm_sha=sha(WARM),shared_replay_sha=sha(SHARED),shared_design_episodes=96,reference_sha=sha(a.references/'RESULTS.json'),native100=True,gamma=1,scope='three real grasps and recurrent completed Options; nominal rule is action0 at rear',checkpoint_selection='fixed final; no best dev',candidate='whole-policy terminal difference versus same-initial frozen joint baseline; clipped episodic score-function update, mature foundation',train=list(range(a.train_start,a.train_start+a.episodes)),dev=list(range(a.dev_start,a.dev_start+a.dev_count)))
 assert not set(manifest['train'])&set(manifest['dev'])
 assets=Assets();m=Learner(assets,a.seed,a.layout);references={v['seed']:v for v in json.loads((a.references/'RESULTS.json').read_text())};data=torch.load(SHARED,weights_only=False)['data'];
 for row in references.values():
  raw=json.load(gzip.open(row['path'],'rt'));rr=AtomicRuntime.__new__(AtomicRuntime);rr.controller_id=row['controller'];pol=Joint(None,None,None);pol.events=row['events'];data.extend(m.decorate(pol.transitions(rr.feedback_transitions(raw,row['path'])),-1000))
 history=[];cursor=0
 if a.resume:
  assert json.loads((a.out/'MANIFEST.json').read_text())==manifest
  saved=torch.load(a.out/'CURRENT.pt',weights_only=False);m.load(saved['model']);data=saved['data'];history=saved['history'];cursor=saved['cursor']
 else:
  assert not(a.out/'MANIFEST.json').exists();atomic_json(a.out/'MANIFEST.json',manifest);shutil.copy2(__file__,a.out/Path(__file__).name)
  atomic_torch(a.out/'INITIAL.pt',m.state());atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=0))
 assert (ROOT/'joint_continuation_20261106/identity_flat_32001/COMPATIBILITY.json').exists()
 rt=AtomicRuntime(a.out/'physics');rt.counter=max([0]+[int(f.name.split('_')[0])for f in rt.out.glob('*.json.gz')]);rt.rows=[h['row']for h in history]
 jobs=[]
 for i in range(a.episodes+1):
  if i in [0,a.episodes//2,a.episodes]:jobs.extend([('eval',i,s)for s in manifest['dev']])
  if i<a.episodes:jobs.append(('train',i,a.train_start+i))
 try:
  for ji in range(cursor,len(jobs)):
   kind,i,seed=jobs[ji];pending=a.out/f'PENDING_{ji:04d}.pt'
   if pending.exists():
    item=torch.load(pending,weights_only=False);assert item['job']==jobs[ji];m.rng.bit_generator.state=item['rng']
   else:
    started=a.out/f'START_{ji:04d}.json';assert not started.exists(),'Inspect incomplete native physics attempt';atomic_json(started,dict(job=jobs[ji]))
    pol=Joint(assets,lambda s:m.choose(s,kind=='train'),lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,f'{kind}_{i}');ts=m.decorate(pol.transitions(ts),i)
    item=dict(row=row,ts=ts,events=pol.events,rng=m.rng.bit_generator.state,job=jobs[ji]);atomic_torch(pending,item)
   log=None
   if kind=='train':
    ref=references[seed];actual=json.load(gzip.open(item['row']['path'],'rt'));reference=json.load(gzip.open(ref['path'],'rt'));assert prefix_hash(actual,0)==prefix_hash(reference,0)
    data.extend(item['ts']);delta=int(item['row']['success'])-int(ref['success'])
    log=paired_update(m,item['ts'],delta,a.updates)if a.arm=='paired_pg'else m.train(data,a.updates,'sac',i)
    log.update(reference_success=bool(ref['success']),terminal_delta=delta,initial_state_equal=True)
   history.append(dict(kind=kind,checkpoint=i,row=item['row'],events=item['events'],effective_decisions=len(item['ts']),log=log));atomic_json(a.out/'HISTORY.json',history)
   atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=ji+1))
   if kind=='eval'and seed==manifest['dev'][-1]:atomic_torch(a.out/f'CHECKPOINT_{i}.pt',m.state())
   print(a.arm,a.seed,kind,i,seed,item['row']['success'],len(item['ts']),flush=True)
 finally:rt.close()
if __name__=='__main__':main()
