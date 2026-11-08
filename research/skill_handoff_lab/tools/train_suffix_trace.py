"""Own-policy full-native continuation experiment, preserving strong checkpoint."""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import torch
from suffix_trace_learner import Learner
from program_full_option import FullJoint
from program_rule_option import Assets
from program_atomic_feedback import AtomicRuntime
from train_option_online import atomic_json,atomic_torch
from train_joint_continuation import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--initial',type=Path,required=True);p.add_argument('--mode',choices=['sac','trace','lagged'],required=True);p.add_argument('--seed',type=int,default=32401);p.add_argument('--episodes',type=int,default=64);p.add_argument('--train-start',type=int,default=769100);p.add_argument('--dev-start',type=int,default=769400);p.add_argument('--dev-count',type=int,default=12);p.add_argument('--updates',type=int,default=40);p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(2)
 manifest={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()if k not in ['resume','stop_after']};manifest.update(initial_sha=sha(a.initial),sources={n:sha(Path(__file__).with_name(n))for n in ['train_suffix_trace.py','suffix_trace_learner.py','train_full_option.py','program_full_option.py','program_atomic_feedback.py','peg_atomic_macro.py']},train=list(range(a.train_start,a.train_start+a.episodes)),dev=list(range(a.dev_start,a.dev_start+a.dev_count)),native_horizon=100,selection='fixed final64; pilot32; no best-dev selection',gamma=1,trace_lambda=.9,max_trace_options=8,reference_period_episodes=8,replay='half inherited replay half own-policy; historical missing behavior likelihood never fabricated',front='common warm real grasp; rear repeated actual Options',seed_scope='same pretrained weights and optimizers, distinct continuation sampling seed; NOT independent initialization')
 assert not set(manifest['train'])&set(manifest['dev']);a.out.mkdir(parents=True,exist_ok=True);assets=Assets();m=Learner(assets,a.seed,a.mode)
 if a.resume:
  assert json.loads((a.out/'MANIFEST.json').read_text())==manifest
  d=torch.load(a.out/'CURRENT.pt',weights_only=False);m.load(d['model']);data=d['data'];history=d['history'];cursor=d['cursor']
 else:
  assert not(a.out/'MANIFEST.json').exists();d=torch.load(a.initial,weights_only=False);m.load(d['model']);m.rng=np.random.default_rng(a.seed);data=copy.deepcopy(d['data']);history=[];cursor=0
  assert all('mu'not in t for t in data);atomic_json(a.out/'MANIFEST.json',manifest)
  for n in manifest['sources']:shutil.copy2(Path(__file__).with_name(n),a.out/n)
  atomic_torch(a.out/'INITIAL.pt',m.state());atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=0))
 rt=AtomicRuntime(a.out/'physics');rt.counter=max([0]+[int(f.name.split('_')[0])for f in rt.out.glob('*.json.gz')]);rt.rows=[h['row']for h in history];jobs=[]
 for i in range(a.episodes+1):
  if i in [0,32,64,a.episodes]:jobs.extend([('eval',i,s)for s in manifest['dev']])
  if i<a.episodes:jobs.append(('train',i,a.train_start+i))
 try:
  for ji in range(cursor,len(jobs)):
   kind,i,seed=jobs[ji];m.episode=i;m.refresh(i);pending=a.out/f'PENDING_{ji:04d}.pt'
   if pending.exists():item=torch.load(pending,weights_only=False);assert item['job']==jobs[ji];m.rng.bit_generator.state=item['rng']
   else:
    start=a.out/f'START_{ji:04d}.json';assert not start.exists(),'Inspect incomplete physics before resume';atomic_json(start,dict(job=jobs[ji]));m.behavior=[]
    pol=FullJoint(assets,lambda s:m.choose(s,kind=='train'),lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,f'{kind}_{i}');ts=m.decorate(pol.transitions(ts),i)
    assert len(ts)==len(m.behavior)
    for t,choice in zip(ts,m.behavior):assert t['a']==choice['a'];t['mu']=choice['mu'];t['policy_episode']=i
    item=dict(job=jobs[ji],row=row,ts=ts,events=pol.events,rng=m.rng.bit_generator.state);atomic_torch(pending,item)
   log=None
   if kind=='train':
    base=len(data)
    for j,t in enumerate(item['ts']):
     if j+1<len(item['ts']):t['successor_index']=base+j+1;assert np.allclose(t['next'],item['ts'][j+1]['s'],atol=1e-7)
    data.extend(item['ts']);log=m.train(data,a.updates,i)
   history.append(dict(kind=kind,checkpoint=i,row=item['row'],events=item['events'],effective_decisions=len(item['ts']),log=log));atomic_json(a.out/'HISTORY.json',history);atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=ji+1))
   if kind=='eval'and seed==manifest['dev'][-1]:atomic_torch(a.out/f'CHECKPOINT_{i}.pt',m.state())
   print(a.mode,a.seed,kind,i,seed,item['row']['success'],len(item['ts']),flush=True)
   if kind=='eval'and seed==manifest['dev'][-1]and a.stop_after==i:break
 finally:rt.close()
if __name__=='__main__':main()
