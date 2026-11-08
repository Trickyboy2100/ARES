"""Own-policy three-arm continuation study; inherited transactional resume."""
import argparse,copy,json,shutil
from pathlib import Path
import torch
from continuation_learner import ContinuationLearner
from continuation_data import build
from collect_continuation_pairs import main as collect_pairs
from program_full_option import FullJoint
from program_rule_option import Assets
from program_atomic_feedback import AtomicRuntime
from train_joint_continuation import sha
from train_option_online import atomic_json,atomic_torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--shared',type=Path,required=True);p.add_argument('--arm',choices=['sac','ranking','retention'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--episodes',type=int,default=128);p.add_argument('--train-start',type=int,default=768100);p.add_argument('--dev-start',type=int,default=768400);p.add_argument('--dev-count',type=int,default=16);p.add_argument('--updates',type=int,default=40);p.add_argument('--front-unlock',type=int,default=100000);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
 manifest={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()if k not in ('resume','stop_after')};manifest.update(source={f:sha(Path(__file__).with_name(f))for f in ['train_continuation_retention.py','continuation_learner.py','continuation_data.py','collect_continuation_pairs.py','program_full_option.py','program_atomic_feedback.py','peg_atomic_macro.py']},shared_sha=sha(a.shared),native100=True,gamma=1,exploration='shared 64-episode ramp from warm behavior to learned stochastic policy, plus10% legal uniform; new action prior logit -3',selection='fixed final; checkpoints0/64/128 when reached',train=list(range(a.train_start,a.train_start+a.episodes)),dev=list(range(a.dev_start,a.dev_start+a.dev_count)),seed_scope='torch reseeded AFTER loading shared Gate; distinct new actor hidden and critic weights',candidate='version-bound physical preference plus empirical successful-suffix trust region; standard ranking/KL components, independent benefit unverified')
 assert not set(manifest['train'])&set(manifest['dev']);assets=Assets();m=ContinuationLearner(assets,a.seed,a.arm);data=[];history=[];cursor=0
 if a.resume:
  assert json.loads((a.out/'MANIFEST.json').read_text())==manifest;d=torch.load(a.out/'CURRENT.pt',weights_only=False);m.load(d['model']);data=d['data'];history=d['history'];cursor=d['cursor']
 else:
  assert not(a.out/'MANIFEST.json').exists();shared=torch.load(a.shared,weights_only=False);data=m.decorate(copy.deepcopy(shared['data']),-1000);atomic_json(a.out/'MANIFEST.json',manifest);shutil.copy2(__file__,a.out/Path(__file__).name);atomic_torch(a.out/'INITIAL.pt',m.state());atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=0))
 shared=torch.load(a.out/'EVIDENCE.pt',weights_only=False)if (a.out/'EVIDENCE.pt').exists()else torch.load(a.shared,weights_only=False)
 m.set_evidence(shared['pairs'],shared['anchors'],shared['reference_sha'])
 rt=AtomicRuntime(a.out/'physics');rt.counter=max([0]+[int(f.name.split('_')[0])for f in rt.out.glob('*.json.gz')]);rt.rows=[h['row']for h in history if 'row'in h];jobs=[]
 for i in range(a.episodes+1):
  if i in [0,64,128,a.episodes]:jobs.extend([('eval',i,s)for s in manifest['dev']])
  if i==64 and i<a.episodes:jobs.append(('refresh',i,768300))
  if i<a.episodes:jobs.append(('train',i,a.train_start+i))
 try:
  for ji in range(cursor,len(jobs)):
   kind,i,seed=jobs[ji];m.episode=i
   if kind=='refresh':
    rt.close();folder=a.out/'refresh64';ref=a.out/'REFERENCE_64.pt'
    if not ref.exists():atomic_torch(ref,m.state())
    if not (folder/'COMPLETE.json').exists():
     assert not folder.exists(),'Inspect incomplete paired collection before resume'
     collect_pairs(['--out',str(folder),'--start',str(seed),'--count','16','--reference',str(ref)]+(['--freeze-front']if i<a.front_unlock else []))
     atomic_json(folder/'COMPLETE.json',dict(completed=True))
    evidence=build([folder],m);data.extend(m.decorate(evidence['data'],i));atomic_torch(a.out/'EVIDENCE.pt',evidence);m.set_evidence(evidence['pairs'],evidence['anchors'],evidence['reference_sha'])
    history.append(dict(kind='refresh',checkpoint=i,episodes=evidence['episodes'],pairs=len(evidence['pairs']),reference_sha=evidence['reference_sha']))
    atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=ji+1));atomic_json(a.out/'HISTORY.json',history)
    rt=AtomicRuntime(a.out/'physics');rt.counter=max([0]+[int(f.name.split('_')[0])for f in rt.out.glob('*.json.gz')]);rt.rows=[h['row']for h in history if 'row'in h]
    continue
   pending=a.out/f'PENDING_{ji:04d}.pt'
   if pending.exists():item=torch.load(pending,weights_only=False);assert item['job']==jobs[ji];m.rng.bit_generator.state=item['rng']
   else:
    start=a.out/f'START_{ji:04d}.json';assert not start.exists(),'Inspect incomplete physics before resume';atomic_json(start,dict(job=jobs[ji]));pol=FullJoint(assets,lambda s:m.warm.choose(s)if s[44]<.5 and i<a.front_unlock else m.choose(s,kind=='train'),lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,f'{kind}_{i}');ts=m.decorate(pol.transitions(ts),i);item=dict(row=row,ts=ts,events=pol.events,rng=m.rng.bit_generator.state,job=jobs[ji]);atomic_torch(pending,item)
   log=None
   if kind=='train':data.extend(item['ts']);log=m.train(data,a.updates,i)
   history.append(dict(kind=kind,checkpoint=i,row=item['row'],events=item['events'],effective_decisions=len(item['ts']),log=log));atomic_json(a.out/'HISTORY.json',history);atomic_torch(a.out/'CURRENT.pt',dict(model=m.state(),data=data,history=history,cursor=ji+1))
   if kind=='eval'and seed==manifest['dev'][-1]:atomic_torch(a.out/f'CHECKPOINT_{i}.pt',m.state())
   print(a.arm,a.seed,kind,i,seed,item['row']['success'],len(item['ts']),flush=True)
   if kind=='eval'and seed==manifest['dev'][-1]and a.stop_after==i:break
 finally:rt.close()
if __name__=='__main__':main()
