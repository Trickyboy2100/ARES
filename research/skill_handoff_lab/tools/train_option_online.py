"""Own-policy native Option learning with immutable protocol and atomic resume.
Mature SAC / persistent success BC controls, not an independent method claim.
Resume is at durable episode/update boundaries, never an approximate PhysX reset.
"""
import argparse, fcntl, hashlib, json, os, random, shutil, subprocess
from pathlib import Path
import numpy as np
import torch
from program_budget_feedback_sac import FeedbackSAC, completion_labels
from program_peg_feedback import FeedbackRuntime
from program_continuation_option import ContinuationOption

ROOT=Path('/home/andyee/robot-research/ARES-Skilllab')
PARENT=ROOT/'runs/research_program/EXP-004/success_retention_20261104/controlled/FINAL.pt'
# V1->V2 changes only resume provenance checking/idempotent completion.
LEGACY_RESUME_TRAINERS={'2df918c72b83beaef3e4f130da97ed7ff35e092753acedfa24eebea97d888627'}
KEYS=['q','target','actor','completion','scalar','qopt','aopt','hopt']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2));os.replace(t,p)
def atomic_torch(p,obj):
 t=p.with_suffix(p.suffix+'.tmp');torch.save(obj,t);os.replace(t,p)
def rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state())
def restore(x):random.setstate(x[0]);np.random.set_state(x[1]);torch.set_rng_state(x[2])
def state(m):return dict(weights={k:getattr(m,k).state_dict() for k in KEYS},rng=m.rng.bit_generator.state,updates=m.updates,history=m.history,alpha=m.alpha)
def load(m,x):
 for k,v in x['weights'].items():getattr(m,k).load_state_dict(v)
 m.rng.bit_generator.state=x['rng'];m.updates=x['updates'];m.history=x['history'];m.alpha=x['alpha']
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=['sac','success_bc'],required=True)
 p.add_argument('--episodes',type=int,default=64);p.add_argument('--train-start',type=int,default=759000);p.add_argument('--dev-start',type=int,default=759200);p.add_argument('--dev-count',type=int,default=16)
 p.add_argument('--seed',type=int,default=31601);p.add_argument('--updates',type=int,default=80);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--pause-after-collection',type=int)
 a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
 lock=open(a.out/'LOCK','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sources={f.name:sha(f) for f in Path(__file__).parent.glob('program_*.py')}
 sources[Path(__file__).name]=sha(__file__)
 train=list(range(a.train_start,a.train_start+a.episodes));dev=list(range(a.dev_start,a.dev_start+a.dev_count));assert not set(train)&set(dev)
 manifest=dict(schema=1,arm=a.arm,train=train,development=dev,environment='PegInsertionSide-v1',horizon=100,success='official terminal evaluate',parent=str(PARENT),parent_sha=sha(PARENT),sources=sources,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),initialization='same inherited neural weights31501; seed below controls replay/exploration, NOT independent neural initialization',optimizer_rng_seed=a.seed,updates_per_episode=a.updates,termination='insertion_queue',frame='measured',predecessor='learned real3 grasp offsets',data_scope='own policy online; no other arm replay',checkpoint_rule='0, half, final; final is main endpoint; no best-dev selection',epsilon='0.15 uniform legal actions at reopened Option boundaries',sealed_used=False)
 mf=a.out/'MANIFEST.json'
 if a.resume:
  recorded=json.loads(mf.read_text())
  # Git commit is provenance. Exact source hashes, not unrelated later docs
  # commits, define compatibility. One explicitly audited V1 migration only.
  manifest['source_commit']=recorded['source_commit']
  old_trainer=recorded['sources'].get(Path(__file__).name)
  if old_trainer in LEGACY_RESUME_TRAINERS:manifest['sources'][Path(__file__).name]=old_trainer
  assert recorded==manifest,'Resume protocol/source/seed mismatch'
 else:
  assert not mf.exists(),'Use --resume; never overwrite a run'
  atomic_json(mf,manifest);src=a.out/'source';src.mkdir()
  for f in sources:shutil.copy2(Path(__file__).parent/f,src/f)
 m=FeedbackSAC.load(PARENT);reference=FeedbackSAC.load(PARENT)
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);m.rng=np.random.default_rng(a.seed)
 marks=sorted(set([0,a.episodes//2,a.episodes]));jobs=[]
 for i in range(a.episodes+1):
  if i in marks:
   for seed in dev:jobs.append(dict(kind='eval',checkpoint=i,seed=seed))
  if i<a.episodes:jobs.append(dict(kind='train',checkpoint=i,seed=train[i]))
 data=[];labels=[];records=[];cursor=0
 current=a.out/'CURRENT.json'
 if a.resume:
  pointer=json.loads(current.read_text());cp=a.out/pointer['checkpoint'];assert sha(cp)==pointer['sha256']
  saved=torch.load(cp,weights_only=False);assert saved['manifest_sha']==sha(mf)
  load(m,saved['model']);restore(saved['global_rng']);data=saved['data'];labels=saved['labels'];records=saved['records'];cursor=saved['cursor']
 def commit():
  cp=a.out/f'RESUME_{cursor:05d}.pt'
  atomic_torch(cp,dict(manifest_sha=sha(mf),model=state(m),global_rng=rng(),data=data,labels=labels,records=records,cursor=cursor))
  atomic_json(current,dict(checkpoint=cp.name,cursor=cursor,sha256=sha(cp)))
  atomic_json(a.out/'HISTORY.json',records)
 if not a.resume:commit()
 if cursor==len(jobs):
  if not (a.out/'FINAL.pt').exists():m.save(a.out/'FINAL.pt')
  print('ALREADY_COMPLETE_NO_PHYSICS',cursor,flush=True);return
 rt=FeedbackRuntime(a.out/'physics')
 # Restore bookkeeping; append raw files, never overwrite after restart.
 rt.rows=[r['row'] for r in records];rt.counter=max([0]+[int(f.name.split('_')[0]) for f in (a.out/'physics').glob('*.json.gz')])
 try:
  for j in range(cursor,len(jobs)):
   job=jobs[j];pending=a.out/f'PENDING_{j:05d}.pt'
   if pending.exists():
    item=torch.load(pending,weights_only=False);assert item['job']==job and item['manifest_sha']==sha(mf)
    assert sha(item['row']['path'])==item['row']['sha256']
    rt.rows.append(item['row'])
    m.rng.bit_generator.state=item['model_rng'];restore(item['global_rng'])
   else:
    # A leftover START means an interrupted physical attempt. Never silently
    # claim its cost zero or replay it as a successful completed episode.
    started=a.out/f'START_{j:05d}.json'
    if started.exists():raise RuntimeError('Uncommitted physical attempt: inspect START/raw costs before explicit recovery: '+str(started))
    atomic_json(started,dict(job=job,cursor=j,manifest_sha=sha(mf)))
    def actor(s):
     if job['kind']=='train':
      if m.rng.random()<.15:return int(m.rng.integers(3 if s[44]<.5 else 8))
      return m.choose(s,stochastic=True)
     return m.choose(s)
    policy=ContinuationOption(actor,reference.choose,'insertion_queue')
    row,ts,d=rt.run(job['seed'],policy,f"{a.arm}_{job['kind']}_{job['checkpoint']}")
    ts=policy.transitions(ts)
    for t in ts:t['continuation_sha']=sha(PARENT)
    item=dict(job=job,manifest_sha=sha(mf),row=row,transitions=ts,labels=completion_labels(d,ts),events=policy.events,model_rng=m.rng.bit_generator.state,global_rng=rng())
    atomic_torch(pending,item)
    if a.pause_after_collection is not None and job['kind']=='train' and job['checkpoint']+1==a.pause_after_collection:
     print('PAUSED_AFTER_DURABLE_COLLECTION',j,flush=True);return
   log=None
   if job['kind']=='train':
    data.extend(item['transitions']);labels.extend(item['labels'])
    if labels:log=m.train(data,labels,a.updates,success_bc=.2 if a.arm=='success_bc' else 0,successor_only=False)
    if job['checkpoint']+1 in marks:m.save(a.out/f"CHECKPOINT_{job['checkpoint']+1}.pt")
   records.append(dict(job=job,row=item['row'],effective_decisions=len(item['transitions']),updates=0 if log is None else a.updates,log=log))
   cursor=j+1;commit();print(a.arm,job,item['row']['success'],'updates',m.updates,'buffer',len(data),flush=True)
   if a.stop_after is not None and job['kind']=='train' and job['checkpoint']+1==a.stop_after:return
  m.save(a.out/'FINAL.pt')
 finally:rt.close()
if __name__=='__main__':main()
