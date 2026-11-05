"""One-deviation stopping-time PPO versus cost-sensitive supervised adoption.
Real terminal return relative to a fully reexecuted frozen-rule baseline.
"""
import argparse,json,hashlib,shutil,copy,gzip
from pathlib import Path
import numpy as np,torch
from torch import nn
from program_rule_option import Assets,SingleDeviation,gate_features,prefix_hash
from program_atomic_feedback import AtomicRuntime
from train_option_online import atomic_json,atomic_torch
class Gate(nn.Module):
 def __init__(self,mu,sd,seed):
  super().__init__();torch.manual_seed(seed);self.net=nn.Sequential(nn.Linear(107,32),nn.Tanh(),nn.Linear(32,1));nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-4.);self.register_buffer('mu',torch.tensor(mu,dtype=torch.float32));self.register_buffer('sd',torch.tensor(sd,dtype=torch.float32))
 def forward(self,x):return self.net((x-self.mu)/self.sd).squeeze(-1)
 def probs(self,x,mask,explore=False):
  logits=torch.cat([torch.zeros((len(x),1)),self(x.reshape(-1,107)).reshape(-1,2)],1).masked_fill(~mask,-1e9);p=logits.softmax(1)
  if explore:
   u=mask.float()*torch.tensor([.8,.1,.1]);p=.95*p+.05*u/u.sum(1,keepdim=True)
  return p

def supervise(model,opt,data,steps,rng):
 if not data:return 0
 for _ in range(steps):
  b=[data[i]for i in rng.integers(len(data),size=32)];x=torch.tensor(np.array([d['features']for d in b]),dtype=torch.float32);delta=torch.tensor([d['delta']for d in b],dtype=torch.float32);w=torch.where(delta<0,2.,torch.where(delta>0,1.,.1));loss=(w*nn.functional.binary_cross_entropy_with_logits(model(x),(delta>0).float(),reduction='none')).mean();opt.zero_grad();loss.backward();opt.step()
 return steps

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--baselines',type=Path,required=True);p.add_argument('--arm',choices=['supervised','stopping_rl','entry_rl'],required=True);p.add_argument('--seed',type=int,default=31801);p.add_argument('--episodes',type=int,default=64);p.add_argument('--train-start',type=int,default=763100);p.add_argument('--dev-start',type=int,default=763300);p.add_argument('--dev-count',type=int,default=16);p.add_argument('--resume',action='store_true');a=p.parse_args();torch.set_num_threads(2);a.out.mkdir(parents=True,exist_ok=True)
 pairs=json.loads(a.pairs.read_text());x=np.array([v['features']for v in pairs]);model=Gate(x.mean(0),np.maximum(x.std(0),.05),a.seed);opt=torch.optim.Adam(model.parameters(),lr=3e-4);rng=np.random.default_rng(a.seed)
 manifest={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()if k!='resume'};manifest.update(pair_sha=hashlib.sha256(a.pairs.read_bytes()).hexdigest(),native100=True,predecessor='shared frozen learned three-grasp actor31702, not retrained',return_signal='native candidate minus frozen-rule baseline; harms weighted2',policy='one complete Option; stopping likelihood includes wait decisions',source_sha={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()for f in ['train_rule_gate.py','program_rule_option.py']})
 own=[];history=[];updates=0;cursor=0
 if a.resume:
  assert json.loads((a.out/'MANIFEST.json').read_text())==manifest;d=torch.load(a.out/'CURRENT.pt',weights_only=False);model.load_state_dict(d['model']);opt.load_state_dict(d['optimizer']);rng.bit_generator.state=d['rng'];own=d['own'];history=d['history'];updates=d['updates'];cursor=d['cursor']
 else:
  assert not(a.out/'MANIFEST.json').exists();atomic_json(a.out/'MANIFEST.json',manifest)
  for f in manifest['source_sha']:shutil.copy2(Path(__file__).with_name(f),a.out/f)
  supervise(model,opt,pairs,500,rng);atomic_torch(a.out/'INITIAL.pt',dict(model=model.state_dict(),manifest=manifest));opt=torch.optim.Adam(model.parameters(),lr=3e-4)
 if not a.resume:atomic_torch(a.out/'CURRENT.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,own=own,history=history,updates=updates,cursor=0,manifest=manifest))
 assets=Assets();rt=AtomicRuntime(a.out/'physics');rt.rows=[v['row']for v in history];rt.counter=len(list((a.out/'physics').glob('*.json.gz')));jobs=[];marks=[0,a.episodes//2,a.episodes]
 for i in range(a.episodes+1):
  if i in marks:jobs +=[( 'eval',i,s)for s in range(a.dev_start,a.dev_start+a.dev_count)]
  if i<a.episodes:jobs.append(('train',i,a.train_start+i))
 try:
  for ji in range(cursor,len(jobs)):
   kind,i,seed=jobs[ji];trajectory=[];pending=a.out/f'PENDING_{ji:04d}.pt'
   if pending.exists():
    saved=torch.load(pending,weights_only=False);row,meta,trajectory=saved['row'],saved['option'],saved['trajectory'];rng.bit_generator.state=saved['rng']
   else:
    started=a.out/f'START_{ji:04d}.json';assert not started.exists(),'Inspect incomplete physical attempt before resuming';atomic_json(started,dict(job=jobs[ji]))
    def selector(s,ab,cs,step):
     if cs[0]==ab and cs[1]==ab:return None
     xx=np.stack([gate_features(s,ab,c)for c in cs]);mask=[True,cs[0]!=ab,cs[1]!=ab]
     with torch.no_grad():pr=model.probs(torch.tensor(xx[None]),torch.tensor([mask]),kind=='train')[0].numpy()
     act=int(rng.choice(3,p=pr/pr.sum()))if kind=='train'else int(pr.argmax());trajectory.append(dict(x=xx.tolist(),mask=mask,act=act,logp=float(np.log(max(pr[act],1e-12))),step=step));return act-1 if act else None
    pol=SingleDeviation(assets,selector,lambda:len(rt.env.records));row,ts,d=rt.run(seed,pol,f'{kind}_{i}');meta=pol.metadata();atomic_torch(pending,dict(row=row,option=meta,trajectory=trajectory,rng=rng.bit_generator.state))
   log=dict(kind=kind,checkpoint=i,row=row,option=meta,gate_decisions=trajectory,optimizer_updates=0)
   if kind=='train':
    bs=json.loads((a.baselines/'RESULTS.json').read_text());b=next(v for v in bs if v['seed']==seed);delta=int(row['success'])-int(b['success']);bd=json.load(gzip.open(b['path'],'rt'));actual=json.load(gzip.open(row['path'],'rt'));cut=meta['entry']['step']if meta['entry']else 100;assert prefix_hash(actual,cut)==prefix_hash(bd,cut),'own-policy baseline prefix mismatch';log.update(baseline=b,delta=delta,physical_prefix_hash=prefix_hash(actual,cut),prefix_steps=cut)
    if meta['entry']:own.append(dict(features=meta['entry']['features'],delta=delta,seed=seed,option=meta))
    if a.arm=='supervised':n=supervise(model,opt,own,8,rng)
    elif trajectory and delta:
     batch=trajectory if a.arm=='stopping_rl'else trajectory[-1:];xx=torch.tensor(np.array([v['x']for v in batch]),dtype=torch.float32);mask=torch.tensor([v['mask']for v in batch]);act=torch.tensor([v['act']for v in batch]);old=torch.tensor([v['logp']for v in batch]);adv=float(delta if delta>0 else 2*delta)
     for _ in range(8):
      pr=model.probs(xx,mask,True);ll=pr[torch.arange(len(act)),act].clamp_min(1e-12).log();ratio=(ll.sum()-old.sum()).clamp(-10,10).exp();loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
     n=8
    else:n=0
    updates+=n;log['optimizer_updates']=n
   history.append(log);atomic_json(a.out/'HISTORY.json',history);atomic_torch(a.out/'CURRENT.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,own=own,history=history,updates=updates,cursor=ji+1,manifest=manifest))
   if kind=='eval'and seed==a.dev_start+a.dev_count-1:atomic_torch(a.out/f'CHECKPOINT_{i}.pt',dict(model=model.state_dict(),manifest=manifest))
   print(a.arm,a.seed,kind,i,seed,row['success'],'entry',meta['entry']and meta['entry']['step'],'delta',log.get('delta'),flush=True)
 finally:rt.close()
if __name__=='__main__':main()
