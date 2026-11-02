"""Finite-action two-boundary Fitted Q and strong nearest baselines.
All transitions are measured physical SkillResult; no synthetic success labels.
STAP-style baseline exhaustively enumerates nine Q0*Q1(D) sequences, in log space.
This is a diagnostic baseline study, not an independent RL contribution.
"""
import argparse,gzip,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_stap_learning import Net,features
GRASPS=[.035,.07,.105];MODES=['baseline','frozen12','slow3']
def dataset(root):
 rows=[]
 for r in json.loads((root/'RESULTS.json').read_text()):
  p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(p,'rt'));dec=d['decisions'];x=features(dec[0]['observation'],100,r['mass'],0,0);valid=len(dec)>1
  z=features(dec[1]['observation'],100-dec[1]['boundary_step'],r['mass'],0,1) if valid else np.zeros(45,np.float32)
  rows.append(dict(seed=r['seed'],mass=r['mass'],g=GRASPS.index(r['grasp']),a=MODES.index(r['mode']),x=x,z=z,valid=valid,y=float(r['success']),path=str(p),sha256=r['sha256']))
 return rows
class Learned:
 def __init__(self,path):
  d=torch.load(path,map_location='cpu',weights_only=False);self.payload=d;self.mean=torch.tensor(d['mean']);self.std=torch.tensor(d['std']);self.nets={}
  for k,w in d['weights'].items():n=Net(46,45 if k=='dynamics' else 1);n.load_state_dict(w);n.eval();self.nets[k]=n
 def call(self,name,s,acts):
  x=(torch.as_tensor(s,dtype=torch.float32)-self.mean)/self.std;x=x.reshape(1,45).repeat(len(acts),1);a=torch.tensor(acts,dtype=torch.float32).reshape(-1,1);return self.nets[name](torch.cat([x,a],-1))
 def choose(self,method,s,stage,planned=None):
  with torch.no_grad():
   if stage==1:
    if method=='stap_open':return planned,{}
    if method=='foresight':return 0,{}
    v=self.call('q1',s,[-1,0,1])[:,0];return int(v.argmax()),dict(logits=v.tolist())
   if method.startswith('stap'):
    q0=torch.nn.functional.logsigmoid(self.call('q0',s,[-1,0,1])[:,0]);pred=self.call('dynamics',s,[-1,0,1])*self.std+torch.tensor(s);pred[:,42:44]=torch.tensor(s)[42:44];pred[:,44]=1
    vals=[]
    for j in range(3):vals.append(q0[j]+torch.nn.functional.logsigmoid(self.call('q1',pred[j],[-1,0,1])[:,0]))
    table=torch.stack(vals);idx=int(table.reshape(-1).argmax());return idx//3,dict(planned_mode=idx%3,log_sequence_values=table.tolist(),queries=12)
   v=self.call(method,s,[-1,0,1])[:,0];return int(v.argmax()),dict(logits=v.tolist(),queries=3)

def train(a):
 a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(31132);rows=dataset(a.data);train=[r for r in rows if r['seed']<745014];cal=[r for r in rows if r['seed']>=745014];assert len(train)==72 and len(cal)==36
 x=torch.tensor(np.array([r['x'] for r in train]));z=torch.tensor(np.array([r['z'] for r in train]));valid=torch.tensor([r['valid'] for r in train]);y=torch.tensor([r['y'] for r in train]);ga=torch.tensor([r['g']-1 for r in train]).reshape(-1,1);sa=torch.tensor([r['a']-1 for r in train]).reshape(-1,1)
 both=torch.cat([x,z[valid]]);mean=both.mean(0);std=both.std(0).clamp_min(.02);xx=torch.cat([(x-mean)/std,ga],1);zz=torch.cat([(z-mean)/std,sa],1);nets={};curves={};counts={};start=time.monotonic()
 def fit(name,inputs,targets,n=2000,regression=False):
  torch.manual_seed(31132);net=Net(46,targets.shape[1] if targets.ndim==2 else 1);opt=torch.optim.Adam(net.parameters(),lr=3e-4);curve=[]
  for k in range(n):
   pred=net(inputs);pred=pred if targets.ndim==2 else pred[:,0];loss=((pred-targets)**2).mean() if regression else nn.functional.binary_cross_entropy_with_logits(pred,targets)
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
   if k%100==0 or k==n-1:curve.append(dict(update=k+1,loss=float(loss)))
  nets[name]=net;curves[name]=curve;counts[name]=n
 fit('q1',zz[valid],y[valid]);fit('q0',xx,valid.float(),1000);fit('dynamics',xx[valid],((z-x)/std)[valid],regression=True)
 with torch.no_grad():
  qnext=torch.stack([torch.sigmoid(nets['q1'](torch.cat([(z-mean)/std,torch.full((len(z),1),float(k))],1))[:,0]) for k in [-1,0,1]],1).max(1).values*valid.float()
 fit('ordinary',xx,qnext);fit('supervised',xx,y);mask=torch.tensor([r['a']==0 for r in train]);fit('foresight',xx[mask],y[mask])
 payload=dict(mean=mean.tolist(),std=std.tolist(),weights={k:n.state_dict() for k,n in nets.items()},seed=31132,train_scenes=sorted({r['seed'] for r in train}),cal_scenes=sorted({r['seed'] for r in cal}),action_grasps=GRASPS,action_modes=MODES,algorithm='two-step Fitted Q gamma1; terminal BCE; predecessor bootstraps max successor value on actual transition; standard not novel')
 torch.save(payload,a.out/'MODEL.pt');m=Learned(a.out/'MODEL.pt');scores={}
 for method in ['ordinary','supervised','stap_open','stap_replan','foresight']:
  result=[]
  for seed,mass in sorted({(r['seed'],r['mass']) for r in cal}):
   rr=[r for r in cal if (r['seed'],r['mass'])==(seed,mass)];g,log=m.choose(method,rr[0]['x'],0);r=next(r for r in rr if r['g']==g);act,_=m.choose(method,r['z'],1,log.get('planned_mode')) if r['valid'] else (0,{})
   actual=next(r for r in rr if (r['g'],r['a'])==(g,act));result.append(dict(seed=seed,mass=mass,g=g,a=act,success=actual['y']))
  scores[method]=result
 with torch.no_grad():
  probabilities=[]
  for r in cal:
   if r['valid']:probabilities.append((float(torch.sigmoid(m.call('q1',r['z'],[r['a']-1])[0,0])),r['y']))
  brier=float(np.mean([(p-y)**2 for p,y in probabilities]))
 (a.out/'TRAINING.json').write_text(json.dumps(dict(train_episodes=len(train),calibration_episodes=len(cal),optimizer_updates=counts,wall_s=time.monotonic()-start,curves=curves,calibration_q1_Brier=brier,calibration_policy_lookup=scores,calibration_scope='actual measured finite grid policy lookup; not a new simulation or sealed test',sources=[{k:r[k] for k in ['seed','mass','path','sha256']} for r in rows],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));print(json.dumps(dict(updates=counts,calibration={k:sum(r['success'] for r in v) for k,v in scores.items()},brier=brier)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);train(p.parse_args())
