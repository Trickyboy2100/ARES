"""Same-physical-data supervised and STAP-core comparators for contact authority.
Not official STAP benchmark reproduction. Log-Q CEM core inherited unchanged.
"""
import argparse,json,gzip,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from program_contact_direction import transitions
from program_stap_kernel import Agent,Box,Critic,Primitive,port_plan
class Net(nn.Module):
 def __init__(self,n,out):super().__init__();self.seq=nn.Sequential(nn.Linear(n,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,out))
 def forward(self,x):return self.seq(x)
class Models:
 def __init__(self,p):
  self.p=p;self.mean=torch.tensor(p['mean']);self.std=torch.tensor(p['std']);self.nets={k:Net(56,54 if k=='dynamics'else 1)for k in p['weights']}
  for k,n in self.nets.items():n.load_state_dict(p['weights'][k]);n.eval()
 def value(self,k,s,a):return self.nets[k](torch.cat([(s-self.mean)/self.std,a],-1))
 def q(self,k,s,a):return torch.sigmoid(self.value('q'+str(k),s,a)[:,0])
 def next(self,s,a):
  z=s+self.value('dynamics',s,a*torch.tensor([1.,0.]))*self.std;z[:,44]=1.;return z
class Prior:
 def __init__(self,stage):self.stage=stage
 def predict(self,s):return torch.zeros((len(s),2)) # neutral, broad CEM covariance explores full authority
class Dynamics:
 def __init__(self,m):self.m=m;self.action_space=Box([-1,-1],[1,1]);self.device=torch.device('cpu')
 def decode(self,s,primitive=None):return s
 def rollout(self,obs,skeleton,policies,batch_size=None,time_index=False):
  n=batch_size or 1;s=obs.unsqueeze(0).repeat(n,1);ss=[s];aa=[]
  for t,p in enumerate(skeleton):
   a=policies[t if time_index else p.idx_policy].actor.predict(s);aa.append(a);s=self.m.next(s,a)if p.idx_policy==0 else s.clone();ss.append(s)
  ss=torch.stack(ss,1);aa=torch.stack(aa,1);return(ss,aa)if batch_size else(ss[0],aa[0])
class Selector:
 def __init__(self,path,kind):self.m=Models(torch.load(path,map_location='cpu',weights_only=False));self.kind=kind;self.logs=[]
 def choose(self,obs):
  m=self.m;s=torch.tensor(obs);stage=int(obs[44]>.5);t=time.monotonic()
  with torch.no_grad():
   if self.kind=='stap':
    agents=[Agent(Box([-1,-1],[1,1]),Prior(k),Critic(lambda s,a,k=k:m.q(k,s,a*torch.tensor([1.,float(k)]))))for k in [0,1]]
    plan=port_plan(obs,[Primitive(k)for k in range(stage,2)],agents,Dynamics(m),num_iterations=4,num_samples=32,num_elites=6,keep_elites_fraction=.25,population_decay=.8);a=plan['actions'][0];log={k:plan[k]for k in ['p_success','trajectory_queries','skill_model_queries','iterations']}
   else:
    grid=torch.linspace(-1,1,9);a0=torch.stack([grid,torch.zeros_like(grid)],1);a1=torch.cartesian_prod(grid,grid);actions=a0 if stage==0 else a1;values=m.value('direct0'if stage==0 else 'q1',s.repeat(len(actions),1),actions)[:,0];i=values.argmax();a=actions[i].numpy();log=dict(predicted_success=float(values[i].sigmoid()),skill_model_queries=len(actions))
  a=np.array(a).copy()
  if stage==0:a[1]=0
  self.logs.append(dict(stage=stage,action=a.tolist(),wall_s=time.monotonic()-t,**log));return a

def train(effect,out,updates):
 torch.set_num_threads(4);torch.manual_seed(31171);np.random.seed(31171);out.mkdir(parents=True,exist_ok=False);e=json.loads(effect.read_text());rows=[]
 for r in e['dataset']:
  d=json.load(gzip.open(r['path'],'rt'));ts=transitions(d);rows.append((ts,float(d['summary']['success'])))
 allobs=np.array([t[0]for ts,_ in rows for t in ts]);mean=allobs.mean(0);std=np.maximum(allobs.std(0),.05);s0=torch.tensor(np.array([ts[0][0]for ts,_ in rows]));a0=torch.tensor(np.array([ts[0][2]for ts,_ in rows]),dtype=torch.float32);reached=torch.tensor([[len(ts)==2]for ts,_ in rows],dtype=torch.float32);end=torch.tensor([[v]for _,v in rows]);valid=[ts for ts,_ in rows if len(ts)==2];s1=torch.tensor(np.array([ts[1][0]for ts in valid]));a1=torch.tensor(np.array([ts[1][2]for ts in valid]),dtype=torch.float32);r1=torch.tensor([[ts[1][3]]for ts in valid]);src=torch.tensor(np.array([ts[0][0]for ts in valid]));ua=torch.tensor(np.array([ts[0][2]for ts in valid]),dtype=torch.float32);mu=torch.tensor(mean);sd=torch.tensor(std);nets={k:Net(56,54 if k=='dynamics'else 1)for k in ['q0','q1','direct0','dynamics']};opt=torch.optim.Adam([p for n in nets.values()for p in n.parameters()],lr=3e-4);logs=[];start=time.monotonic();bce=nn.BCEWithLogitsLoss()
 for i in range(updates):
  v=lambda k,s,a:nets[k](torch.cat([(s-mu)/sd,a],-1));loss=bce(v('q0',s0,a0),reached)+bce(v('q1',s1,a1),r1)+bce(v('direct0',s0,a0),end)+.1*((v('dynamics',src,ua)-(s1-src)/sd)**2).mean();opt.zero_grad();loss.backward();opt.step()
  if i%100==0:logs.append(dict(update=i,loss=float(loss)))
 torch.save(dict(mean=mean.tolist(),std=std.tolist(),weights={k:n.state_dict()for k,n in nets.items()}),out/'MODEL.pt');(out/'TRAINING.json').write_text(json.dumps(dict(updates=updates,optimizer_steps=updates,neural_components=4,episodes=len(rows),actual_postcontact=len(valid),wall_s=time.monotonic()-start,logs=logs,scope='same-data bounded functional STAP core and direct supervised score; not official checkpoints or convergence claim',q0_target='actual postcontact boundary reached',q1_target='native terminal success'),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--effect',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=1000);a=p.parse_args();train(a.effect,a.out,a.updates)
