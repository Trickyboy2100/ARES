"""Same physical dataset: STAP-style model/Q, supervised and downstream value controls.
Not official full benchmark reproductions; query and training costs explicit.
"""
import argparse,json,gzip,hashlib,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from program_stap_learning import features
class Net(nn.Module):
 def __init__(self,n,out=1):
  super().__init__();self.body=nn.Sequential(nn.Linear(n,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,128),nn.LayerNorm(128),nn.ReLU(),nn.Linear(128,out))
 def forward(self,x):return self.body(x)


def data(roots):
 rows=[]
 for root in roots:
  for r in json.loads((root/'RESULTS.json').read_text()):
   p=Path(r['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];d=json.load(gzip.open(p,'rt'));ds=d['decisions'];acts=d['policy_actions'];x=features(ds[0]['observation'],100,r['mass'],0,0);valid=len(acts)>1;z=features(ds[1]['observation'],100-ds[1]['boundary_step'],r['mass'],0,1) if valid else np.zeros(45,np.float32)
   rows.append(dict(x=x,z=z,valid=valid,g=acts[0]['normalized'],h=acts[1]['normalized'] if valid else 0.,y=float(r['success']),seed=r['seed'],path=str(p),sha256=r['sha256']))
 return rows
class Policies:
 def __init__(self,path):
  self.d=torch.load(path,map_location='cpu',weights_only=False);self.mean=torch.tensor(self.d['mean']);self.std=torch.tensor(self.d['std']);self.nets={};self.grid=torch.linspace(-1,1,11)
  for k,w in self.d['weights'].items():n=Net(46,45 if k=='dynamics' else 1);n.load_state_dict(w);n.eval();self.nets[k]=n
 def call(self,name,s,actions):
  x=(torch.as_tensor(s,dtype=torch.float32)-self.mean)/self.std;return self.nets[name](torch.cat([x.reshape(1,-1).repeat(len(actions),1),torch.as_tensor(actions).float().reshape(-1,1)],1))
 def choose(self,method,s,stage,planned=None):
  with torch.no_grad():
   if stage:
    if method=='stap_open':return planned,dict(queries=0)
    if method=='downstream':return -1.,dict(queries=0)
    v=self.call('q1',s,self.grid)[:,0];idx=int(v.argmax());return float(self.grid[idx]),dict(logits=v.tolist(),queries=11)
   if method.startswith('stap'):
    log0=nn.functional.logsigmoid(self.call('q0',s,self.grid)[:,0]);z=self.call('dynamics',s,self.grid)*self.std+torch.as_tensor(s);z[:,42:44]=torch.as_tensor(s)[42:44];z[:,44]=1;values=torch.stack([log0[i]+nn.functional.logsigmoid(self.call('q1',zz,self.grid)[:,0])for i,zz in enumerate(z)]);idx=int(values.reshape(-1).argmax());return float(self.grid[idx//11]),dict(planned=float(self.grid[idx%11]),log_values=values.tolist(),queries=132)
   v=self.call('downstream' if method=='downstream' else 'supervised',s,self.grid)[:,0];i=int(v.argmax());return float(self.grid[i]),dict(logits=v.tolist(),queries=11)

def main():
 p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);rows=data(a.roots);x=torch.tensor(np.array([r['x']for r in rows]));z=torch.tensor(np.array([r['z']for r in rows]));valid=torch.tensor([r['valid']for r in rows]);y=torch.tensor([r['y']for r in rows]);g=torch.tensor([r['g']for r in rows]);h=torch.tensor([r['h']for r in rows]);both=torch.cat([x,z[valid]]);mean=both.mean(0);std=both.std(0).clamp_min(.02);xx=torch.cat([(x-mean)/std,g[:,None]],1);zz=torch.cat([(z-mean)/std,h[:,None]],1);nets={};curves={};t=time.monotonic()
 def fit(name,inputs,targets,reg=False):
  torch.manual_seed(31142);n=Net(46,45 if reg else 1);optim=torch.optim.Adam(n.parameters(),lr=3e-4);curve=[]
  for k in range(2000):
   idx=torch.randint(len(inputs),(min(128,len(inputs)),));v=n(inputs[idx]);v=v if reg else v[:,0];loss=(v-targets[idx]).square().mean() if reg else nn.functional.binary_cross_entropy_with_logits(v,targets[idx]);optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(n.parameters(),5);optim.step()
   if k%100==0 or k==1999:curve.append(dict(update=k+1,loss=float(loss)))
  nets[name]=n;curves[name]=curve
 fit('q0',xx,valid.float());fit('q1',zz[valid],y[valid]);fit('dynamics',xx[valid],((z-x)/std)[valid],True);fit('supervised',xx,y)
 # Fixed-successor value analogue uses the Q prediction at nominal26mm, bootstrapped at real SkillResult.
 with torch.no_grad():value=torch.sigmoid(nets['q1'](torch.cat([(z-mean)/std,torch.full((len(z),1),-1.)],1))[:,0])*valid.float()
 fit('downstream',xx,value)
 torch.save(dict(mean=mean.tolist(),std=std.tolist(),weights={k:n.state_dict()for k,n in nets.items()}),a.out/'MODEL.pt');(a.out/'TRAINING.json').write_text(json.dumps(dict(episodes=len(rows),scenes=len({r['seed']for r in rows}),optimizer_updates=10000,seed=31142,wall_s=time.monotonic()-t,curves=curves,sources=[{k:r[k]for k in ['path','sha256','seed']}for r in rows],downstream_scope='learned nominal-successor value on actual boundary, not official ViSkill/SeqDex/Foresight reproduction'),indent=2))
if __name__=='__main__':main()
