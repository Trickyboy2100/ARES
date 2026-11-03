"""Finite physical Option learning; standard baselines, no novelty claim.
All targets enumerate legal physical IDs. Duration is retained; gamma=1 matches
native finite-horizon terminal-success objective, budget is observed. Online
updates mix prior and new transitions 50:50 (RLPD-inspired, not full RLPD).
"""
import copy,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_paired_skill_value import peg_features,load_peg

CONTROLLER='Peg100_GPU_OBB_speed2_original_three_refines_v1'
OFFSETS=(-.01,0.,.01)
def legal(x):
    m=torch.ones((len(x),3),dtype=torch.bool,device=x.device)
    m[:,2]=x[:,44]<.5
    return m
class Net(nn.Module):
    def __init__(self,out=3):
        super().__init__();self.net=nn.Sequential(nn.Linear(54,64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,64),nn.LayerNorm(64),nn.ReLU(),nn.Linear(64,out))
    def forward(self,x):return self.net(x)
def episode_transitions(d,offset,refresh,source):
    s=d['summary'];boundary=next((c['start'] for c in s['calls'] if c['phase']=='INSERT' and c['start']<100),None)
    x=peg_features(d['initial'],100,0);z=peg_features(d['steps'][-1],0,1);a=int(np.argmin(np.abs(np.array(OFFSETS)-offset)))
    y=float(s['success']);base=dict(scene=s['seed'],source=source,controller=CONTROLLER,outcome=y)
    if boundary is None:return [dict(base,s=x.tolist(),next=z.tolist(),a=a,r=y,done=True,duration=100,stage=0)]
    b=peg_features(d['steps'][boundary-1],100-boundary,1)
    return [dict(base,s=x.tolist(),next=b.tolist(),a=a,r=0.,done=False,duration=boundary,stage=0),dict(base,s=b.tolist(),next=z.tolist(),a=int(refresh),r=y,done=True,duration=100-boundary,stage=1)]
def prior(root):
    data=[];groups=load_peg(Path(root))
    for g in groups:
        for r,d,_ in g['items']:
            manifest=json.loads((Path(r['path']).parent/'MANIFEST.json').read_text())
            assert manifest['task']=='PegInsertionSide-v1' and manifest['horizon']==100 and manifest['speed']==2. and manifest['gpu_obb'] and not manifest['measured_insert'], 'INCOMPATIBLE_CONTROLLER_DATA'
            data+=episode_transitions(d,g['grasp'],bool(r['action'][0]),r['path'])
    return data,groups
class Learner:
    def __init__(self,kind='dsac',seed=31301,normalizer=None):
        torch.manual_seed(seed);self.rng=np.random.default_rng(seed);self.kind=kind;self.seed=seed;self.updates=0;self.gamma=1.;self.alpha=.02
        self.q=nn.ModuleList([Net(),Net()]);self.target=copy.deepcopy(self.q);self.actor=Net();self.v=Net(1);self.dyn=Net(54)
        self.qopt=torch.optim.Adam(self.q.parameters(),lr=3e-4);self.aopt=torch.optim.Adam(self.actor.parameters(),lr=3e-4);self.vopt=torch.optim.Adam(self.v.parameters(),lr=3e-4);self.dopt=torch.optim.Adam(self.dyn.parameters(),lr=3e-4)
        # normalizer frozen to prior; stage coordinate stays exactly binary.
        self.mu=np.zeros(54,np.float32) if normalizer is None else normalizer[0];self.sd=np.ones(54,np.float32) if normalizer is None else normalizer[1]
        self.history=[]
    def x(self,a):
        t=torch.as_tensor(np.asarray(a),dtype=torch.float32);mu=torch.as_tensor(self.mu);sd=torch.as_tensor(self.sd);return (t-mu)/sd
    def probs(self,x):return self.actor(x).masked_fill(~legal(x),-1e9).softmax(-1)
    def values(self,obs):
        with torch.no_grad():
            x=self.x(np.atleast_2d(obs));qs=torch.stack([q(x) for q in self.q]);qs=qs.masked_fill(~legal(x)[None],-1e9)
            if self.kind=='stap':
                # Functional Q-product analogue: separate predecessor reach and
                # actual successor-state model, exact finite enumeration.
                vals=nn.functional.logsigmoid(qs.mean(0));stage=x[:,44]<.5
                for a in range(3):
                    inp=x.clone();inp[:,-3:]=0;inp[:,-3+a]=1
                    z=self.dyn(inp);z[:,44]=1
                    suf=nn.functional.logsigmoid(torch.stack([q(z) for q in self.q]).mean(0)[:,:2]).max(1).values
                    vals[stage,a]+=suf[stage]
                return vals.numpy(),qs.numpy()
            if self.kind=='iql':return self.probs(x).numpy(),qs.numpy()
            if self.kind=='dsac':return self.probs(x).numpy(),qs.numpy()
            return qs.mean(0).numpy(),qs.numpy()
    def choose(self,obs,stochastic=False):
        vals,_=self.values(obs);v=vals[0];n=3 if obs[44]<.5 else 2
        if stochastic:
            p=v[:n] if self.kind in ('dsac','iql') else np.exp((v[:n]-max(v[:n]))/.1)
            p=p/p.sum();return int(self.rng.choice(n,p=p))
        return int(np.argmax(v[:n]))
    def train(self,offline,online=(),steps=100):
        start=time.monotonic();last={}
        for _ in range(steps):
            rows=[offline[i] for i in self.rng.integers(len(offline),size=32 if online else 64)]
            if online:rows +=[online[i] for i in self.rng.integers(len(online),size=32)]
            x=self.x([t['s'] for t in rows]);z=self.x([t['next']for t in rows]);a=torch.tensor([t['a']for t in rows]);r=torch.tensor([t['r']for t in rows]);done=torch.tensor([t['done']for t in rows]);dur=torch.tensor([t['duration']for t in rows]);y=torch.tensor([t['outcome']for t in rows]);idx=torch.arange(len(rows));mask=legal(x)
            with torch.no_grad():
                tq=torch.stack([q(z) for q in self.target]).min(0).values.masked_fill(~legal(z),-1e9)
                if self.kind=='dsac':
                    p=self.probs(z);nxt=(p*(tq-self.alpha*p.clamp_min(1e-9).log())).sum(1)
                elif self.kind=='iql':nxt=self.v(z)[:,0]
                else:nxt=tq.max(1).values
                target=r+(~done).float()*self.gamma**dur*nxt
                if self.kind=='supervised':target=y
                if self.kind=='stap':target=torch.where(x[:,44]<.5,(~done).float(),y)
            qs=[q(x)for q in self.q]
            if self.kind=='stap':ql=sum(nn.functional.binary_cross_entropy_with_logits(q[idx,a],target)for q in qs)/2
            else:ql=sum(((q[idx,a]-target)**2).mean()for q in qs)/2
            if self.kind=='cql':ql+=.1*sum((q.masked_fill(~mask,-1e9).logsumexp(1)-q[idx,a]).mean()for q in qs)/2
            self.qopt.zero_grad();ql.backward();self.qopt.step()
            al=torch.tensor(0.);vl=torch.tensor(0.);dl=torch.tensor(0.)
            if self.kind=='dsac':
                with torch.no_grad():av=torch.stack([q(x)for q in self.q]).min(0).values
                p=self.probs(x);al=(p*(self.alpha*p.clamp_min(1e-9).log()-av)).sum(1).mean();self.aopt.zero_grad();al.backward();self.aopt.step()
            if self.kind=='iql':
                with torch.no_grad():qdata=torch.stack([q(x)[idx,a]for q in self.target]).min(0).values
                diff=qdata-self.v(x)[:,0];vl=(torch.where(diff>0,.7,.3)*diff.square()).mean();self.vopt.zero_grad();vl.backward();self.vopt.step()
                with torch.no_grad():w=torch.exp(3*(qdata-self.v(x)[:,0])).clamp(max=100)
                al=-(w*self.probs(x)[idx,a].clamp_min(1e-9).log()).mean();self.aopt.zero_grad();al.backward();self.aopt.step()
            if self.kind=='stap' and (~done).any():
                inp=x[~done].clone();inp[:,-3:]=0;inp[torch.arange(len(inp)), -3+a[~done]]=1;dl=((self.dyn(inp)-z[~done])**2).mean();self.dopt.zero_grad();dl.backward();self.dopt.step()
            with torch.no_grad():
                for p,t in zip(self.q.parameters(),self.target.parameters()):t.lerp_(p,.005)
            self.updates+=1;last=dict(update=self.updates,q_loss=float(ql),actor_loss=float(al),v_loss=float(vl),dynamics_loss=float(dl),target_min=float(target.min()),target_max=float(target.max()),n_offline=len(offline),n_online=len(online))
        last['wall_s']=time.monotonic()-start;self.history.append(last);return last
    def save(self,path):
        torch.save(dict(kind=self.kind,seed=self.seed,updates=self.updates,q=self.q.state_dict(),target=self.target.state_dict(),actor=self.actor.state_dict(),v=self.v.state_dict(),dyn=self.dyn.state_dict(),qopt=self.qopt.state_dict(),aopt=self.aopt.state_dict(),vopt=self.vopt.state_dict(),dopt=self.dopt.state_dict(),mu=self.mu,sd=self.sd,rng=self.rng.bit_generator.state,history=self.history,controller=CONTROLLER),path)
    @classmethod
    def load(cls,path):
        d=torch.load(path,weights_only=False,map_location='cpu');assert d.get('controller')==CONTROLLER,'INCOMPATIBLE_MODEL_CONTRACT';m=cls(d['kind'],d['seed'],(d['mu'],d['sd']))
        for k in ['q','target','actor','v','dyn','qopt','aopt','vopt','dopt']:getattr(m,k).load_state_dict(d[k])
        m.updates=d['updates'];m.history=d['history'];m.rng.bit_generator.state=d['rng'];return m

def normalizer(data):
    x=np.array([t['s']for t in data]);mu=x.mean(0).astype(np.float32);sd=np.maximum(x.std(0),.05).astype(np.float32);mu[44]=0;sd[44]=1;return mu,sd
