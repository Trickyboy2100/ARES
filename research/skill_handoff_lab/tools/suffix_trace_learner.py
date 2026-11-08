"""Strong-checkpoint continuation study; Retrace is a mature baseline.
The lagged arm tests reference-versioned evaluation, NOT a novelty claim.
"""
import copy,time
import numpy as np
import torch
from torch import nn
from train_full_option import Learner as Base

def backup(rewards, values, taken_q, coefficients, done):
    """Q return: continuation correction starts at the NEXT actual Option."""
    out=[0.]*len(rewards)
    for j in reversed(range(len(out))):
        out[j]=float(rewards[j])+(0. if done[j] else float(values[j]))
        if j+1<len(out) and not done[j]:
            out[j]+=float(coefficients[j+1])*(out[j+1]-float(taken_q[j+1]))
    return out

class Learner(Base):
    def __init__(self,assets,seed,mode):
        super().__init__(assets,seed,'sac');self.mode=mode
        self.reference=copy.deepcopy(self.actor);self.version=-1;self.behavior=[]
    def refresh(self,episode):
        if self.version<0 or (self.mode=='lagged' and episode//8!=self.version//8):
            self.reference.load_state_dict(self.actor.state_dict());self.version=episode
    def choose(self,s,train=False):
        b,mask=self.context([s])
        with torch.no_grad():p=self.probs(self.norm([s]),b,mask,[s])[0].numpy()
        # Same strong inherited policy from the first online interaction: no warm reset.
        if s[44]<.5:
            a=self.warm.choose(s);mu=1.
        elif train:
            p=.9*p+.1*mask[0].numpy()/float(mask[0].sum())
            a=int(self.rng.choice(13,p=p/p.sum()));mu=float(p[a]/p.sum())
        else:a=int(p.argmax());mu=1.
        self.behavior.append(dict(a=a,mu=mu));return a
    def evaluation_probs(self,ss):
        b,m=self.context(ss);x=self.norm(ss)
        if self.mode!='lagged':return self.probs(x,b,m,ss)
        actor=self.actor
        try:self.actor=self.reference;return self.probs(x,b,m,ss)
        finally:self.actor=actor
    def targets(self,data,indices):
        chains=[]
        for i in indices:
            chain=[i]
            if self.mode!='sac':
                while len(chain)<8 and not data[chain[-1]]['done']:
                    ni=data[chain[-1]].get('successor_index')
                    if ni is None or 'mu' not in data[ni]:break
                    chain.append(ni)
            chains.append(chain)
        flat=sorted(set(i for c in chains for i in c));lookup={i:j for j,i in enumerate(flat)}
        rows=[data[i]for i in flat];ss=[t['s']for t in rows];ns=[t['next']for t in rows]
        with torch.no_grad():
            pn=self.evaluation_probs(ns);ps=self.evaluation_probs(ss)
            qn=torch.stack([v(self.norm(ns))for v in self.target]).mean(0)
            qs=torch.stack([v(self.norm(ss))for v in self.target]).mean(0)
            values=(pn*(qn-.01*pn.clamp_min(1e-9).log())).sum(1).tolist()
            taken=qs[torch.arange(len(rows)),torch.tensor([t['a']for t in rows])].tolist()
            prob=ps[torch.arange(len(rows)),torch.tensor([t['a']for t in rows])].tolist()
        ys=[];cs=[]
        for chain in chains:
            rr=[data[i]for i in chain];ii=[lookup[i]for i in chain]
            coeff=[.9*min(1.,prob[j]/r['mu'])if 'mu'in r else 0. for j,r in zip(ii,rr)]
            ys.append(backup([r['r']for r in rr],[values[j]for j in ii],[taken[j]for j in ii],coeff,[r['done']for r in rr])[0]);cs.extend(coeff[1:])
        return torch.tensor(ys),dict(mean_trace_length=float(np.mean([len(c)for c in chains])),mean_coefficient=float(np.mean(cs))if cs else 0.)
    def train(self,data,steps,episode):
        self.refresh(episode);tic=time.monotonic();logs=[]
        old=[i for i,t in enumerate(data)if 'mu'not in t];own=[i for i,t in enumerate(data)if 'mu'in t]
        for _ in range(steps):
            # Equal replay mixture in every arm, including SAC.
            indices=[(old or own)[i]for i in self.rng.integers(len(old or own),size=32)]+[own[i]for i in self.rng.integers(len(own),size=32)]
            rows=[data[i]for i in indices];ss=[r['s']for r in rows];x=self.norm(ss);a=torch.tensor([r['a']for r in rows]);ix=torch.arange(len(rows));b,mask=self.context(ss)
            y,diag=self.targets(data,indices)
            qloss=sum((q(x)[ix,a]-y).square().mean()for q in self.q)/2
            self.qo.zero_grad();qloss.backward();nn.utils.clip_grad_norm_(self.q.parameters(),10);self.qo.step()
            p=self.probs(x,b,mask,ss)
            with torch.no_grad():q=torch.stack([v(x)for v in self.q]).mean(0)
            aloss=(p*(.01*p.clamp_min(1e-9).log()-q)).sum(1).mean()
            w=torch.tensor([r['outcome']for r in rows]);aloss-=.1*(w*p[ix,a].clamp_min(1e-9).log()).sum()/w.sum().clamp_min(1)
            self.ao.zero_grad();aloss.backward();nn.utils.clip_grad_norm_(self.actor.parameters(),1);self.ao.step()
            with torch.no_grad():
                for q,t in zip(self.q.parameters(),self.target.parameters()):t.lerp_(q,.01)
            self.updates+=1;logs.append(dict(q=float(qloss),actor=float(aloss),target_mean=float(y.mean()),**diag))
        return dict(updates=steps,wall_s=time.monotonic()-tic,last=logs[-1],reference_version=self.version)
    def state(self):
        d=super().state();d.update(mode=self.mode,reference=self.reference.state_dict(),version=self.version);return d
    def load(self,d):
        super().load(d)
        self.reference.load_state_dict(d.get('reference',d['actor']));self.version=d.get('version',-1)
