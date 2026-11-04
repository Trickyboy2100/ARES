"""Controlled finite-Option value-improvement study, not a novelty assertion.
VIAC/USAC are finite-action functional adaptations. Empirical handoff targets
are compared to Q-graph-style bounds; no extra physical queries or observations.
"""
import copy,math,time,gzip,json,hashlib
import numpy as np
import torch
from program_discrete_options import Learner,legal,CONTROLLER

def utility(q,kappa):
    g=0. if kappa==0 else math.log(1/(1-kappa*kappa))/(math.sqrt(2)*kappa)
    return q.mean(0)+g*q.std(0,unbiased=False)

def state_key(s):
    # Exact observation and remaining budget; never merge nearby handoffs.
    return np.asarray(s,dtype=np.float64).tobytes()

def suffix_support(rows):
    support={}
    for t in rows:
        if t['stage']==1 and t['done']:
            support.setdefault(state_key(t['s']),{}).setdefault(t['a'],[]).append(t['r'])
    return {k:{a:float(np.mean(v)) for a,v in aa.items()} for k,aa in support.items()}

def verify_physical_support(rows):
    """A shared feature key must correspond to the identical full physical prefix.
    This prevents substituting labels across aliased privileged observations.
    Only raw simulation records are inspected; no extra physical query is used.
    """
    seen={};checked=0
    for t in rows:
        assert t['controller']in (CONTROLLER,CONTROLLER+'_split_refresh_option_v1'),'INCOMPATIBLE_PHYSICAL_CONTROLLER'
        if t['stage']==1 and t['a']==2:assert t['controller'].endswith('_split_refresh_option_v1')
        if t['stage']!=0 or t['done']:continue
        raw=json.load(gzip.open(t['source'],'rt'));duration=t['duration']
        payload=dict(initial=raw['initial'],steps=raw['steps'][:duration])
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        key=state_key(t['next'])
        if key in seen:assert seen[key]==digest,'ALIASED_HANDOFF_OBSERVATION'
        seen[key]=digest;checked+=1
    return dict(prefixes=checked,unique_handoffs=len(seen),exact_full_prefix=True)

class Improved(Learner):
    def __init__(self,kind='vi_mean',seed=31401,normalizer=None,successor_actions=2):
        super().__init__(kind,seed,normalizer,successor_actions=successor_actions)
    def values(self,obs):
        with torch.no_grad():
            x=self.x(np.atleast_2d(obs));q=torch.stack([m(x) for m in self.q]);q=q.masked_fill(~self.mask(x)[None],-1e9)
            if self.kind in ('sac_min','sac_mean','vi_expectile','usac','bee'):v=self.probs(x)
            else:v=q.mean(0)
            return v.numpy(),q.numpy()
    def choose(self,obs,stochastic=False):
        v,_=self.values(obs);n=3 if obs[44]<.5 else self.successor_actions
        if stochastic:
            p=v[0,:n] if self.kind in ('sac_min','sac_mean','vi_expectile','usac','bee') else np.exp((v[0,:n]-max(v[0,:n]))/self.alpha)
            return int(self.rng.choice(n,p=p/p.sum()))
        return int(v[0,:n].argmax())
    def train(self,offline,online=(),steps=100):
        start=time.monotonic();support=suffix_support(list(offline)+list(online));last={}
        for _ in range(steps):
            rows=[offline[i]for i in self.rng.integers(len(offline),size=32 if online else 64)]
            if online:rows +=[online[i]for i in self.rng.integers(len(online),size=32)]
            x=self.x([t['s']for t in rows]);z=self.x([t['next']for t in rows]);a=torch.tensor([t['a']for t in rows]);r=torch.tensor([t['r']for t in rows]);done=torch.tensor([t['done']for t in rows]);idx=torch.arange(len(rows));matched=0;delta=0.
            assert bool(self.mask(x)[idx,a].all()),'ACTION_OUTSIDE_MODEL_CONTRACT'
            with torch.no_grad():
                tq=torch.stack([q(z)for q in self.target]);mu=tq.mean(0)
                agg=tq.min(0).values if self.kind in ('sac_min','vi_expectile','bee') else utility(tq,-.831559) if self.kind=='usac' else mu
                agg=agg.masked_fill(~self.mask(z),-1e9)
                p=self.probs(z)
                if self.kind in ('sac_min','sac_mean','usac','vi_expectile','bee'):nxt=(p*(agg-self.alpha*p.clamp_min(1e-9).log())).sum(1)
                else:
                    nxt=self.alpha*torch.logsumexp(agg/self.alpha,1)
                    if self.kind in ('handoff','lower_bound'):
                        for i,t in enumerate(rows):
                            if t['done']:continue
                            known=support.get(state_key(t['next']),{})
                            if not known:continue
                            original=nxt[i].clone();v=agg[i,:self.successor_actions].clone()
                            if self.kind=='handoff':
                                # Replace only physically sampled terminal action values.
                                # No oracle labels, approximate-neighbor merge or reward shaping.
                                for act,y in known.items():v[act]=y
                                nxt[i]=self.alpha*torch.logsumexp(v/self.alpha,0)
                            else:
                                lb=self.alpha*torch.logsumexp(torch.tensor(list(known.values()))/self.alpha,0)
                                nxt[i]=torch.maximum(nxt[i],lb)
                            delta+=float(abs(nxt[i]-original));matched+=1
                if self.kind=='bee':nxt=.5*nxt+.5*self.v(z)[:,0]
                target=r+(~done).float()*nxt
            qs=[q(x)for q in self.q];errors=[target-q[idx,a]for q in qs]
            if self.kind=='vi_expectile':ql=sum((torch.where(e>0,.75,.25)*e.square()).mean()for e in errors)/2
            else:ql=sum(e.square().mean()for e in errors)/2
            self.qopt.zero_grad();ql.backward();self.qopt.step()
            vl=torch.tensor(0.)
            if self.kind=='bee':
                with torch.no_grad():
                    qdata=torch.stack([q(x)[idx,a]for q in self.target]).min(0).values
                    ap=self.probs(x);logp=ap.clamp_min(1e-9).log()
                # Exact finite expectation of BAC's sampled entropy/expectile loss.
                err=qdata[:,None]-(self.v(x)-self.alpha*logp)
                vl=(ap*torch.where(err>0,.7,.3)*err.square()).sum(1).mean()
                self.vopt.zero_grad();vl.backward();self.vopt.step()
            with torch.no_grad():
                aq=torch.stack([q(x)for q in self.q]);av=aq.min(0).values if self.kind in ('sac_min','vi_expectile','bee') else aq.mean(0)
            p=self.probs(x);al=(p*(self.alpha*p.clamp_min(1e-9).log()-av)).sum(1).mean();self.aopt.zero_grad();al.backward();self.aopt.step()
            with torch.no_grad():
                for q,t in zip(self.q.parameters(),self.target.parameters()):t.lerp_(q,.005)
            self.updates+=1
            last=dict(update=self.updates,q_loss=float(ql),actor_loss=float(al),v_loss=float(vl),support_matched=matched,target_delta_mean=delta/max(matched,1),target_min=float(target.min()),target_max=float(target.max()),n_offline=len(offline),n_online=len(online))
        last['wall_s']=time.monotonic()-start;self.history.append(last);return last
