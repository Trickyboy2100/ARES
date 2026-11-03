"""Hypothesis: acquire full-prefix paired data by expected predecessor decision change.
Not a new critic/loss. Virtual two-step SGD is acquisition compute only. No oracle
outcomes, held-out states, restored physics or extra rollout permissions.
"""
import copy,time
import numpy as np,torch
from torch.func import functional_call

def update_score(model,record,anchors,propagate=True,joint=None):
    """Expected policy regret after one hypothetical paired observation/update.
    Four terminal label possibilities, plug-in independent Bernoulli weights;
    not a calibrated posterior. Include this approximation in the claim boundary.
    """
    x=model.x([record['prefix']['s']]);b=model.x([record['suffix']['s']]);a0=record['prefix']['a'];ref=model.x(anchors)
    qbase=model.q[0];base=dict(qbase.named_parameters());fixed=dict(qbase.named_buffers());base={k:v.detach().clone().requires_grad_(True)for k,v in base.items()}
    def call(params,z):return functional_call(qbase,(params,fixed),(z,))
    with torch.no_grad():
        means=torch.stack([q(b)for q in model.q]).mean(0)[0,:2].clamp(.02,.98).numpy();incumbent=call(base,ref).argmax(1)
    total=0.;branches=[]
    for y0 in [0.,1.]:
        for y1 in [0.,1.]:
            ys=torch.tensor([[y0,y1]]);loss=(call(base,b)[:,:2]-ys).square().mean();grads=torch.autograd.grad(loss,tuple(base.values()));p1={k:p-.01*g for (k,p),g in zip(base.items(),grads)}
            target=call(p1,b)[0,:2].max().detach()
            if propagate:loss2=(call(p1,x)[0,a0]-target).square()
            else:loss2=(call(p1,b)[:,:2]-ys).square().mean()
            grads2=torch.autograd.grad(loss2,tuple(p1.values()));p2={k:p-.01*g for (k,p),g in zip(p1.items(),grads2)}
            with torch.no_grad():
                vals=call(p2,ref);gain=(vals.max(1).values-vals[torch.arange(len(ref)),incumbent]).mean().item();switch=(vals.argmax(1)!=incumbent).float().mean().item()
            prob=(means[0] if y0 else 1-means[0])*(means[1] if y1 else 1-means[1])if joint is None else joint[int(y0),int(y1)];total+=prob*gain;branches.append(dict(labels=[y0,y1],weight=float(prob),gain=gain,switch=switch))
    return float(total),branches

def choose_record(model,pool,mode,rng,anchors,history):
    start=time.monotonic();scores=[];details=[]
    for i,r in enumerate(pool):
        b=r['suffix']['s'];_,qs=model.values(b);q=qs[:,0,:2];mu=np.clip(q.mean(0),0,1)
        if mode=='random_pair':score=float(rng.random());detail=None
        elif mode=='uq_pair':score=float(q.std(0).mean());detail=None
        elif mode=='ees_pair':
            # Optimistic competence extrapolation inspired by EES. Estimate
            # per physical prefix using observed outcomes only, extrapolate the
            # largest cycle change, situate in current predecessor plan value.
            h=history.get(i,[]);competence=(1+sum(h))/(2+len(h));cycles=[np.mean(h[j:j+2])for j in range(0,len(h),2)]
            gain=(max(cycles)-min(cycles)) if len(cycles)>1 else .1
            extrap=min(1.,competence+gain);initial=r['prefix']['s'];_,q0=model.values(initial);v=q0[:,0,:3].mean(0);a=r['prefix']['a'];score=float(max(0.,v[a]+extrap-competence-max(v)))+.01/(1+len(h));detail=dict(competence=competence,extrapolated=extrap,cycles=cycles)
        else:score,detail=update_score(model,r,anchors,mode!='local_update')
        scores.append(score);details.append(detail)
    chosen=int(np.argmax(scores));return chosen,dict(mode=mode,scores=scores,chosen=chosen,details=details,wall_s=time.monotonic()-start,queries='virtual critic updates only',physical_pair_cost=200)

def frontier_score(model,record,anchors):
    """Expected initial-decision change for an as-yet unexecuted first option.
    Hypothetical paired successor returns affect Q0 through their maximum; an
    explicit approximation until a real handoff is observed, no learned dynamics.
    """
    x=model.x([record['prefix']['s']]);ref=model.x(anchors);a=record['grasp'];net=model.q[0];params={k:v.detach().clone().requires_grad_(True)for k,v in net.named_parameters()};buffers=dict(net.named_buffers())
    def call(p,z):return functional_call(net,(p,buffers),(z,))
    with torch.no_grad():
        p=float(torch.stack([q(x)[0,a]for q in model.q]).mean().clamp(.02,.98));inc=call(params,ref).argmax(1)
    score=0.;branches=[]
    for y in [0.,1.]:
        loss=(call(params,x)[0,a]-y).square();grad=torch.autograd.grad(loss,tuple(params.values()));updated={k:v-.01*g for (k,v),g in zip(params.items(),grad)}
        with torch.no_grad():
            vals=call(updated,ref);gain=float((vals.max(1).values-vals[torch.arange(len(ref)),inc]).mean())
        weight=p if y else 1-p;score+=weight*gain;branches.append(dict(label=y,weight=weight,gain=gain))
    return score,branches

class JointOutcomeKernel:
    """Frozen nonparametric paired-label model; same-marginal independence control.
    Five neighbors and Dirichlet .25 fixed before new physical acquisition.
    No neural auxiliary head, no extra data or reward changes.
    """
    def __init__(self,groups):
        self.x=[];self.y=[]
        for g in groups:
            if len(g['items'][0][2])!=2:continue
            self.x.append(g['items'][0][2][1][0]);self.y.append(tuple(int(r['success'])for r,d,t in g['items']))
        self.x=np.array(self.x);self.sd=np.maximum(self.x.std(0),.05)
    def predict(self,obs,independent=False):
        ids=np.argsort(np.mean(((self.x-obs)/self.sd)**2,axis=1))[:5];counts=np.ones((2,2))*.25
        for i in ids:counts[self.y[i]]+=1
        pp=counts/counts.sum()
        return pp.sum(1)[:,None]*pp.sum(0)[None,:]if independent else pp
