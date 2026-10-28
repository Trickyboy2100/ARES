"""Two physically executed option decisions: matched PPO and paired-risk PPO pilot.

No simulator outcome cache or scene setters. Every online episode runs native250
steps. Candidate adds a frozen training-only paired-regression risk regularizer;
this is a conventional heuristic ablation, not a claim of a new RL theorem.
"""
import argparse,copy,json,time,gzip
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from handoff_vector import VectorHandoff
from handoff_physics import sha

torch.set_num_threads(4)
class Head(nn.Module):
    def __init__(self,out):
        super().__init__();self.net=nn.Sequential(nn.Linear(57,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,out))
    def forward(self,x):return self.net(x)
class ActorCritic(nn.Module):
    def __init__(self):super().__init__();self.actor=Head(3);self.critic=Head(1)
    def forward(self,x):return self.actor(x),self.critic(x).squeeze(-1)

def transform(obs,mode,initial=None):
    x=np.array(obs,dtype=np.float32,copy=True)
    if mode=='no_next' and x[-1]==0:
        # Remove ALL representations containing next object C, including qC and
        # current TCP-C, rather than just a redundant derived contract scalar.
        x[3:9]=0;x[17:21]=0;x[45:48]=0
    if mode=='no_posterior' and x[-1]==1:
        assert initial is not None
        # Open-loop snapshot replaces object geometry and effect channels. Live
        # proprioception, stage and remaining budget remain available.
        x[:21]=initial[:21];x[39:55]=initial[39:55];x[39:48]=0
    if mode=='no_second' and x[-1]==1:
        x=np.array(initial,copy=True);x[-1]=1
    return x

def dataset(root):
    rows=json.loads((root/'RESULTS.json').read_text())['rows'];ys={}
    for r in rows:
        a,c=map(int,[r['policy'][1],r['policy'][4]])
        for i,seed in enumerate(r['seeds']):ys[(seed,a,c)]=int(r['success'][i])
    records=[]
    for r in rows:
        a,c=map(int,[r['policy'][1],r['policy'][4]])
        for i,seed in enumerate(r['seeds']):
            y=int(r['success'][i]);ds=r['decisions'][i]
            for d in ds:
                st=d['stage'];base=ys[(seed,1,1)] if st==0 else ys[(seed,a,1)]
                records.append(dict(seed=seed,initial_obs=ds[0]['observation'],obs=d['observation'],action=d['action'],y=y,risk=int(base==1 and y==0),risk_eligible=st==1 or c==1))
    return records

def fit_prior(data,output):
    output.mkdir(parents=True,exist_ok=False);torch.manual_seed(1107)
    fit=[r for r in data if r['seed']<420048];cal=[r for r in data if r['seed']>=420048]
    raw=np.array([r['obs'] for r in fit],dtype=np.float32);mean=raw.mean(0);std=np.maximum(raw.std(0),.05);mean[-1]=0;std[-1]=1
    x=torch.tensor((raw-mean)/std);a=torch.tensor([r['action'] for r in fit]);y=torch.tensor([r['y'] for r in fit],dtype=torch.float32);risk=torch.tensor([r['risk'] for r in fit],dtype=torch.float32);eligible=torch.tensor([r['risk_eligible'] for r in fit]);q=Head(3);h=Head(3);opt=torch.optim.Adam(list(q.parameters())+list(h.parameters()),lr=.002)
    # Fixed epoch count; calibration is reported, not used to select a model.
    for epoch in range(400):
        logits=q(x).gather(1,a[:,None]).squeeze(1);rl=h(x).gather(1,a[:,None]).squeeze(1);loss=nn.functional.binary_cross_entropy_with_logits(logits,y)+nn.functional.binary_cross_entropy_with_logits(rl[eligible],risk[eligible]);opt.zero_grad();loss.backward();opt.step()
    def metrics(rows):
        with torch.no_grad():
            xx=torch.tensor((np.array([r['obs'] for r in rows],np.float32)-mean)/std);aa=torch.tensor([r['action'] for r in rows]);yy=torch.tensor([r['y'] for r in rows],dtype=torch.float32);p=q(xx).sigmoid().gather(1,aa[:,None]).squeeze();return dict(n=len(rows),brier=float(((p-yy)**2).mean()))
    bundle=dict(q=q.state_dict(),risk=h.state_dict(),mean=mean,std=std,fit=metrics(fit),calibration=metrics(cal),fit_seed_range=[420000,420047],calibration_seed_range=[420048,420063],source_sha256=sha(__file__),data_sha256=sha(output.parent/'opportunity64'/'RESULTS.json'))
    torch.save(bundle,output/'prior.pt');(output/'METRICS.json').write_text(json.dumps({k:v for k,v in bundle.items() if k in ['fit','calibration','fit_seed_range','calibration_seed_range','source_sha256','data_sha256']},indent=2));return bundle

def load_prior(path):
    b=torch.load(path,weights_only=False);q=Head(3);q.load_state_dict(b['q']);h=Head(3);h.load_state_dict(b['risk']);q.eval();h.eval();return b,q,h

def initial_model(prior,q,seed,mode='full'):
    torch.manual_seed(seed);m=ActorCritic();opt=torch.optim.Adam(m.parameters(),lr=.002)
    # Same train-only supervised warm start for both online algorithms.
    # No future labels are available at deployment. Ablations receive masked
    # features during both warm start and online updates.
    data=prior['_fit'];xx=np.array([transform(r['obs'],mode,np.array(r['initial_obs'])) for r in data]);x=torch.tensor((xx-prior['mean'])/prior['std'],dtype=torch.float32)
    full=torch.tensor((np.array([r['obs'] for r in data])-prior['mean'])/prior['std'],dtype=torch.float32)
    with torch.no_grad():p=nn.functional.softmax(q(full).sigmoid()/.25,dim=-1);v=(p*q(full).sigmoid()).sum(-1)
    for _ in range(120):
        logits,value=m(x);loss=-(p*nn.functional.log_softmax(logits,dim=-1)).sum(-1).mean()+.5*((value-v)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return m

def policy_for(model,b,mode='full',sample=False,rngseed=0):
    initial={};committed={};gen=torch.Generator().manual_seed(rngseed)
    def policy(o,i,stage):
        if stage==0:initial[i]=o.copy()
        x=transform(o,mode,initial.get(i));xx=torch.tensor((x-b['mean'])/b['std'],dtype=torch.float32)[None]
        with torch.no_grad():
            logits,value=model(xx);probs=logits.softmax(-1)[0];a=int(torch.multinomial(probs,1,generator=gen)) if sample else int(probs.argmax());logp=float(probs[a].log())
            if mode=='no_second':
                if stage==0:
                    xo=x.copy();xo[-1]=1;nextlog,_=model(torch.tensor((xo-b['mean'])/b['std'],dtype=torch.float32)[None]);pp=nextlog.softmax(-1)[0];committed[i]=int(torch.multinomial(pp,1,generator=gen)) if sample else int(pp.argmax())
                else:a=committed[i];logp=float(probs[a].log())
        return a,dict(policy_input=x.tolist(),logp=logp,value=float(value[0]),committed_second=committed.get(i),decision_reobserves=mode!='no_second' or stage==0)
    return policy

def train(ex,b,q,h,output,seed,beta,updates=12,mode='full'):
    output.mkdir(parents=True,exist_ok=False);m=initial_model(b,q,seed,mode);optimizer=torch.optim.Adam(m.parameters(),lr=3e-4);history=[];torch.manual_seed(seed)
    for update in range(updates):
        group=update%32;seeds=list(range(420000+group*16,420016+group*16));p=policy_for(m,b,mode,True,seed*1000+update);t=time.monotonic();r=ex.run(seeds,p);obs=[];act=[];lp=[];val=[];ret=[]
        for i,ds in enumerate(r['decisions']):
            for d in ds:
                obs.append(d['policy_input']);act.append(d['action']);lp.append(d['logp']);val.append(d['value']);ret.append(float(r['success'][i]))
        x=torch.tensor((np.array(obs)-b['mean'])/b['std'],dtype=torch.float32);actions=torch.tensor(act);oldlog=torch.tensor(lp);returns=torch.tensor(ret);advantages=returns-torch.tensor(val)
        # gamma=1 finite native250 horizon, terminal native success returned to
        # BOTH physically visited boundaries. No decision token for failed A.
        advantages=(advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-6)
        with torch.no_grad():risks=h(x).sigmoid()
        for epoch in range(8):
            logits,values=m(x);dist=Categorical(logits=logits);newlog=dist.log_prob(actions);ratio=(newlog-oldlog).exp();clip=torch.clamp(ratio,.8,1.2);policy_loss=-torch.min(ratio*advantages,clip*advantages).mean();value_loss=((values-returns)**2).mean();reg=(dist.probs*risks).sum(-1).mean();entropy=dist.entropy().mean();loss=policy_loss+.5*value_loss-.02*entropy+beta*reg;optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),.5);optimizer.step()
        row=dict(update=update+1,seed=seed,beta=beta,mode=mode,success=sum(r['success']),episodes=16,physical_interactions=(update+1)*4000,decision_tokens=len(act),stage0_verified=sum(any(z['valid'] and z['stage']==0 for z in bs) for bs in r['boundaries']),wall_s=time.monotonic()-t,planning_wall_s=r['planning_wall_s'],policy_loss=float(policy_loss),value_loss=float(value_loss),predicted_risk=float(reg));history.append(row)
        (output/f'rollout_{update+1:02d}.json').write_text(json.dumps(r,separators=(',',':')));(output/'HISTORY.json').write_text(json.dumps(history,indent=2))
        if update+1 in [4,12]:torch.save(dict(model=m.state_dict(),prior_path=str(output.parent/'prior'/'prior.pt'),mean=b['mean'],std=b['std'],seed=seed,beta=beta,mode=mode,updates=update+1,physical_interactions=(update+1)*4000,source_sha256=sha(__file__)),output/f'model_{update+1:02d}.pt')
        print(json.dumps(dict(event='TRAIN',**row)),flush=True)
    return m

def eval_policy(kind,b,q,h,model=None,mode='full',fixed=(1,1)):
    if model is not None:return policy_for(model,b,mode)
    def policy(o,i,st):
        if kind in ['nominal','fixed']:a=fixed[st]
        elif kind=='rule':
            if st==0:
                axis=o[:2];angle=np.arctan2(axis[1],axis[0]);C_from_B=o[3:5]+axis/2;angles=angle+np.array([-1,0,1])*np.pi/2;points=.05*np.c_[np.cos(angles),np.sin(angles)];a=int(np.argmax(np.linalg.norm(points-C_from_B,axis=1)))
            else:a=1
        else:
            with torch.no_grad():
                xx=torch.tensor((o-b['mean'])/b['std'],dtype=torch.float32)[None];scores=q(xx).sigmoid()[0];base=scores[1];score=scores if kind=='absolute_q' else scores-base-2*h(xx).sigmoid()[0];a=int(score.argmax())
        return a,dict(kind=kind)
    return policy

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['fit','train','eval'],required=True);p.add_argument('--ablate',action='store_true');a=p.parse_args();root=a.root
    if a.phase=='fit':fit_prior(dataset(root/'opportunity64'),root/'prior');return
    b,q,h=load_prior(root/'prior'/'prior.pt');b['_fit']=[r for r in dataset(root/'opportunity64') if r['seed']<420048];ex=VectorHandoff(trace=a.phase=='eval')
    if a.phase=='train':
        configs=[(s,be,'full',12) for s in [31,37,43] for be in [0.,2.]] if not a.ablate else [(31,2.,mode,4) for mode in ['no_next','no_posterior','no_second']]
        for seed,beta,mode,updates in configs:
            path=root/f'ppo_s{seed}_b{beta:g}_{mode}';train(ex,b,q,h,path,seed,beta,updates,mode)
    else:
        means={}
        for r in json.loads((root/'opportunity64'/'RESULTS.json').read_text())['rows']:means.setdefault(r['policy'],[]).extend(r['success'])
        best=max(means,key=lambda k:(np.mean(means[k]),k=='A1_C1'));fixed=(int(best[1]),int(best[4]));models=[]
        for kind in ['nominal','rule','fixed','absolute_q','paired_ranker']:models.append((kind,eval_policy(kind,b,q,h,fixed=(1,1) if kind=='nominal' else fixed),None))
        for seed in [31,37,43]:
            for beta in [0,2]:
                path=root/f'ppo_s{seed}_b{beta}_full'/'model_12.pt';ck=torch.load(path,weights_only=False);m=ActorCritic();m.load_state_dict(ck['model']);models.append((path.parent.name,eval_policy('model',b,q,h,m),sha(path)))
        for mode in ['no_next','no_posterior','no_second']:
            path=root/f'ppo_s31_b2_{mode}'/'model_04.pt';ck=torch.load(path,weights_only=False);m=ActorCritic();m.load_state_dict(ck['model']);models.append((path.parent.name,eval_policy('model',b,q,h,m,mode),sha(path)))
        # Same-budget reference for the three64-episode ablations.
        path=root/'ppo_s31_b2_full'/'model_04.pt';ck=torch.load(path,weights_only=False);m=ActorCritic();m.load_state_dict(ck['model']);models.append(('ppo_s31_b2_full_04',eval_policy('model',b,q,h,m),sha(path)))
        out=root/'development64';out.mkdir(exist_ok=False);allrows=[]
        for name,policy,modelsha in models:
            for batch in range(4):
                r=ex.run(list(range(520000+batch*16,520016+batch*16)),policy);r.update(method=name,model_sha256=modelsha,scope='UNSEEN_DEVELOPMENT_NOT_SEALED');trajectory=out/f'{name}_{batch}.trace.json.gz'
                with gzip.open(trajectory,'wt') as f:json.dump(dict(initial=ex.initial,transitions=ex.rows,plans=ex.plans),f,separators=(',',':'))
                r['trajectory_sha256']=sha(trajectory);r['trajectory_file']=trajectory.name;allrows.append(r);(out/f'{name}_{batch}.json').write_text(json.dumps(r,separators=(',',':')));(out/'RESULTS.json').write_text(json.dumps(dict(rows=allrows,physical_interactions=sum(z['physical_interactions'] for z in allrows),best_fixed_training_only=best),indent=2));print(json.dumps(dict(event='EVAL',method=name,batch=batch,success=sum(r['success']),verified=sum(any(x['valid'] and x['stage']==0 for x in bs) for bs in r['boundaries']),wall_s=r['wall_s'])),flush=True)
    ex.close()
if __name__=='__main__':main()
