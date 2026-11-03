"""Offline->online physical study. No replayed-state physics; full prefixes charged."""
import argparse,json,time,shutil,gzip,hashlib
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,prior,normalizer,CONTROLLER

def savejson(p,d):p.write_text(json.dumps(d,indent=2))
def evaluate_logged(model,groups):
    out=[]
    for scene in sorted({g['seed']for g in groups}):
        gs=sorted([g for g in groups if g['seed']==scene],key=lambda g:g['grasp']);a=model.choose(gs[0]['items'][0][2][0][0]);g=gs[a];ts=g['items'][0][2];b=model.choose(ts[1][0])if len(ts)==2 else 0
        r=g['items'][b][0];out.append(dict(scene=scene,a=a,b=b,success=int(r['success'])))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['offline','online','fixture'],default='offline');p.add_argument('--method',default='dsac');p.add_argument('--seed',type=int,default=31301);p.add_argument('--allocation',default='uniform');p.add_argument('--episodes',type=int,default=32);p.add_argument('--dev-start',type=int,default=751200);p.add_argument('--dev-n',type=int,default=6);p.add_argument('--train-start',type=int,default=751000);p.add_argument('--train-count',type=int,default=16);p.add_argument('--eval-every',type=int,default=16);p.add_argument('--checkpoint',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    for name in ['program_discrete_options.py','program_peg_option_runtime.py','run_discrete_handoff.py']:shutil.copy2(Path(__file__).with_name(name),a.out/name)
    data,groups=prior(a.prior);savejson(a.out/'PRIOR.json',data);savejson(a.out/'PROTOCOL.json',dict(args={k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()},controller=CONTROLLER,claims='standard discrete baselines; candidate acquisition UNVERIFIED',gamma=1.,reward='native success only at100',prior_scenes=sorted({g['seed']for g in groups}),new_train=list(range(a.train_start,a.train_start+a.train_count)),new_dev=list(range(a.dev_start,a.dev_start+a.dev_n)),sealed=False))
    if a.mode=='offline':
        allscores=[]
        for seed in [31301,31302]:
            for kind in ['dsac','fqi','iql','cql','supervised','stap']:
                m=Learner(kind,seed,normalizer(data));m.save(a.out/f'{kind}_{seed}_0.pt')
                for step in [200,800,2000]:
                    m.train(data,steps=step-m.updates);m.save(a.out/f'{kind}_{seed}_{step}.pt');allscores.append(dict(seed=seed,method=kind,updates=step,logged_training_selection=evaluate_logged(m,groups),training=m.history[-1]));savejson(a.out/'TRAINING.json',allscores)
                print('OFFLINE',kind,seed,allscores[-1]['logged_training_selection'],flush=True)
        return
    from program_peg_option_runtime import Runtime
    rt=Runtime(a.out/'physics');online=[];logs=[];rng=np.random.default_rng(a.seed);model=Learner.load(a.checkpoint)if a.checkpoint else Learner(a.method,a.seed,normalizer(data));
    if not a.checkpoint:model.train(data,steps=800)
    # Every allocator uses the same ordinary discrete SAC. Candidate changes ONLY
    # data collection, never Bellman targets, optimizer, observations or permissions.
    try:
        def eval_cp(k):
            model.save(a.out/f'MODEL_{k}.pt');rr=[]
            for scene in range(a.dev_start,a.dev_start+a.dev_n):
                row,_,_=rt.run(scene,lambda obs:model.choose(obs),f'dev{k}');rr.append(row)
            logs.append(dict(episodes=k,updates=model.updates,success=sum(r['success']for r in rr),n=len(rr),training_steps=100*k,eval_steps=sum(r['steps']for r in rt.rows if r['tag'].startswith('dev')),decisions=len(online),training=model.history[-1]if model.history else {},sources=[r['path']for r in rr]));savejson(a.out/'CURVE.json',logs)
        if a.mode=='fixture':
            for scene in [750103]:
                for act in [0,1]:rt.run(scene,lambda obs:0 if obs[44]<.5 else act,f'fixture{act}')
            return
        eval_cp(0)
        # Same initial-condition schedule for all allocations; no access to dev
        # outcomes or future scene measurements. New initial states seen on reset.
        for k in range(a.episodes):
            scene=a.train_start+k%a.train_count;decisions=[]
            def collect(obs):
                n=3 if obs[44]<.5 else 2;values,qq=model.values(obs);q=qq[:,0,:n];mu=q.mean(0);std=q.std(0);greedy=model.choose(obs)
                if a.allocation=='uniform':act=int(rng.integers(n));score=np.ones(n)/n
                elif a.allocation=='policy':act=model.choose(obs,True);score=values[0,:n]
                elif a.allocation=='uncertainty':score=std+.1*mu;act=int(score.argmax())
                elif a.allocation=='failure':score=1-np.clip(mu,0,1);act=int(rng.choice(n,p=(score+.05)/(score+.05).sum()))
                else:
                    # EES-style local competence improvement proxy: uncertainty
                    # scaled by remaining headroom. Not full symbolic EES.
                    competence=np.clip(mu,0,1);potential=std*(1-competence)
                    if a.allocation=='practice':score=potential
                    else:
                        # Decision relevance: expected reduction of regret against
                        # incumbent option. Common-value uncertainty cancels in
                        # paired advantage, unlike independent competence/UCB.
                        adv=q-q[:,greedy,None];gap=mu-mu[greedy];score=adv.std(0)*np.exp(-np.abs(gap)/(.05+adv.std(0)))
                        if a.allocation=='handoff':score*=float(obs[41]) if obs[44]>.5 else 1.
                    act=int(score.argmax())
                if rng.random()<.15:act=int(rng.integers(n))
                decisions.append(dict(stage=int(obs[44]),q=q.tolist(),score=score.tolist(),chosen=act,greedy=greedy,remaining=float(obs[41])*100));return act
            row,ts,d=rt.run(scene,collect,f'train{k+1}');online+=ts;model.train(data,online,steps=50);savejson(a.out/'ONLINE.json',online);savejson(a.out/f'ACQUIRE_{k+1:03d}.json',decisions)
            if (k+1)%a.eval_every==0 or k+1==a.episodes:eval_cp(k+1)
        model.save(a.out/'FINAL.pt');savejson(a.out/'HISTORY.json',model.history)
    finally:rt.close()
if __name__=='__main__':main()
