"""Bounded matched-data contract auxiliary / finite-horizon fitted-value experiment.

No future boundary is available to the deployed policy. Auxiliary targets are
measured training outcomes, masked when a boundary was never reached. This is a
STAP/skill-chaining-inspired value backup, not a reproduction of either method.
"""
import argparse,csv,gzip,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from handoff_opportunity import load
from handoff_learn import load_prior,eval_policy,ActorCritic
from handoff_vector import VectorHandoff
from handoff_physics import sha

torch.set_num_threads(4)
class Net(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(57,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,18))
    def forward(self,x):return self.net(x).reshape(-1,3,6)

def feature(x,mode):
    x=np.array(x,dtype=np.float32,copy=True)
    if mode=='contract_no_history':x[...,39:48]=0
    return x

def records(episodes):
    rows=[[],[]]
    for (s,a,c),r in sorted(episodes.items()):
        for d in r['decisions']:
            st=d['stage']
            if st==0 and c!=1:continue
            bs=[b for b in r['boundaries'] if b['stage']==st];b=bs[-1] if bs else None
            aux=np.zeros(5,np.float32);mask=np.zeros(5,np.float32)
            # The validity label is observed even if a motion failed earlier.
            aux[0]=float(b is not None and b['valid']);mask[0]=1
            if b:
                aux[1:]=[np.clip(1-b['target_xy_error_m']/.015,-1,1),np.clip(b['native_AB_margin_m']/.02,-1,1),np.clip(b['predicted_centered_min_overlap_m2' if st==0 else 'actual_min_overlap_m2']/.0008,0,1),1-b['step']/250];mask[1:]=1
            yy=np.mean([episodes[s,a,cc]['y'] for cc in range(3)]) if st==0 else r['y']
            nxt=r['decisions'][1]['observation'] if st==0 and len(r['decisions'])==2 else None
            rows[st].append(dict(seed=s,obs=d['observation'],action=d['action'],y=float(yy),aux=aux,mask=mask,next=nxt))
    return rows

def fit(source,out,contact=None):
    out.mkdir(parents=True,exist_ok=False);data=records(load(source));fitrows=[[r for r in rs if r['seed']<420048] for rs in data];raw=np.array([r['obs'] for rs in fitrows for r in rs]);mean=raw.mean(0);std=np.maximum(raw.std(0),.05);metrics={}
    if contact is not None:
        lookup={(int(r['seed']),int(r['policy'][1]),int(r['policy'][4])):r for r in csv.DictReader(contact.open())}
        # Match stage1 using the exact observed prefix and suffix action.
        episodes=load(source)
        by_observation={(s,tuple(r['decisions'][1]['observation']),c):lookup[s,a,c] for (s,a,c),r in episodes.items() if len(r['decisions'])==2}
        for r in data[1]:
            z=by_observation[r['seed'],tuple(r['obs']),r['action']]
            r['aux']=np.array([1-int(z['grasp_failed']),np.clip(1-float(z['tcp_cube_xy_mm'])/30,-1,1),np.clip(1-float(z['tcp_target_error_mm'])/40,-1,1),np.clip(1-float(z['cube_moved_since_handoff_mm'])/30,-1,1),np.clip(float(z['finger_opening_mm'])/40,0,1)],np.float32)
            r['mask']=np.ones(5,np.float32)
    for mode in (['contract_contact'] if contact is not None else ['supervised','fitted_value','contract','contract_no_history']):
        torch.manual_seed(20261029);nets=[Net(),Net()];ms={}
        def x(rows):return torch.tensor((feature([r['obs'] for r in rows],mode)-mean)/std,dtype=torch.float32)
        for st in [1,0]:
            rr=fitrows[st];xx=x(rr);aa=torch.tensor([r['action'] for r in rr]);yy=torch.tensor([r['y'] for r in rr],dtype=torch.float32)
            if st==0 and mode!='supervised':
                # One true physical prefix transition, terminal zero on failure.
                with torch.no_grad():
                    yy=torch.tensor([float(nets[1](torch.tensor((feature(r['next'],mode)-mean)/std,dtype=torch.float32)[None])[0,:,0].sigmoid().max()) if r['next'] is not None else 0. for r in rr])
            aux=torch.tensor(np.stack([r['aux'] for r in rr]));mask=torch.tensor(np.stack([r['mask'] for r in rr]));net=nets[st];opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.001)
            for epoch in range(300):
                pred=net(xx)[torch.arange(len(rr)),aa];loss=nn.functional.binary_cross_entropy_with_logits(pred[:,0],yy)
                if mode.startswith('contract'):loss=loss+.2*((pred[:,1:]-aux).square()*mask).sum()/mask.sum()
                opt.zero_grad();loss.backward();opt.step()
            cal=[r for r in data[st] if r['seed']>=420048]
            with torch.no_grad():
                pp=net(x(cal))[torch.arange(len(cal)),torch.tensor([r['action'] for r in cal]),0].sigmoid().numpy();target=np.array([r['y'] for r in cal]);ms[str(st)]=dict(n_fit=len(rr),n_cal=len(cal),cal_brier_vs_observed_suffix=float(np.mean((pp-target)**2)),note='stage0 fitted backup estimates optimal suffix, not observed uniform suffix; Brier not same estimand' if st==0 and mode!='supervised' else 'observed terminal return')
        torch.save(dict(nets=[m.state_dict() for m in nets],mean=mean,std=std,mode=mode,epochs=300,seed=20261029,source_sha256=sha(__file__),data_sha256=sha(source/'RESULTS.json')),out/f'{mode}.pt');metrics[mode]=ms
    (out/'TRAINING.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics,indent=2))

def learned(path):
    b=torch.load(path,weights_only=False);nets=[Net(),Net()]
    for m,s in zip(nets,b['nets']):m.load_state_dict(s);m.eval()
    def policy(o,i,st):
        with torch.no_grad():scores=nets[st](torch.tensor((feature(o,b['mode'])-b['mean'])/b['std'],dtype=torch.float32)[None])[0,:,0].sigmoid().numpy()
        return int(scores.argmax()),dict(method=b['mode'],value_scores=scores.tolist(),future_outcome_input=False)
    return policy

def evaluate(old,models,out,batches,contact_only=False):
    out.mkdir(parents=True,exist_ok=False);b,q,h=load_prior(old/'prior/prior.pt');ck=torch.load(old/'ppo_s31_b0_full/model_12.pt',weights_only=False);m=ActorCritic();m.load_state_dict(ck['model'])
    policies=[('nominal',eval_policy('nominal',b,q,h)),('rule',eval_policy('rule',b,q,h)),('fixed',eval_policy('fixed',b,q,h,fixed=(0,0))),('ordinary_ppo',eval_policy('model',b,q,h,m))]+([] if contact_only else [(mode,learned(models/f'{mode}.pt')) for mode in ['supervised','fitted_value','contract','contract_no_history']])
    if contact_only:policies=[('contract_contact',learned(models/'contract_contact.pt'))]
    ex=VectorHandoff(trace=True);allrows=[];digests={}
    try:
        for batch in range(batches):
            for name,policy in policies:
                r=ex.run(list(range(530000+16*batch,530016+16*batch)),policy)
                if batch not in digests:digests[batch]=r['initial_sha256']
                assert digests[batch]==r['initial_sha256']
                path=out/f'{name}_{batch}.trace.json.gz'
                with gzip.open(path,'wt') as f:json.dump(dict(initial=ex.initial,transitions=ex.rows,plans=ex.plans),f,separators=(',',':'))
                r.update(method=name,batch=batch,trajectory_file=path.name,trajectory_sha256=sha(path));allrows.append(r)
                (out/'RESULTS.json').write_text(json.dumps(dict(rows=allrows,scope='NEW_EXPLORATORY_DEVELOPMENT; fixed settings, no sealed test',physical_interactions=sum(r['physical_interactions'] for r in allrows)),separators=(',',':')))
                print(json.dumps(dict(method=name,batch=batch,success=sum(r['success']),wall_s=r['wall_s'])),flush=True)
    finally:ex.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['fit','eval','fit-contact','eval-contact'],required=True);p.add_argument('--batches',type=int,default=2);p.add_argument('--contact',type=Path);a=p.parse_args()
    if a.phase=='fit':fit(a.old/'opportunity64',a.out/'models')
    elif a.phase=='fit-contact':
        assert a.contact is not None
        fit(a.old/'opportunity64',a.out/'models_contact',a.contact)
    elif a.phase=='eval-contact':evaluate(a.old,a.out/'models_contact',a.out/'physical32_contact',a.batches,True)
    else:evaluate(a.old,a.out/'models',a.out/'physical32',a.batches)
if __name__=='__main__':main()
