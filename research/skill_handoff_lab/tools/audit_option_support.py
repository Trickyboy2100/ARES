"""Actual legal actions need not have historical execution support."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from program_discrete_options import prior,Learner
from program_option_improvement import Improved

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);data,_=prior(a.prior);obs=np.unique(np.array([t['s']for t in data if t['stage']==1]),axis=0);out=[]
 for seed in [31401,31402]:
  for kind in ['sac_mean','vi_mean','handoff','handoff_complete','fqi','cql']:
   folder=a.root/(f'train_complete_{seed}'if kind=='handoff_complete'else f'train_cql_{seed}'if kind=='cql'else f'train_{seed}')
   for cp in [0,288]:
    f=folder/f'{kind}_{cp}.pt'
    if not f.exists():continue
    cls=Learner if kind in ('cql','fqi')else Improved;m=cls.load(f);v,q=m.values(obs);q=q.mean(0)
    out.append(dict(seed=seed,kind=kind,checkpoint=cp,unique_handoffs=len(obs),unobserved_action_greedy=int((q.argmax(1)==2).sum()),unobserved_over_one=int((q[:,2]>1).sum()),new_action_min=float(q[:,2].min()),new_action_max=float(q[:,2].max()),values=q.tolist(),scope='unexecuted new action on old compatible handoffs; bounded-terminal-reward violation is genuine, but unknown actual new-action outcomes are NOT assumed failures'))
 a.out.write_text(json.dumps(out,indent=2));print([(x['kind'],x['seed'],x['checkpoint'],x['unobserved_over_one'])for x in out])
if __name__=='__main__':main()
