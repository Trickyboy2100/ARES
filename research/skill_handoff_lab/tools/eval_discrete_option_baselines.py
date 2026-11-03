"""Fresh full native100 baseline execution, fixed scenes, no development fitting."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner
from program_peg_option_runtime import Runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False);torch.set_num_threads(2);rt=Runtime(a.out/'physics');scores=[]
 try:
  for seed in [31301,31302]:
   for kind in ['dsac','fqi','iql','cql','supervised','stap','projected']:
    if kind=='projected':
     import program_contact_sac as sac
     sac.mask=lambda obs:torch.stack([torch.ones_like(obs[:,44]),torch.zeros_like(obs[:,44])],1)
     model=sac.BoundarySAC.load(a.root/f'projected_control/SAC_{seed}_2000.zip',device='cpu');norm=np.load(a.root/'projected_control/NORMALIZATION.npz')
     def policy(obs):
      act=model.predict((obs-norm['mu'])/norm['sd'],deterministic=True)[0][0];return int(np.clip(np.rint(act)+1,0,2))if obs[44]<.5 else int(act>=0)
    else:
     model=Learner.load(a.root/f'offline/{kind}_{seed}_2000.pt');policy=model.choose
    rows=[]
    for scene in range(751200,751206):rows.append(rt.run(scene,policy,f'{kind}_{seed}')[0])
    scores.append(dict(method=kind,seed=seed,success=sum(r['success']for r in rows),n=len(rows),sources=[r['path']for r in rows]));(a.out/'SCORES.json').write_text(json.dumps(scores,indent=2))
  for grasp in [0,1,2]:
   for refresh in [0,1]:
    rows=[rt.run(s,lambda obs:grasp if obs[44]<.5 else refresh,f'fixed{grasp}{refresh}')[0]for s in range(751200,751206)];scores.append(dict(method=f'fixed{grasp}{refresh}',seed=None,success=sum(r['success']for r in rows),n=len(rows),sources=[r['path']for r in rows]));(a.out/'SCORES.json').write_text(json.dumps(scores,indent=2))
 finally:rt.close()
if __name__=='__main__':main()
