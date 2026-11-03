"""Predeclared fresh development expansion. Freeze every model hash before rollout.
No training on these scenes, no sealed data. Final checkpoints only, no selection.
"""
import argparse,json,hashlib,time,gzip
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,prior,normalizer
from program_peg_option_runtime import Runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False);torch.set_num_threads(2);models=[]
 for seed in [31301,31302]:
  for mode in ['uniform','ees','handoff','local','policy','joint','independent_kernel']:models.append((f'frontier_{mode}',seed,a.root/f'frontier_{mode}_{seed}/FINAL.pt'))
  models.append(('mature_policy128',seed,a.root/f'mature_policy_{seed}/FINAL.pt'))
  for kind in ['dsac','iql','stap']:models.append((kind,seed,a.root/f'offline/{kind}_{seed}_2000.pt'))
  models.append(('projected',seed,a.root/f'projected_control/SAC_{seed}_2000.zip'))
 while not all(path.exists() for _,_,path in models):time.sleep(5)
 for name,seed,path in models:assert path.exists(),path
 # Same acquired history, standard supervised/FQI controls, exact update schedule.
 # This tests whether any effect requires RL-specific Bellman treatment.
 data,_=prior(a.prior);histories=[]
 for seed in [31301,31302]:
  for acquisition,prefix in [('frontier_handoff','matched'),('mature_policy','matched_mature'),('frontier_joint','matched_joint')]:
   online=json.loads((a.root/f'{acquisition}_{seed}/ONLINE.json').read_text());paths=list(dict.fromkeys(t['source']for t in online))
   for kind in ['supervised','fqi','iql','cql','stap']:
    m=Learner(kind,seed,normalizer(data));m.train(data,steps=800);buf=[]
    for k,path in enumerate(paths):
     buf.extend(t for t in online if t['source']==path)
     if acquisition=='mature_policy'or k<8:m.train(data,buf,steps=50)
     elif k%2==1:m.train(data,buf,steps=100)
    f=a.out/f'{prefix}_{kind}_{seed}.pt';m.save(f);histories.append(dict(kind=kind,seed=seed,acquisition=acquisition,updates=m.updates,history=m.history,transitions=len(buf)));models.append((f'{prefix}_{kind}',seed,f))
 (a.out/'MATCHED_TRAINING.json').write_text(json.dumps(histories,indent=2))
 manifest=dict(scenes=list(range(751220,751228)),split='new development expansion, never sealed; no fitting on these scenes',models=[dict(method=n,seed=s,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())for n,s,p in models],checkpoints='predeclared FINAL/2000 only',fixed_candidates='all six 3x2, best is hindsight benchmark not deployed oracle',evaluation_protocol='All48 finite physical branches; primary18 model/seed conditions also executed directly with exact100step equality; remaining policies explicitly logged-branch only, not fresh rollouts')
 (a.out/'FROZEN_MODELS.json').write_text(json.dumps(manifest,indent=2));rt=Runtime(a.out/'physics');scores=[]
 try:
  panel={}
  for aa in range(3):
   for bb in range(2):
    rr=[]
    for scene in manifest['scenes']:
     row,ts,d=rt.run(scene,lambda o:aa if o[44]<.5 else bb,f'fixed{aa}{bb}');panel[(scene,aa,bb)]=(row,ts,d);rr.append(row)
    scores.append(dict(method=f'fixed{aa}{bb}',seed=None,success=sum(r['success']for r in rr),n=len(rr),sources=[r['path']for r in rr],evaluation='DIRECT_NATIVE_PHYSICS'))
  for scene in manifest['scenes']:
   for aa in range(3):
    left=panel[(scene,aa,0)];right=panel[(scene,aa,1)];n=left[1][0]['duration'];assert left[2]['initial']==right[2]['initial']and left[2]['steps'][:n]==right[2]['steps'][:n]
  direct={'frontier_policy','frontier_ees','frontier_joint','frontier_independent_kernel','mature_policy128','dsac','projected','matched_joint_supervised','matched_joint_iql'}
  predictions=[]
  for kind,seed,path in models:
   if kind=='projected':
    import program_contact_sac as sac
    sac.mask=lambda obs:torch.stack([torch.ones_like(obs[:,44]),torch.zeros_like(obs[:,44])],1);m=sac.BoundarySAC.load(path,device='cpu');norm=np.load(a.root/'projected_control/NORMALIZATION.npz')
    def policy(obs):
     u=m.predict((np.asarray(obs,dtype=np.float32)-norm['mu'])/norm['sd'],deterministic=True)[0][0];return int(np.clip(np.rint(u)+1,0,2))if obs[44]<.5 else int(u>=0)
   else:m=Learner.load(path);policy=m.choose
   rows=[]
   for scene in manifest['scenes']:
    first=policy(panel[(scene,0,0)][1][0]['s']);ts=panel[(scene,first,0)][1];second=policy(ts[1]['s'])if len(ts)==2 else 0;reference,_,raw=panel[(scene,first,second)];entry=dict(method=kind,seed=seed,scene=scene,actions=[first,second]if len(ts)==2 else [first],success=reference['success'],reference=reference['path'],reference_sha=reference['sha256'],direct_checked=False)
    if kind in direct:
     row,_,d=rt.run(scene,policy,f'{kind}_{seed}');assert d['initial']==raw['initial']and d['steps']==raw['steps'],('NONDETERMINISTIC_BRANCH',kind,seed,scene);assert row['success']==reference['success'];entry.update(direct_checked=True,direct_source=row['path'],all100steps_equal=True);rows.append(row)
    else:rows.append(reference)
    predictions.append(entry)
   scores.append(dict(method=kind,seed=seed,success=sum(r['success']for r in rows),n=len(rows),sources=[r['path']for r in rows],evaluation='DIRECT_NATIVE_PHYSICS_EXACT_BRANCH_CHECK'if kind in direct else 'FROZEN_POLICY_ON_REAL_FINITE_BRANCH_PANEL_NOT_NEW_ROLLOUT'))
   (a.out/'SCORES.json').write_text(json.dumps(scores,indent=2));(a.out/'BRANCH_PREDICTIONS.json').write_text(json.dumps(predictions,indent=2))
 finally:rt.close()
if __name__=='__main__':main()
