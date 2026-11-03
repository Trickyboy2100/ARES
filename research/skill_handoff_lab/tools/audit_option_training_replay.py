"""Reconstruct optimizer history from real physical episode records and checkpoints.
Also measure acquisition predictions against actual actor change / old-panel return.
"""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,prior
from run_discrete_handoff import evaluate_logged

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);data,groups=prior(a.prior);report=[]
 for folder in sorted(a.root.iterdir()):
  if not folder.is_dir()or not(folder/'ONLINE.json').exists()or not(folder/'FINAL.pt').exists():continue
  real=json.loads((folder/'ONLINE.json').read_text());sources=list(dict.fromkeys(t['source']for t in real));m=Learner.load(folder/'MODEL_0.pt');buf=[];records=[];start=time.monotonic();paired=folder.name.startswith(('pairs_','frontier_'))and not folder.name.startswith('frontier_policy_')
  action_rng=np.random.default_rng(m.seed)
  if folder.name.startswith('frontier_policy_'):
   for _ in range(8):action_rng.integers(3);action_rng.integers(2)
  for k,source in enumerate(sources):
   episode=[t for t in real if t['source']==source]
   if folder.name.startswith(('online_policy_','mature_policy_')):
    for t in episode:m.choose(t['s'],stochastic=True)
   elif folder.name.startswith('frontier_policy_')and k>=8:
    for t in episode:
     if action_rng.random()<.15:action_rng.integers(3 if t['stage']==0 else 2)
     else:m.choose(t['s'],stochastic=True)
   buf.extend(episode)
   if paired and k>=8 and k%2==0:continue
   before=evaluate_logged(m,groups)if paired and k>=8 else None
   m.train(data,buf,steps=100 if paired and k>=8 else 50)
   after=evaluate_logged(m,groups)if before else None
   if before:
    ai=(k-9)//2;acq=json.loads((folder/f'ACQUIRE_{ai:03d}.json').read_text());selected=acq['chosen'];records.append(dict(pair=ai,acquisition_score=acq['scores'][selected],old_training_panel_before=sum(r['success']for r in before),old_training_panel_after=sum(r['success']for r in after),rescues=sum(z['success']>x['success']for x,z in zip(before,after)),regressions=sum(z['success']<x['success']for x,z in zip(before,after)),first_policy_changes=sum(x['a']!=z['a']for x,z in zip(before,after)),actual_pair_outcomes=acq['outcomes'],scope='old training panel diagnostic, not generalization'))
  final=Learner.load(folder/'FINAL.pt');errs={key:max(float((v-final.__dict__[key].state_dict()[name]).abs().max())for name,v in m.__dict__[key].state_dict().items())for key in ['q','target','actor','v','dyn']};assert max(errs.values())<1e-6,(folder,errs);assert m.updates==final.updates
  report.append(dict(run=folder.name,episodes=len(sources),transitions=len(real),optimizer_updates=m.updates,additional_updates=m.updates-800,max_parameter_error=errs,acquisition_effects=records,replay_audit_wall_s=time.monotonic()-start));(a.root/'TRAINING_REPLAY_AUDIT.json').write_text(json.dumps(report,indent=2));print(folder.name,'EXACT',m.updates,errs,flush=True)
if __name__=='__main__':main()
