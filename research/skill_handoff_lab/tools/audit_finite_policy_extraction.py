"""Post-hoc baseline diagnosis: actor vs exact min/mean finite-Q choices.
No model fitting, no new physics, no claim of independent method gain.
"""
import argparse,json,gzip
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,episode_transitions
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;panel={}
for row in json.loads((r/'expansion_eval/physics/RESULTS.json').read_text()):
 if row['tag'].startswith('fixed'):
  aa,bb=map(int,row['tag'][-2:]);d=json.load(gzip.open(row['path'],'rt'));panel[row['seed'],aa,bb]=(row,episode_transitions(d,[-.01,0,.01][aa],bb,row['path']))
reports=[]
for family in ['frontier_joint','frontier_current_joint','mature_policy']:
 for seed in [31301,31302]:
  m=Learner.load(r/f'{family}_{seed}/FINAL.pt');policies={};diagnostics=[]
  def greedy(obs,mode):
   _,qs=m.values(obs);n=3 if obs[44]<.5 else 2;v=qs[:,0,:n];return int((v.min(0)if mode=='min'else v.mean(0)).argmax())
  for mode in ['actor','min','mean']:
   out=[]
   for scene in range(751220,751228):
    s0=panel[scene,0,0][1][0]['s'];aa=m.choose(s0)if mode=='actor'else greedy(s0,mode);ts=panel[scene,aa,0][1];bb=(m.choose(ts[1]['s'])if mode=='actor'else greedy(ts[1]['s'],mode))if len(ts)==2 else 0;out.append(dict(scene=scene,actions=[aa,bb]if len(ts)==2 else[aa],success=panel[scene,aa,bb][0]['success']))
    if mode=='actor':
     for t in ts:
      obs=t['s'];_,qs=m.values(obs);n=3 if obs[44]<.5 else 2;v=qs[:,0,:n].min(0);exact=np.exp((v-v.max())/m.alpha);exact/=exact.sum();pp=m.values(obs)[0][0,:n];diagnostics.append(dict(scene=scene,stage=t['stage'],actor=pp.tolist(),min_q=v.tolist(),mean_q=qs[:,0,:n].mean(0).tolist(),exact_softmax_min=exact.tolist(),actor_vs_exact_l1=float(abs(pp-exact).sum()),actor_argmax=int(pp.argmax()),min_argmax=int(v.argmax())))
   policies[mode]=dict(success=sum(x['success']for x in out),n=8,rows=out)
  reports.append(dict(run=f'{family}_{seed}',policies=policies,diagnostics=diagnostics))
(r/'POLICY_EXTRACTION_AUDIT.json').write_text(json.dumps(dict(scope='posthoc logged finite-branch diagnostic, no fitting and no extra physics; min vs mean distinguishes actor approximation from clipped-Q pessimism',models=reports),indent=2));print([(x['run'],{k:v['success']for k,v in x['policies'].items()})for x in reports])
