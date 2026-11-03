"""Exact decision shadow on already executed acquisition histories; no new physics.
If a choice changes, this is not a counterfactual physical outcome. If all choices
match, deterministic continuation is established for that particular history only.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner,prior
from program_handoff_acquisition import JointOutcomeKernel,update_score,frontier_score

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);data,groups=prior(a.prior);kernel=JointOutcomeKernel(groups);reports=[]
 for seed in [31301,31302]:
  folder=a.root/f'frontier_handoff_{seed}'
  if not(folder/'FINAL.pt').exists():continue
  online=json.loads((folder/'ONLINE.json').read_text());sources=list(dict.fromkeys(t['source']for t in online));buf=[t for t in online if t['source']in sources[:8]];anchors=[t['s']for t in data+buf if t['stage']==0];pool={}
  for source in sources[:8]:
   ts=[t for t in online if t['source']==source];scene=ts[0]['scene'];aa=ts[0]['a']
   for act in range(3):pool[(scene,act)]=dict(scene=scene,grasp=act,prefix=dict(ts[0],a=act),suffix=None)
   pool[(scene,aa)]['suffix']=ts[1]if len(ts)==2 else None
  m=Learner.load(folder/'MODEL_8.pt')
  for j in range(8):
   acq=json.loads((folder/f'ACQUIRE_{j:03d}.json').read_text());keys=[tuple(k)for k in acq['keys']];scores={'independent_kernel':[],'joint_kernel':[]}
   for key in keys:
    r=pool[key]
    if r['suffix']is None:
     score,_=frontier_score(m,r,anchors)
     for mode in scores:scores[mode].append(score)
    else:
     for mode in scores:
      pp=kernel.predict(r['suffix']['s'],mode=='independent_kernel');score,_=update_score(m,r,anchors,joint=pp);scores[mode].append(score)
   reports.append(dict(seed=seed,pair=j,actual_chosen=keys[acq['chosen']],scores=scores,shadow_chosen={mode:keys[int(np.argmax(ss))]for mode,ss in scores.items()},same_as_executed={mode:int(np.argmax(ss))==acq['chosen']for mode,ss in scores.items()}))
   for source in sources[8+2*j:10+2*j]:
    ts=[t for t in online if t['source']==source];buf+=ts;pool[(ts[0]['scene'],ts[0]['a'])]['prefix']=ts[0];pool[(ts[0]['scene'],ts[0]['a'])]['suffix']=ts[1]if len(ts)==2 else None
   m.train(data,buf,steps=100)
  final=Learner.load(folder/'FINAL.pt');assert max(float((p-q).abs().max())for p,q in zip(m.q.parameters(),final.q.parameters()))==0
 (a.root/'JOINT_SHADOW.json').write_text(json.dumps(dict(scope='decision shadow only; no invented counterfactual task results',rows=reports),indent=2));print([(r['seed'],r['pair'],r['actual_chosen'],r['shadow_chosen'])for r in reports])
if __name__=='__main__':main()
