"""Verify training-mined local actions from native starts, not a policy benchmark.
Hindsight selected TRAINING cases explicitly; no claims of held-out generalization.
"""
import argparse,json,gzip,hashlib
from pathlib import Path
from program_peg_feedback import FeedbackRuntime,rule
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
choices=[('ordinary',755008,3),('ordinary',755011,1),('ordinary',755015,1),('budget',755005,1),('budget',755010,7),('suffix',755007,1)];manifest=[]
for mode,seed,j in choices:
 f=a.root/f'local_{mode}_31501/capture_{seed}/suffix_{j}.json.gz';d=json.load(gzip.open(f,'rt'));pf=f.parent/'PREFIX.json.gz';pre=json.load(gzip.open(pf,'rt'));manifest.append(dict(source=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),seed=seed,grasp=pre['decisions'][0]['id'],decisions=d['summary']['decisions'],local_state_restore=d['summary']['state_reset'],scope='HINDSIGHT_TRAINING_ACTION_AUTHORITY_NOT_RL_TEST'))
(a.out/'SELECTION.json').write_text(json.dumps(manifest,indent=2));rt=FeedbackRuntime(a.out/'physics');paired=[]
for case in manifest:
 traces={}
 for mode in ['zero','training_schedule','rule']:
  def policy(s):
   if s[44]<.5:return case['grasp']
   if mode=='zero':return 0
   if mode=='rule':return rule(s)
   step=round(100-s[41]*100)+int(case['local_state_restore']);before=[z for z in case['decisions']if z['step']<=step];return (before[-1]if before else case['decisions'][0])['id']
  row,t,d=rt.run(case['seed'],policy,mode);traces[mode]=d;print(case['seed'],mode,row['success'],flush=True)
 n=traces['zero']['summary']['decisions'][1]['step'];import numpy as np
 err=max(float(np.max(np.abs(np.array(traces['zero']['steps'][i][key])-np.array(traces['training_schedule']['steps'][i][key]))))for i in range(n)for key in ['q','qvel','peg','tcp','action'])
 paired.append(dict(seed=case['seed'],shared_grasp=case['grasp'],prefix_steps=n,prefix_max_difference=err,success={k:v['summary']['success']for k,v in traces.items()}));assert err==0
rt.close();(a.out/'PAIRS.json').write_text(json.dumps(paired,indent=2))
