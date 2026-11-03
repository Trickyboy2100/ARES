"""Post-hoc physical mechanism confirmation, explicitly NOT an unseen evaluation.
Frozen weights; change only finite-Q extraction. No learning or new observations.
"""
import argparse,json,gzip,hashlib,datetime
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner
from program_peg_option_runtime import Runtime
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;out=r/'extraction_eval';out.mkdir(exist_ok=False);torch.set_num_threads(2)
conditions=[(f'{family}_{seed}','mean')for family in ['frontier_current_joint','mature_policy']for seed in [31301,31302]]+[('frontier_current_joint_31301','min'),('mature_policy_31302','min')]
protocol=dict(declared=datetime.datetime.now().isoformat(),scope='POSTHOC_DEVELOPMENT_MECHANISM_CONFIRMATION; all eight scenes already observed; not an independent test or new RL method',changes='policy extraction only; no weights/physics/observation changes',conditions=[dict(run=n,extraction=m,model_sha=hashlib.sha256((r/n/'FINAL.pt').read_bytes()).hexdigest())for n,m in conditions],training_updates=0,physical_episodes=48)
(out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2));panel={}
for row in json.loads((r/'expansion_eval/physics/RESULTS.json').read_text()):
 if row['tag'].startswith('fixed'):
  aa,bb=map(int,row['tag'][-2:]);panel[row['seed'],aa,bb]=row
rt=Runtime(out/'physics');scores=[]
try:
 for name,mode in conditions:
  m=Learner.load(r/name/'FINAL.pt')
  def policy(obs):
   _,q=m.values(obs);n=3 if obs[44]<.5 else 2;v=q[:,0,:n];return int((v.mean(0)if mode=='mean'else v.min(0)).argmax())
  rows=[]
  for scene in range(751220,751228):
   row,ts,d=rt.run(scene,policy,name+'_'+mode);aa=row['decisions'][0]['id'];bb=row['decisions'][1]['id']if len(row['decisions'])>1 else 0;reference=panel[scene,aa,bb];raw=json.load(gzip.open(reference['path'],'rt'));assert d['initial']==raw['initial']and d['steps']==raw['steps'];rows.append(row)
  scores.append(dict(run=name,extraction=mode,success=sum(x['success']for x in rows),n=8,sources=[x['path']for x in rows],evaluation='DIRECT_NATIVE100_POSTHOC_MECHANISM_CONFIRMATION',all100steps_equal=True));(out/'SCORES.json').write_text(json.dumps(scores,indent=2))
finally:rt.close()
