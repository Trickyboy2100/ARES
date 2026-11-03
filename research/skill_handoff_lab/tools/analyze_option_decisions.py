"""Full finite physical-panel attribution. Logged counterfactual diagnostic only."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
from program_discrete_options import Learner,episode_transitions
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;e=root/'expansion_eval';pred=json.loads((e/'BRANCH_PREDICTIONS.json').read_text());manifest=json.loads((e/'FROZEN_MODELS.json').read_text());rows=json.loads((e/'physics/RESULTS.json').read_text());panel={}
extra=root/'anchor_eval';pred+=json.loads((extra/'BRANCH_PREDICTIONS.json').read_text());manifest['models']+=json.loads((extra/'FROZEN_MODELS.json').read_text())['models']
for row in rows:
 if not row['tag'].startswith('fixed'):continue
 aa,bb=map(int,row['tag'][-2:]);d=json.load(gzip.open(row['path'],'rt'));ts=episode_transitions(d,[-.01,0,.01][aa],bb,row['path']);panel[row['seed'],aa,bb]=(row,ts)
report=[]
for model in manifest['models']:
 key=(model['method'],model['seed']);selected=[z for z in pred if(z['method'],z['seed'])==key];dd=[];m=None if key[0]=='projected'else Learner.load(model['path'])
 for z in selected:
  scene=z['scene'];aa=z['actions'][0];bb=z['actions'][1]if len(z['actions'])>1 else 0;matrix=np.array([[int(panel[scene,i,j][0]['success'])for j in range(2)]for i in range(3)]);union=bool(matrix.max());first_viable=bool(matrix[aa].max());out=dict(scene=scene,selected=z['actions'],success=z['success'],measured_matrix=matrix.tolist(),failure_class='SUCCESS'if z['success']else'NO_MEASURED_LEGAL_SOLUTION'if not union else'PREDECESSOR_OPTION_SELECTION'if not first_viable else'SUCCESSOR_OPTION_SELECTION')
  if m is not None:
   s0=panel[scene,0,0][1][0]['s'];_,q=m.values(s0);q0=q[:,0,:3].mean(0);qa=int(q0.argmax());ts=panel[scene,qa,0][1];qb=0
   if len(ts)>1:
    _,q=m.values(ts[1]['s']);qb=int(q[:,0,:2].mean(0).argmax())
   out.update(critic_greedy_actions=[qa,qb]if len(ts)>1 else[qa],critic_greedy_logged_success=panel[scene,qa,qb][0]['success'],q0=q0.tolist())
  dd.append(out)
 report.append(dict(method=key[0],seed=key[1],counts={k:sum(x['failure_class']==k for x in dd)for k in ['SUCCESS','NO_MEASURED_LEGAL_SOLUTION','PREDECESSOR_OPTION_SELECTION','SUCCESSOR_OPTION_SELECTION']},critic_greedy_success=sum(x.get('critic_greedy_logged_success',False)for x in dd)if m else None,rows=dd))
(root/'DECISION_ATTRIBUTION.json').write_text(json.dumps(dict(scope='Full real finite action panel; counterfactual logged diagnostics, not extra policy rollouts; best measured action is hindsight oracle',models=report),indent=2));print('Attributed',len(report),'models')
