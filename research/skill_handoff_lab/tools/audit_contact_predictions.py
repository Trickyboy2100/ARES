"""Compare saved model values against actual native outcomes; never feed labels to actors."""
import argparse,json,gzip
from pathlib import Path
import numpy as np,torch
from program_contact_direction import sac,DirectionPolicy,transitions
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;torch.set_num_threads(4);rows=[]
for folder,ckpt in [('eval_offline','OFFLINE'),('eval_online128','SAC_128'),('eval_online256','SAC_256')]:
 cases=json.load(open(root/folder/'RESULTS.json'))
 for name in ['ordinary','conditional']:
  model=sac.BoundarySAC.load(root/f'sac_{name}'/f'{ckpt}.zip',device='cpu')
  for r in cases:
   if r['method']!=name:continue
   d=json.load(gzip.open(r['path'],'rt'))
   for stage,t in enumerate(transitions(d)):
    x,z,u,reward,done=t
    with torch.no_grad():q=[float(v[0,0])for v in model.critic(torch.tensor(x[None]),torch.tensor(np.array(u)[None],dtype=torch.float32))]
    rows.append(dict(cohort=folder,method=name,seed=r['seed'],stage=stage,q_min=min(q),q_values=q,native_success=r['success'],definition='stage0 soft option value; stage1 terminal-return estimate; network is unbounded, not a calibrated probability'))
summary=[]
for folder in ['eval_offline','eval_online128','eval_online256']:
 for name in ['ordinary','conditional']:
  rr=[x for x in rows if x['cohort']==folder and x['method']==name and x['stage']==1];summary.append(dict(cohort=folder,method=name,actual_contact_cases=len(rr),terminal_value_mse=float(np.mean([(x['q_min']-x['native_success'])**2 for x in rr])),out_of_probability_range=sum(not 0<=x['q_min']<=1 for x in rr)))
stap=[]
for r in json.load(open(root/'eval_offline/RESULTS.json')):
 if r['method']=='stap':
  for s in r['selection']:stap.append(dict(seed=r['seed'],stage=s['stage'],predicted_probability=s['model_query']['p_success'],actual_native_success=r['success'],model_queries=s['model_query']['skill_model_queries']))
(root/'analysis/PREDICTIONS_VS_PHYSICS.json').write_text(json.dumps(dict(sac_summary=summary,sac_rows=rows,stap_rows=stap,stap_full_task_Brier=float(np.mean([(x['predicted_probability']-x['actual_native_success'])**2 for x in stap if x['stage']==0])),scope='New development cohort only; n8 scene clusters, not calibrated or sealed-test evidence'),indent=2));print(summary)
