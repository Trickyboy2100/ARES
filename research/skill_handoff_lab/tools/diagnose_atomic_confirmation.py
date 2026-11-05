"""Frozen actor/critic disagreement on actual confirmation interventions.
Uses own training replay only; no confirmation labels enter any optimizer.
"""
import argparse,json,torch,numpy as np
from pathlib import Path
from program_budget_feedback_sac import FeedbackSAC
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);out=[]
for name,cp in [('macro_bc64',32),('macro_seed31702',64)]:
 model=FeedbackSAC.load(a.root/name/f'CHECKPOINT_{cp}.pt');pointer=json.loads((a.root/name/'CURRENT.json').read_text());data=torch.load(a.root/name/pointer['checkpoint'],weights_only=False)['data'];data=[t for t in data if t['stage']==1 and int(t['scene'])<761016+cp]
 rows=json.loads((a.root/('confirm_'+name)/'RESULTS.json').read_text())
 for row in rows:
  if row['seed']not in [762201,762204,762206,762213,762214]:continue
  actions=[x for x in row['decisions']if x['stage']==1 and x['id']!=0]
  if not actions:continue
  event=actions[0];x=model.x([event['obs']]);tx=model.x([t['s']for t in data]);dist=((tx-x)**2).mean(1).sqrt();j=int(dist.argmin())
  with torch.no_grad():prob=model.probs(x)[0];q=torch.stack([v(x)for v in model.q]).mean(0)[0]
  out.append(dict(run=name,checkpoint=cp,seed=row['seed'],success=row['success'],step=event['step'],selected=event['id'],actor_p=prob.tolist(),mean_q=q.tolist(),q_argmax=int(q.argmax()),selected_vs_continue_q=float(q[event['id']]-q[0]),nearest_training_rms=float(dist[j]),nearest_scene=data[j]['scene'],nearest_action=data[j]['a'],nearest_episode_success=data[j]['outcome'],scope='descriptive normalized state proximity; Q not a probability; nearest outcome not causal credit'))
(a.root/'CONFIRMATION_VALUES.json').write_text(json.dumps(out,indent=2));print([(x['run'],x['seed'],x['selected'],x['q_argmax'],round(x['selected_vs_continue_q'],3),x['success'])for x in out])
