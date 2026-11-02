"""Recompute posterior Q and physical evidence; no simulation or model fitting."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import torch
from program_stap_learning import Model

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);m=Model.load(a.model);rows=[]
 for path in sorted(a.root.glob('*_stap/TRAJECTORY.json.gz')):
  d=json.load(gzip.open(path,'rt'));o=json.load(gzip.open(path.parent.with_name(path.parent.name+'_open')/'TRAJECTORY.json.gz','rt'))
  if len(d['selector_logs'])<2:continue
  log=d['selector_logs'][1];obs=torch.tensor(log['observation']).reshape(1,-1);grid=torch.linspace(-1,1,81).reshape(-1,1)
  with torch.no_grad():q=m.q(1,obs.repeat(81,1),grid).numpy();prior=float(o['selector_logs'][0]['planned_actions'][1][0]);qp=float(m.q(1,obs,torch.tensor([[prior]]))[0])
  prefix=log['step'];assert d['steps'][:prefix]==o['steps'][:prefix]
  row=dict(case=path.parent.name,open_success=o['summary']['task_success'],posterior_success=d['summary']['task_success'],prefix_identical=True,boundary_step=prefix,remaining_steps=100-prefix,prior_hook_mm=(-.065+.01*prior)*1000,posterior_hook_mm=(-.065+.01*log['normalized_action'])*1000,prior_predicted_sequence_success=o['selector_logs'][0]['p_success'],posterior_q_retained=qp,posterior_q_selected=log['p_success'],grid_hook_mm=((- .065+.01*grid[:,0].numpy())*1000).tolist(),grid_q=q.tolist(),actual_error=d['summary']['error'],open_error=o['summary']['error'],posterior_final_phase=d['steps'][-1]['phase'],open_final_phase=o['steps'][-1]['phase'],posterior_target_error_m=d['summary']['target_distance_m'],open_target_error_m=o['summary']['target_distance_m'],planning=[{k:v for k,v in x.items() if k!='position'} for x in d['plans'] if x.get('phase') in ['SEARCH','CONTRACT_BUDGET_GUARD']])
  predicted=np.array(d['selector_logs'][0]['predicted_states'][1]);actual=obs.numpy()[0]
  row['predicted_vs_actual']={name:float(np.linalg.norm(predicted[lo:hi]-actual[lo:hi])) for name,lo,hi in [('q_rad',0,7),('tool_xyz_m',14,17),('cube_xyz_m',21,24),('attachment_xyz_m',28,31),('velocity',35,41),('budget_fraction',41,42)]}
  row['native_success']={'open':o['summary']['success'],'posterior':d['summary']['success']}
  row['phase_steps']={name:{phase:sum(s['phase']==phase for s in tr['steps']) for phase in set(s['phase'] for s in tr['steps'])} for name,tr in [('open',o),('posterior',d)]}
  rows.append(row)
 (a.out/'REGRESSIONS.json').write_text(json.dumps(rows,indent=2))
 for x in rows:
  if x['open_success'] and not x['posterior_success']:print({k:v for k,v in x.items() if k not in ['grid_hook_mm','grid_q','planning']})
if __name__=='__main__':main()
