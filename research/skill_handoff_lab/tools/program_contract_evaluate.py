"""Checkpoint and comparator actual physical evaluation on common new design cases."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from program_tool_smdp_sac import SkillEnv,SAC
from program_contract_comparators import Policies

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--sac',type=Path,required=True);p.add_argument('--comparators',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=list(range(746200,746208)));p.add_argument('--methods',nargs='+',default=['fixed105','fixed070','sac256','sac512','sac1024','supervised','stap_open','stap_replan','downstream']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);models={};other=Policies(a.comparators)if a.comparators else None;allrows=[]
 for method in a.methods:
  model=SAC.load(a.sac/f'SAC_{method[3:]}.zip',device='cpu')if method.startswith('sac')else None
  env=SkillEnv(a.out/method,a.seeds)
  try:
   for seed in a.seeds:
    obs,_=env.reset();planned=None;logs=[];done=False
    while not done:
     stage=env.stage
     if model is not None:action=model.predict(obs,deterministic=True)[0];log={}
     elif method.startswith('fixed'):action=np.array([1. if method=='fixed105' else 0.])if stage==0 else np.array([-1.]);log={}
     else:
      value,log=other.choose(method,obs,stage,planned);planned=log.get('planned',planned);action=np.array([value])
     logs.append(dict(stage=stage,action=action.tolist(),observation=obs.tolist(),**log));obs,_,done,_,_=env.step(action)
    row=dict(env.rows[-1],method=method);allrows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(allrows,indent=2));(a.out/method/f'{seed}_POLICY.json').write_text(json.dumps(logs,indent=2))
  finally:env.close()
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(seeds=a.seeds,split='NEW_DEVELOPMENT_NOT_SEALED',common_controller='corrected_feedback',methods=a.methods,sac=str(a.sac),comparators=str(a.comparators),physical_steps=100*len(allrows)),indent=2));shutil.copy2(__file__,a.out/Path(__file__).name)
if __name__=='__main__':main()
