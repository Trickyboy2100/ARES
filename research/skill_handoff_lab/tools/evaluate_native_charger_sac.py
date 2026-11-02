"""Native200 complete-task comparison; identical observable controls and authority."""
import argparse,json,hashlib,time,shutil
from pathlib import Path
import numpy as np
from program_native_charger_sac import NativeEnv,BoundarySAC
from program_native_charger_pace import action_for

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--controls',nargs='+',default=['nominal','feedback']);p.add_argument('--models',type=Path,nargs='*',default=[]);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);allrows=[]
 models={('sparse_' if 'sparse'in p.parent.name else 'dense_')+p.stem.split('_')[-1]:BoundarySAC.load(p,device='cpu')for p in a.models};last=next(reversed(models),None);methods=[*a.controls,*models,*(['matched_'+last]if last else [])]
 (a.out/'PROTOCOL.json').write_text(json.dumps(dict(task='PlugCharger-v1',horizon=200,seeds=a.seeds,split='NEW_DEVELOPMENT_NOT_SEALED',methods=methods,models={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in a.models},matched='force same15deg grasp, then trained successor policy; verify actual prefix; full policy learns15..45deg grasp and pace',source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));shutil.copy2(__file__,a.out/Path(__file__).name)
 for name in methods:
  env=NativeEnv(a.out/name,a.seeds)
  try:
   for seed in a.seeds:
    obs,_=env.reset();assert env.case==seed;done=False;decisions=[];t=time.monotonic()
    while not done:
     if env.stage==0 and(name in a.controls or name.startswith('matched_')):action=np.array([-1.,0.])
     elif name in a.controls:action=action_for(name,env.ep.observation())
     else:action=models[name.removeprefix('matched_')].predict(obs,deterministic=True)[0]
     decisions.append(dict(stage=env.stage,step=len(env.ep.env.records),action=np.asarray(action).tolist()));obs,_,done,_,_=env.step(action)
    row=dict(env.rows[-1],method=name,selections=decisions,evaluation_wall_s=time.monotonic()-t);allrows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(allrows,indent=2));print(name,seed,row['success'],row['angle'],row['error'],flush=True)
  finally:env.close()
if __name__=='__main__':main()
