"""Predeclared fresh development execution of same-data models; native100."""
import argparse,json,time,random,hashlib,shutil
from pathlib import Path
import numpy as np,torch
import program_contact_sac as sac
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,a.out/'SOURCE.py');torch.set_num_threads(2);seeds=list(range(748200,748206));methods=[('none',200),('paired',200),('absolute',200),('shuffled',200),('none',1200),('paired',1200),('fixed875',0)];rows=[]
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(seeds=seeds,split='NEW_DEVELOPMENT_NOT_SEALED',methods=methods,task='PullCubeTool-v1',horizon=100,action='same full continuous grasp and postcontact lateral/yaw permissions, unlike closed-grid diagnostic',training_seed=31201,shared_controller='unchanged post_contact relative_yaw'),indent=2))
for method,checkpoint in methods:
 name=f'{method}_{checkpoint}';env=sac.ContactEnv(a.out/name,seeds,'relative_yaw');path=None if method=='fixed875'else a.root/f'full_{method}'/f'SAC{checkpoint}_f-1s31201.zip';model=None if path is None else sac.BoundarySAC.load(path,device='cpu')
 try:
  for seed in seeds:
   random.seed(31211);np.random.seed(31211);torch.manual_seed(31211);obs,_=env.reset();done=False;selection=[]
   while not done:
    t=time.monotonic();action=model.predict(obs,deterministic=True)[0]if model else np.array([.5,0.])if env.stage==0 else np.zeros(2);selection.append(dict(stage=env.stage,action=action.tolist(),wall_s=time.monotonic()-t));obs,reward,done,_,info=env.step(action)
   row=dict(env.rows[-1],method=name,selection=selection,model_sha=None if path is None else hashlib.sha256(path.read_bytes()).hexdigest());rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(name,seed,row['success'],row['error'],flush=True)
 finally:env.close()
