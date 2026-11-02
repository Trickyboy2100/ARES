"""New-design (not sealed) complete native100 comparator evaluation."""
import argparse,json,time,random,gzip,hashlib,shutil
from pathlib import Path
import numpy as np,torch
import program_contact_sac as sac
from program_contact_direction import DirectionPolicy,DirectionActor,transitions
from program_contact_auxiliary import GenericAuxSAC
from program_contact_comparators import Selector
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--methods',default='fixed105,fixed70,feedback,ordinary,conditional,global,generic_aux,supervised,stap');p.add_argument('--checkpoint',default='256');p.add_argument('--fixed-grasp',type=float);p.add_argument('--seeds',type=int,nargs='+',default=list(range(748100,748108)));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);rows=[]
for name in ['evaluate_contact_methods.py','program_contact_direction.py','program_contact_auxiliary.py','program_contact_sac.py','program_post_contact.py','program_contact_comparators.py']:shutil.copy2(Path(__file__).with_name(name),a.out/name)
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(seeds=a.seeds,methods=a.methods,checkpoint=a.checkpoint,fixed_grasp=a.fixed_grasp,native_horizon=100,task='PullCubeTool-v1',scope='new development scenes; not sealed, no claim from individual wins'),indent=2))
for name in a.methods.split(','):
 env=sac.ContactEnv(a.out/name,a.seeds,'relative_yaw');model=None;selector=None;model_path=None
 if name=='lookahead':env.geometry_grasp_search=True;env.env.grasp_candidates=[.035,.0525,.07,.0875,.105]
 if name in ['ordinary','conditional','global','generic_aux']:
  model_path=a.root/f'sac_{name}'/('OFFLINE.zip'if a.checkpoint=='OFFLINE'else f'SAC_{a.checkpoint}.zip');model=(GenericAuxSAC if name=='generic_aux'else sac.BoundarySAC).load(model_path,device='cpu')
 if name in ['supervised','stap']:selector=Selector(a.root/'comparators/MODEL.pt',name)
 try:
  for seed in a.seeds:
   random.seed(31191);np.random.seed(31191);torch.manual_seed(31191);obs,_=env.reset();done=False;selection=[]
   while not done:
    t=time.monotonic()
    if model is not None:action=model.predict(obs,deterministic=True)[0]
    elif selector is not None:action=selector.choose(obs)
    elif env.stage==0:action=np.array([0. if name=='fixed70'else .5 if name=='fixed875'else 1.,0.])
    else:
     action=np.zeros(2)
     if name=='feedback':
      from program_contact_direction import physical_features
      f=physical_features(obs)
      # Same actual state, conservative native budget. Fixed after design, no development labels.
      if obs[41]>=.25 and f[2]>-.25:action=np.array([0.,1.])
    if env.stage==0 and a.fixed_grasp is not None:action=np.array([(a.fixed_grasp-.07)/.035,0.])
    selection.append(dict(stage=env.stage,action=np.asarray(action).tolist(),selection_wall_s=time.monotonic()-t,model_query=None if selector is None else selector.logs[-1]));obs,reward,done,_,info=env.step(action)
   row=dict(env.rows[-1],method=name,selection=selection,model_sha256=None if model_path is None else hashlib.sha256(model_path.read_bytes()).hexdigest());rows.append(row);(a.out/'RESULTS.json').write_text(json.dumps(rows,indent=2));print(name,seed,row['success'],row['error'],flush=True)
 finally:env.close()
