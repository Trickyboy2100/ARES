"""Numerical action authority and entropy Jacobian check, not physical evidence."""
import json,argparse
from pathlib import Path
import numpy as np,torch,gymnasium as gym
from program_contact_direction import DirectionPolicy
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.manual_seed(17)
effect=dict(mean=[0.]*6,std=[1.]*6,coef=np.r_[np.array([[.02,.04,-.01,.03]]),np.zeros((6,4))].tolist(),global_jacobian=[[.02,.04],[-.01,.03]])
policy=DirectionPolicy(gym.spaces.Box(-np.inf,np.inf,(54,),np.float32),gym.spaces.Box(-1.,1.,(2,),np.float32),lambda _:3e-4,net_arch=[128,128],effect=effect);obs=torch.randn(64,54);obs[:,44]=1.;obs[:,51:53]=obs[:,14:16]+torch.tensor([-.3,.1]);act=torch.rand(64,2)*1.8-.9;mapped,ld=policy.actor.mapped(obs,act);radius_error=(mapped.abs().amax(1)-act.abs().amax(1)).abs().max().item();errors=[]
for i in range(32):
 j=torch.autograd.functional.jacobian(lambda z:policy.actor.mapped(obs[i:i+1],z[None])[0][0],act[i]);errors.append(abs(float(torch.log(torch.det(j).abs()))-float(ld[i])))
obs[:,44]=0;masked=act.clone();masked[:,1]=0;mapped0,ld0=policy.actor.mapped(obs,masked)
r=dict(kind='NUMERICAL_UNIT_CHECK_NOT_PHYSICS',samples=64,square_radius_max_error=radius_error,entropy_logdet_max_error=max(errors),stage0_action_max_error=float((mapped0-masked).abs().max()),stage0_logdet_max_error=float(ld0.abs().max()));assert max(r[k]for k in ['square_radius_max_error','entropy_logdet_max_error','stage0_action_max_error','stage0_logdet_max_error'])<1e-5;r['status']='PASS';a.out.write_text(json.dumps(r,indent=2));print(r)
