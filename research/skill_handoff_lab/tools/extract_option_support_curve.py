"""Actual saved V0 critic on a measured handoff, not an invented physics curve."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from program_discrete_options import prior
import program_contact_sac as sac

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);_,groups=prior(a.prior);g=next(g for g in groups if g['seed']==750103 and g['grasp']==-.01);obs=g['items'][0][2][1][0]
 sac.mask=lambda obs:torch.stack([torch.ones_like(obs[:,44]),torch.zeros_like(obs[:,44])],1);path=a.prior/'peg_full_none/SAC1200_f-1s31201.zip';m=sac.BoundarySAC.load(path,device='cpu');u=np.linspace(-1,1,201,dtype=np.float32);acts=torch.tensor(np.c_[u,np.zeros_like(u)]);x=torch.tensor(np.repeat(obs[None],len(u),axis=0));ap=acts.clone();ap[:,0]=torch.where(acts[:,0]>=0,1.,-1.)
 with torch.no_grad():q=torch.cat(m.critic(x,acts),1).min(1).values.numpy();qp=torch.cat(m.critic(x,ap),1).min(1).values.numpy()
 d=dict(scene=750103,grasp=-.01,model=str(path),model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=g['items'][0][0]['path'],obs=obs.tolist(),continuous_requested=u.tolist(),critic=q.tolist(),critic_at_executed_option=qp.tolist(),actual_discrete_outcomes=[int(r['success'])for r,d,t in g['items']],actor_request=m.predict(obs,deterministic=True)[0].tolist(),scope='network query only; actual physical outcomes exist only at two options; no claim this example causes V0 failure')
 a.out.write_text(json.dumps(d,indent=2))
if __name__=='__main__':main()
