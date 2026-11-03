"""Independent semantic/physical audit; old continuous support vs actual IDs."""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np,torch
from program_discrete_options import prior,normalizer,Learner,legal

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);data,groups=prior(a.prior);report={}
 assert len(data)==118
 for t in data:
  assert 0<=t['a']<(3 if t['stage']==0 else 2)
  assert t['duration']>0 and (t['stage']==0 or t['done'])
  assert abs(t['s'][41]*100-t['next'][41]*100-t['duration'])<1e-4
 report['prior']=dict(transitions=len(data),episodes=len({t['source']for t in data}),scenes=len({t['scene']for t in data}),controller_set=sorted({t['controller']for t in data}))
 # Mask and exact categorical expectation against independent numpy arithmetic.
 x=torch.tensor([data[0]['s'],next(t['s']for t in data if t['stage']==1)]);mask=legal(x);assert mask.tolist()==[[True,True,True],[True,True,False]]
 q=torch.tensor([[.2,.8,.3],[.4,.7,1e6]]);logits=torch.tensor([[1.,2.,3.],[1.,2.,1e6]]);prob=logits.masked_fill(~mask,-1e9).softmax(-1)
 assert prob[1,2]==0
 calc=(prob*(q-.02*prob.clamp_min(1e-9).log())).sum(1).numpy();manual=[]
 for row,n in enumerate([3,2]):
  pp=np.exp(logits[row,:n].numpy()-logits[row,:n].numpy().max());pp/=pp.sum();manual.append(float(np.sum(pp*(q[row,:n].numpy()-.02*np.log(pp)))))
 assert np.max(abs(calc-manual))<1e-6
 report['finite_bellman_mask']='PASS_ILLEGAL_HIGH_Q_AND_LOGIT_EXCLUDED';report['duration_gamma']='gamma=1 finite100 whole-task objective; durations retained and budget included, not per-option gamma<1'
 # Actual old models: continuous off-support target samples and their critic gap.
 import program_contact_sac as sac
 sac.mask=lambda obs:torch.stack([torch.ones_like(obs[:,44]),torch.zeros_like(obs[:,44])],1)
 old=sac.BoundarySAC.load(a.prior/'peg_full_none/SAC1200_f-1s31201.zip',device='cpu');torch.manual_seed(31301);metrics=[]
 for stage in [0,1]:
  obs=np.array([t['s']for t in data if t['stage']==stage]);xx=torch.tensor(np.repeat(obs,64,axis=0),dtype=torch.float32);uu,logp=old.actor.action_log_prob(xx);ap=uu.detach().clone();ap[:,0]=ap[:,0].round()if stage==0 else torch.where(ap[:,0]>=0,1.,-1.)
  q=torch.cat(old.critic_target(xx,uu),1).min(1).values;qp=torch.cat(old.critic_target(xx,ap),1).min(1).values;delta=(q-qp).detach().numpy();mass=np.bincount((ap[:,0]+1).long().numpy(),minlength=3);mass=mass/mass.sum();h=float(-sum(p*np.log(p)for p in mass if p>0))
  per_state_entropy=[]
  projected_ids=(ap[:,0]+1).long().numpy().reshape(-1,64)
  for ids in projected_ids:
   counts=np.bincount(ids,minlength=3)/64;per_state_entropy.append(float(-sum(p*np.log(p)for p in counts if p>0)))
  continuous_h=(-logp.detach().numpy()).reshape(-1,64).mean(1);q_gap=delta.reshape(-1,64).mean(1);backup_gap=q_gap+.02*(continuous_h-np.array(per_state_entropy))
  metrics.append(dict(conditional_option_entropy_mean=float(np.mean(per_state_entropy)),same_actor_projected_backup_gap_mean=float(backup_gap.mean()),same_actor_projected_backup_gap_max_abs=float(abs(backup_gap).max()),per_state_backup_gap=backup_gap.tolist(),stage=stage,samples=len(xx),off_support_fraction=float((abs(uu[:,0]-ap[:,0])>1e-6).float().mean()),mean_abs_q_gap=float(np.mean(abs(delta))),signed_q_gap=float(delta.mean()),max_abs_q_gap=float(abs(delta).max()),mean_continuous_entropy=float(-logp.detach().mean()),pooled_projected_option_entropy=h,projected_mass=mass.tolist(),entropy_warning='pooled option entropy includes between-state variation; not conditional entropy'))
 report['old_continuous']=metrics
 fixture=[]
 for p,folder in zip(sorted((a.root/'fixture/physics').glob('*.json.gz')),['grid_-0.01_nominal','grid_-0.01_refresh']):
  d=json.load(gzip.open(p,'rt'));o=json.load(gzip.open(a.prior/folder/'750103.json.gz','rt'));assert d['steps']==o['steps'] and d['initial']==o['initial'];fixture.append(dict(path=str(p),exact100steps=True,success=d['summary']['success']))
 report['unchanged_physics_fixture']=fixture
 (a.root/'SEMANTICS_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
