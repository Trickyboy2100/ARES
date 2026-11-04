"""Separate Actor approximation, aggregation and true legal-action value error."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from program_discrete_options import Learner
from eval_option_improvement import panel,evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2);pp=panel(a.panel);scenes=sorted({s for s,aa,b in pp});out=[]
 for seed in [31301,31302]:
  m=Learner.load(a.old/f'mature_policy_{seed}/FINAL.pt');records=[]
  for scene in scenes:
   for first in range(3):
    ts=pp[scene,first,0][1]
    for stage in range(len(ts)):
     if stage==0 and first!=0:continue
     o=ts[stage]['s'];v,q=m.values(o);n=3 if stage==0 else 2;q=q[:,0,:n];mean=q.mean(0);mini=q.min(0);actor=int(v[0,:n].argmax());ma=int(mean.argmax());mi=int(mini.argmax())
     y=np.array([any(pp[scene,aa,b][0]['success']for b in range(2))for aa in range(3)]if stage==0 else[pp[scene,first,b][0]['success']for b in range(2)],float)
     soft=np.exp((mini-mini.max())/m.alpha);soft/=soft.sum()
     policy_truth=y.copy()
     if stage==0:
      for aa in range(3):
       child=pp[scene,aa,0][1]
       if len(child)==1:continue
       with torch.no_grad():probs=m.probs(m.x(np.atleast_2d(child[1]['s'])))[0,:2].numpy()
       yy=np.array([pp[scene,aa,b][0]['success']for b in range(2)],float)
       policy_truth[aa]=float(probs@yy-m.alpha*np.sum(probs*np.log(np.maximum(probs,1e-9))))
     records.append(dict(scene=scene,first=first,stage=stage,remaining=o[41]*100,grasped=o[42],actual=y.tolist(),actor_soft_policy_truth=policy_truth.tolist(),mean_bias_to_actor_soft_policy=float((mean-policy_truth).mean()),q=q.tolist(),actor=v[0,:n].tolist(),actor_vs_softmin_L1=float(abs(v[0,:n]-soft).sum()),actor_disagrees_min=actor!=mi,min_disagrees_mean=mi!=ma,actor_regret=float(y.max()-y[actor]),min_regret=float(y.max()-y[mi]),mean_regret=float(y.max()-y[ma]),mean_minus_measured_optimal=float((mean-y).mean()),mean_mse_to_measured_optimal=float(((mean-y)**2).mean())))
  out.append(dict(seed=seed,scene_count=len(scenes),full_task={rule:evaluate(m,pp,rule)for rule in ['deployed','min','mean']},records=records))
 a.out.write_text(json.dumps(out,indent=2));print(json.dumps([dict(seed=x['seed'],scenes=x['scene_count'],success={k:sum(y['success']for y in v)for k,v in x['full_task'].items()})for x in out]))
if __name__=='__main__':main()
