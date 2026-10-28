"""Posthoc calibration on predeclared train/calibration cases, not test tuning."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from handoff_learn import dataset,load_prior

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();b,q,h=load_prior(a.root/'prior/prior.pt');rows=dataset(a.root/'opportunity64');metrics={}
 for split,cond in [('fit',lambda s:s<420048),('calibration',lambda s:s>=420048)]:
  r=[r for r in rows if cond(r['seed']) and r['risk_eligible']];x=torch.tensor((np.array([z['obs'] for z in r])-b['mean'])/b['std'],dtype=torch.float32);actions=torch.tensor([z['action'] for z in r]);y=np.array([z['risk'] for z in r]);
  with torch.no_grad():pred=h(x).sigmoid().gather(1,actions[:,None]).squeeze(1).numpy()
  bins=[]
  for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
   mask=(pred>=lo)&(pred<hi if hi<1 else pred<=hi)
   if mask.any():bins.append(dict(lower=float(lo),upper=float(hi),n=int(mask.sum()),predicted=float(pred[mask].mean()),observed=float(y[mask].mean())))
  metrics[split]=dict(n=len(r),positive_labels=int(y.sum()),brier=float(np.mean((pred-y)**2)),constant_prevalence_brier=float(np.mean((y-y.mean())**2)),mean_predicted_risk=float(pred.mean()),observed_prevalence=float(y.mean()),bins=bins)
 a.output.write_text(json.dumps(dict(scope='Posthoc diagnosis, NOT new training or development model selection',metrics=metrics),indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
