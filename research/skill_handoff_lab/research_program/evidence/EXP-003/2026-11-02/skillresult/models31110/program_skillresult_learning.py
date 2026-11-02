"""Bounded SkillResult pilot. Actual terminal contracts, paired-action ranking.
No altered reward: Q is undiscounted terminal full-task success. Auxiliary labels
are measured outcomes used only in training; no future outcome is observed online.
"""
import argparse,gzip,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from program_stap_learning import collect,train,Model,Net

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=1200);a=p.parse_args();torch.set_num_threads(4)
 train(a.root,a.out,31110,a.updates,['fq','foresight'],True)
 rows=collect(a.root);valid=[r for r in rows if r['seed'] in [743046,743047]]
 # Base training root excludes calibration scenes; calibration passed separately.
 calroot=a.root.parent/'calibration';valid=collect(calroot)
 rows=[r for r in rows if r['has_suffix']];valid=[r for r in valid if r['has_suffix']]
 model=Model.load(a.out/'MODEL.pt')
 def x(rs):return torch.cat([model.norm(torch.tensor([r['s1'] for r in rs])),torch.tensor([[r['a1']] for r in rs])],-1)
 X=x(rows);V=x(valid);Y=torch.tensor([[r['r1']] for r in rows]);VY=np.array([r['r1'] for r in valid])
 contract=[];generic=[];groups={}
 for i,r in enumerate(rows):
  d=json.load(gzip.open(r['path'],'rt'));end=d['steps'][-1];pull=[s['step'] if 'step' in s else k+1 for k,s in enumerate(d['steps']) if s['phase']=='PULL']
  contract.append([(.6-end['cube_base_distance_m'])/.1,(.04-d['summary']['target_distance_m'])/.1,(100-max(pull))/100 if pull else -1,float(end['grasped'])])
  generic.append([*end['cube'][:3],end['tool'][0]])
  key=(r['seed'],r['density'],r['goal'],round(r['a0'],5));groups.setdefault(key,[]).append(i)
 pairs=[(i,j) for ids in groups.values() for i in ids for j in ids if i<j and rows[i]['r1']!=rows[j]['r1'] and rows[i]['s1']==rows[j]['s1']]
 pi=torch.tensor([i for i,j in pairs],dtype=torch.long);pj=torch.tensor([j for i,j in pairs],dtype=torch.long)
 payload={};metrics={};bce=nn.BCEWithLogitsLoss()
 for name in ['ordinary','generic','contract','relative','relative_contract']:
  torch.manual_seed(31111);net=Net(46,5);opt=torch.optim.Adam(net.parameters(),lr=.001)
  aux=torch.tensor(generic if name=='generic' else contract);am=aux.mean(0);asd=aux.std(0).clamp(min=.05);aux=(aux-am)/asd
  for k in range(a.updates):
   z=net(X);loss=bce(z[:,:1],Y)
   if name in ['generic','contract','relative_contract']:loss=loss+.1*((z[:,1:]-aux)**2).mean()
   if name.startswith('relative') and pairs:loss=loss+.2*bce((z[pi,0]-z[pj,0])[:,None],(Y[pi]>Y[pj]).float())
   opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():logits=net(V)[:,0].numpy()
  ts=np.geomspace(.25,32,81);briers=[float(np.mean((1/(1+np.exp(-np.clip(logits/t,-80,80)))-VY)**2)) for t in ts];temp=float(ts[np.argmin(briers)])
  payload[name]=dict(weights=net.state_dict(),temperature=temp,aux_mean=am.tolist(),aux_std=asd.tolist())
  metrics[name]=dict(calibration_brier_before=float(np.mean((1/(1+np.exp(-np.clip(logits,-80,80)))-VY)**2)),calibration_brier_after=min(briers),temperature=temp,calibration_predictions=(1/(1+np.exp(-np.clip(logits/temp,-80,80)))).tolist(),calibration_labels=VY.tolist(),training_loss=float(loss))
 ensemble=[]
 for boot in range(3):
  torch.manual_seed(31200+boot);rng=np.random.default_rng(31200+boot);seeds=sorted(set(r['seed'] for r in rows));chosen=rng.choice(seeds,len(seeds),replace=True);ids=torch.tensor([i for seed in chosen for i,r in enumerate(rows) if r['seed']==seed])
  net=Net(46,5);opt=torch.optim.Adam(net.parameters(),lr=.001)
  for k in range(a.updates):
   loss=bce(net(X[ids])[:,:1],Y[ids]);opt.zero_grad();loss.backward();opt.step()
  ensemble.append(net.state_dict())
 torch.save(dict(ensemble=ensemble,models=payload,mean=model.mean.tolist(),std=model.std.tolist()),a.out/'SUCCESSOR.pt')
 report=dict(training_suffixes=len(rows),calibration_suffixes=len(valid),calibration_scenes=[743046,743047],paired_discordant_actions=len(pairs),pairing='Exact equality of recorded successor feature vector within scene/mass/goal/grasp; no unmatched contrast labels',extra_optimizer_updates=8*a.updates,metrics=metrics,limitations='Single training seed. Temperature calibrated on held-out scenes; calibration-set metrics are fitting diagnostics, not held-out test. Terminal supervised Q and terminal fitted Q are mathematically equivalent; ordinary is both. All second-stage heads have identical capacity. Relative ranking and contract auxiliary are separately ablated; no originality claim.',train_paths=[r['path'] for r in rows],calibration_paths=[r['path'] for r in valid])
 (a.out/'SUCCESSOR_TRAINING.json').write_text(json.dumps(report,indent=2));(a.out/Path(__file__).name).write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items() if k not in ['metrics','train_paths','calibration_paths']}))
if __name__=='__main__':main()
