"""Actual deployment divergences and corrected backup activity, not new trials."""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np,torch
from suffix_trace_learner import Learner
from program_rule_option import Assets
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=int,default=32);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
h=json.loads((a.run/'HISTORY.json').read_text());base={r['row']['seed']:r for r in h if r['kind']=='eval'and r['checkpoint']==0};later={r['row']['seed']:r for r in h if r['kind']=='eval'and r['checkpoint']==a.checkpoint};rows=[]
for seed,b in base.items():
 if seed not in later:continue
 c=later[seed];bp=Path(b['row']['path']);cp=Path(c['row']['path']);bd=json.load(gzip.open(bp,'rt'));cd=json.load(gzip.open(cp,'rt'));common=0
 for x,y in zip(bd['steps'],cd['steps']):
  if not all(x[k]==y[k]for k in ['action','q','qvel','tcp','peg']):break
  common+=1
 be=[z for z in b['events']if z['boundary']];ce=[z for z in c['events']if z['boundary']]
 rows.append(dict(seed=seed,before=b['row']['success'],after=c['row']['success'],recorded_prefix_steps=common,before_boundaries=be,after_boundaries=ce,paths=[str(bp),str(cp)],sha=[hashlib.sha256(p.read_bytes()).hexdigest()for p in [bp,cp]]))
m=Learner(Assets(),32401,'trace');d=torch.load(a.run/'CURRENT.pt',weights_only=False);frozen=torch.load(a.run/f'CHECKPOINT_{a.checkpoint}.pt',weights_only=False);m.load(frozen);data=[t for t in d['data']if 'mu'not in t or t['policy_episode']<a.checkpoint];ix=[i for i,t in enumerate(data)if 'mu'in t]
m.mode='trace';yt,diag=m.targets(data,ix);m.mode='sac';ys,_=m.targets(data,ix)
s=dict(scope='same final model/replay target ablation; actual dev before-after traces; no new physical execution',checkpoint=a.checkpoint,actual_model_updates=frozen['updates'],n_own_decisions=len(ix),target_difference_mean=float((yt-ys).mean()),target_difference_abs_mean=float((yt-ys).abs().mean()),trace_activity=diag,paired_trajectories=rows)
a.out.write_text(json.dumps(s,indent=2));print({k:v for k,v in s.items()if k!='paired_trajectories'})
