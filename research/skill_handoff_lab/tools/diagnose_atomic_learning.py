"""Frozen-policy values at an actual reached rescue state, and support counts."""
import argparse,json,gzip,collections,torch
from pathlib import Path
from program_budget_feedback_sac import FeedbackSAC
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
h=json.loads((a.root/'macro_bc64/HISTORY.json').read_text());row=next(v['row']for v in h if v['job']==dict(kind='eval',checkpoint=32,seed=761202));d=json.load(gzip.open(row['path'],'rt'));s=next(x['obs']for x in d['summary']['decisions']if x['step']==63);values=[]
for arm in ['queue_bc64','phase_bc64','macro_bc64']:
 for c in [0,32,64]:
  path=a.root/'warm31701/INITIAL.pt'if c==0 else a.root/arm/f'CHECKPOINT_{c}.pt'
  if not path.exists():continue
  m=FeedbackSAC.load(path);x=m.x([s])
  with torch.no_grad():prob=m.probs(x)[0].numpy();q=torch.stack([v(x)for v in m.q]).mean(0)[0].numpy()
  values.append(dict(arm=arm,checkpoint=c,actor=prob.tolist(),mean_q=q.tolist(),chosen=int(prob.argmax())))
support={}
for arm in ['queue_bc64','phase_bc64','macro_bc64']:
 pointer=json.loads((a.root/arm/'CURRENT.json').read_text());d=torch.load(a.root/arm/pointer['checkpoint'],weights_only=False);z=d['data'];counts={}
 for t in z:
  if t['stage']==0:continue
  key=str(t['a']);c=counts.setdefault(key,dict(n=0,successful_episode=0,duration=0));c['n']+=1;c['successful_episode']+=bool(t['outcome']);c['duration']+=t['duration']
 support[arm]=counts
(a.root/'LEARNING_DIAGNOSIS.json').write_text(json.dumps(dict(actual_state_source=row['path'],seed=761202,step=63,scope='posthoc actual-state policy/value comparison; Q is not calibrated probability; successful episode is NOT intervention causal credit',values=values,option_training_support=support),indent=2));print([(v['arm'],v['checkpoint'],v['chosen'])for v in values])
