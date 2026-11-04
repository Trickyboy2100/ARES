"""Rebuild replay at a saved checkpoint from archived actual local executions.
No simulation and no fabricated transitions. Explicitly records RNG restart.
"""
import argparse,gzip,json,shutil,hashlib
from pathlib import Path
from program_peg_feedback import features,potential
from program_budget_feedback_sac import completion_labels
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--last-scene',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);data=json.loads((a.parent/'TRANSITIONS.json').read_text());recent=[];cost=0;rows=[]
for d in sorted(a.run.glob('capture_*')):
 if int(d.name.split('_')[-1])>a.last_scene:continue
 prefix=json.load(gzip.open(d/'PREFIX.json.gz','rt'));cost+=len(prefix['steps']);assert prefix['scope']=='ACTUAL_REACHABLE_PREFIX'
 for f in sorted(d.glob('suffix_*.json.gz'),key=lambda f:int(f.name.split('_')[1].split('.')[0])):
  z=json.load(gzip.open(f,'rt'));row=z['summary'];full=prefix['steps']+z['steps'];assert len(full)==100;assert len(z['steps'])==row['actual_steps'];cost+=len(z['steps']);rows.append(row);ts=[]
  for i,b in enumerate(row['decisions']):
   end=row['decisions'][i+1]['step']if i+1<len(row['decisions'])else 100;done=i+1==len(row['decisions']);nxt=row['decisions'][i+1]['obs']if not done else features(full[-1],0,1).tolist()
   ts.append(dict(s=b['obs'],next=nxt,a=b['id'],r=(float(row['success'])if done else 0)+.2*((0 if done else potential(full[end-1]))-potential(full[b['step']-1])),done=done,duration=end-b['step'],stage=1,outcome=float(row['success']),scene=row['seed'],source=str(f),controller=json.loads((a.parent/'TRANSITIONS.json').read_text())[0]['controller'],step=b['step'],scope='LOCAL_REACHABLE_HANDOFF'))
  data+=ts;recent.append(completion_labels(dict(summary=dict(success=row['success']),steps=full),ts))
  if not row['state_reset']:
   b=prefix['decisions'][0];data.append(dict(s=b['obs'],next=ts[0]['s'],a=b['id'],r=.2*(potential(prefix['steps'][-1])-potential(prefix['initial'])),done=False,duration=len(prefix['steps']),stage=0,outcome=float(row['success']),scene=row['seed'],source=str(d/'PREFIX.json.gz'),controller=ts[0]['controller'],step=0,scope='ONCE_EXECUTED_PREFIX'))
assert int(a.checkpoint.stem.split('_')[-1])==cost
(a.out/'TRANSITIONS.json').write_text(json.dumps(data));(a.out/'LABELS.json').write_text(json.dumps([z for ll in recent[-48:]for z in ll]));(a.out/'LOCAL_ROWS.json').write_text(json.dumps(rows));shutil.copy2(a.checkpoint,a.out/'FINAL.pt');(a.out/'RECOVERY.json').write_text(json.dumps(dict(checkpoint=str(a.checkpoint),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),actual_training_steps=cost,recovered_rollouts=len(rows),recovered_transitions=len(data),model_optimizer_and_model_rng='exact saved checkpoint',exploration_rng='explicit fresh local-script seed at restart; NOT bit-identical continuation',note='Terminal UnboundLocalError occurred only in attempted zero-step phase after native step100; actual archived100-step outcomes preserved.'),indent=2));print(cost,len(data))
