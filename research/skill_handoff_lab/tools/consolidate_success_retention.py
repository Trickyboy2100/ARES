"""Matched additional off-policy updates on already collected real trajectories.
No synthetic transitions, model calls or extra physical training queries.
"""
import argparse,json,shutil,hashlib,time,collections
from pathlib import Path
import torch
from program_budget_feedback_sac import FeedbackSAC
from program_peg_feedback import FeedbackRuntime
p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=4000);p.add_argument('--success-bc',type=float,default=0);p.add_argument('--anchor-mode',choices=['none','all_success','nominal_success'],default='none');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
m=FeedbackSAC.load(a.parent/'FINAL.pt');d=json.loads((a.parent/'TRANSITIONS.json').read_text());l=json.loads((a.parent/'LABELS.json').read_text());rt=FeedbackRuntime(a.out/'physics')
(a.out/'PROTOCOL.json').write_text(json.dumps(dict(parent=str(a.parent),parent_sha=hashlib.sha256((a.parent/'FINAL.pt').read_bytes()).hexdigest(),data_sha=hashlib.sha256((a.parent/'TRANSITIONS.json').read_bytes()).hexdigest(),updates=a.updates,success_bc=a.success_bc,cql=0,successor_only=True,predecessor=1,new_physical_training_steps=0),indent=2))
anchor=None
if a.anchor_mode!='none':
 groups=collections.defaultdict(list)
 for t in d:groups[t['source']].append(t)
 anchor=[t for ts in groups.values() if ts[0]['outcome'] and (a.anchor_mode=='all_success' or all(z['a']==0 for z in ts if z['stage']==1)) for t in ts if t['stage']==1]
 (a.out/'BC_ANCHOR.json').write_text(json.dumps(anchor))
 protocol=json.loads((a.out/'PROTOCOL.json').read_text());protocol.update(anchor_mode=a.anchor_mode,anchor_rows=len(anchor),anchor_episodes=len(set(t['source']for t in anchor)),anchor_sha=hashlib.sha256((a.out/'BC_ANCHOR.json').read_bytes()).hexdigest());(a.out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
src=a.out/'source';src.mkdir()
for name in [Path(__file__).name,'program_budget_feedback_sac.py','program_peg_feedback.py','program_peg_option_runtime.py']:shutil.copy2(Path(__file__).parent/name,src/name)
for n in range(0,a.updates,1000):
 m.train(d,l,min(1000,a.updates-n),success_bc=a.success_bc,cql=0,successor_only=True,bc_anchor=anchor);m.save(a.out/f'CHECKPOINT_{min(n+1000,a.updates)}.pt');print('UPDATE',m.updates,flush=True)
rows=[]
for seed in range(753200,753216):
 row,_,_=rt.run(seed,lambda s:1 if s[44]<.5 else m.choose(s),'consolidated');rows.append(row);print('EVAL',seed,row['success'],flush=True)
(a.out/f'EVAL_{a.updates}.json').write_text(json.dumps(rows,indent=2));m.save(a.out/'FINAL.pt');rt.close()
