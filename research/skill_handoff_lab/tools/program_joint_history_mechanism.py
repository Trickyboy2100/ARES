"""Physical failure localization from original n16 traces, not n2 reruns."""
import json,numpy as np,hashlib
from pathlib import Path
root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/single_paper_20261028/opportunity64');out=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-003/contact_duration_20261102');rows=[]
for path in sorted(root.glob('*_A*_C*.json')):
 d=json.loads(path.read_text());s=d['summary']
 for seed in [420002,420023]:
  if seed not in s['seeds']:continue
  i=s['seeds'].index(seed);states=d['transitions'];close=[t for t in states if t['segments'][i]=='C_CLOSE'];reaches=[t for t in states if t['segments'][i].startswith('C_REACH')];desc=[t for t in states if t['segments'][i].startswith('C_DESCEND')];target=next((p for p in reversed(d['plans']) if p['env']==i and p['label']=='C_DESCEND'),None)
  r=dict(seed=seed,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),success=s['success'][i],error=s['errors'][i],boundaries=s['boundaries'][i])
  if close and reaches and target:
   first=reaches[0];last=close[-1];cube0=np.array(first['state']['actors']['cubeC'][i][:3]);cube1=np.array(last['state']['actors']['cubeC'][i][:3]);tcp=np.array(last['tcp'][i][:3]);r.update(close_step=last['step'],actual_C_motion_since_reach_mm=float(np.linalg.norm(cube1-cube0)*1000),actual_tcp_C_distance_mm=float(np.linalg.norm(tcp-cube1)*1000),target_tracking_mm=float(np.linalg.norm(tcp-target['target_p'])*1000),finger_opening_mm=float(sum(last['q'][i][-2:])*1000))
  rows.append(r)
(out/'HISTORICAL_MECHANISM.json').write_text(json.dumps(rows,indent=2));print(json.dumps([{k:v for k,v in r.items() if k not in ['boundaries','sha256']} for r in rows],indent=2))
