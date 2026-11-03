"""Reconstruct each saved SAC buffer from actual recorded physics, no training."""
import argparse,pickle,json,hashlib,re
from pathlib import Path
import numpy as np
import program_contact_sac
from program_paired_skill_value import load,load_peg
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tool-root',type=Path,required=True);a=p.parse_args();tool=load(a.tool_root);peg=load_peg(a.root);rows=[]
for folder in sorted(a.root.iterdir()):
 if not folder.is_dir()or not(folder/'TRAINING.json').exists():continue
 logs=json.loads((folder/'TRAINING.json').read_text());is_peg=bool(logs and logs[0].get('task')=='peg');groups=peg if is_peg else tool;scenes=sorted({g['seed']for g in groups})
 for f in sorted(folder.glob('BUFFER_*.pkl')):
  match=re.search(r'f(-?\d+)s(\d+)',f.name);fold=int(match[1]);g=[z for z in groups if scenes.index(z['seed'])%3!=fold];expected=[]
  for z in g:
   for _,_,ts in z['items']:
    for i,t in enumerate(ts):
     u=list(t[2]);u[0]=2*u[0]-1 if is_peg and i==1 else u[0];expected.append((t[0],t[1],u,t[3],t[4]))
  b=pickle.loads(f.read_bytes());n=b.size();assert n==len(expected);e={}
  for name,col in [('observations',0),('next_observations',1),('actions',2),('rewards',3),('dones',4)]:
   actual=getattr(b,name)[:n];wanted=np.array([t[col]for t in expected],dtype=np.float32);actual=actual.reshape(wanted.shape);err=float(np.max(np.abs(actual-wanted)));assert err<1e-6,(f,name,err);e[name]=err
  rows.append(dict(file=str(f.relative_to(a.root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),transitions=n,max_errors=e))
(a.root/'BUFFER_AUDIT.json').write_text(json.dumps(dict(status='PASS',buffers=len(rows),rows=rows,scope='Stored SAC transitions independently reconstructed from source physics; no new simulation'),indent=2));print('PASS',len(rows),'buffers')
