"""Sequential frozen native confirmation inside the existing offline wrapper."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--frame',choices=['measured','remaining_target'],required=True);p.add_argument('--start',type=int,default=760200);p.add_argument('--count',type=int,default=16);p.add_argument('--cohort',default='');a=p.parse_args();tool=Path(__file__).with_name('eval_decoupled_interface.py')
# All four models must be frozen before reading any fresh confirmation outcome.
for seed in [31601,31602]:assert(a.root/f'FROZEN_SELECTION_{seed}.json').exists()
for seed in [31601,31602]:
 m=json.loads((a.root/f'FROZEN_SELECTION_{seed}.json').read_text())['selection'][a.frame]['chosen'];cp=Path(m['path']);assert hashlib.sha256(cp.read_bytes()).hexdigest()==m['sha256']
 subprocess.run([sys.executable,str(tool),'--out',str(a.root/f'confirm_{a.frame}_{seed}{a.cohort}'),'--frame',a.frame,'--checkpoint',str(cp),'--start',str(a.start),'--count',str(a.count),'--split','fresh_confirmation'],check=True)
for control in (['frozen_rule','nominal']if a.frame=='measured' else ['frozen_rule']):
 subprocess.run([sys.executable,str(tool),'--out',str(a.root/f'confirm_{a.frame}_{control}{a.cohort}'),'--frame',a.frame,'--control',control,'--start',str(a.start),'--count',str(a.count),'--split','fresh_confirmation'],check=True)
