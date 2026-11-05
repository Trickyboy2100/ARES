import argparse,json,subprocess,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();d=json.loads((a.root/'FROZEN_SELECTION.json').read_text());t=Path(__file__).with_name('eval_atomic_feedback.py')
for row in d['models']:
 if row['seed']!=a.seed:continue
 assert hashlib.sha256(Path(row['model']).read_bytes()).hexdigest()==row['sha256']
 subprocess.run([sys.executable,str(t),'--out',str(a.root/('confirm_'+row['run'])),'--checkpoint',row['model'],'--termination',row['termination'],'--start','762200','--count','24'],check=True)
