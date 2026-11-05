"""Frozen references and small independent-task capability run, sequential GPU use."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();t=Path(__file__).parent
for control in ['nominal','rule']:
 subprocess.run([sys.executable,str(t/'eval_atomic_feedback.py'),'--out',str(a.root/f'confirm_{control}'),'--control',control,'--termination','chunk','--start','762200','--count','24'],check=True)
subprocess.run([sys.executable,str(t/'run_atomic_followups.py'),'--root',str(a.root),'--part','charger'],check=True)
