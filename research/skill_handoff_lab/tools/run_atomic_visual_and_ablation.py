"""Explain the only changed predecessor in frozen confirmation; then render witness."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();t=Path(__file__).parent
for mode,extra in [('front_fixed',['--grasp','1']),('rear_fixed',['--front-only'])]:
 subprocess.run([sys.executable,str(t/'eval_atomic_feedback.py'),'--out',str(a.root/('ablation_'+mode)),'--checkpoint',str(a.root/'macro_seed31702/CHECKPOINT_64.pt'),'--termination','macro','--start','762209','--count','1',*extra],check=True)
subprocess.run([sys.executable,str(t/'render_atomic_witness.py'),'--root',str(a.root)],check=True)
