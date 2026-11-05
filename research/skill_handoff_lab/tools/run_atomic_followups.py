"""Sequential native followups, intended as ONE of at most two GPU jobs."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--part',choices=['controls','mechanism','charger'],required=True);a=p.parse_args();t=Path(__file__).parent

def run(script,args):subprocess.run([sys.executable,str(t/script),*map(str,args)],check=True)
if a.part=='controls':
 for control,termination in [('rule','chunk'),('rule','phase'),('rule','insertion_queue'),('budget_guard','chunk')]:
  run('eval_atomic_feedback.py',['--out',a.root/f'control_{control}_{termination}','--control',control,'--termination',termination])
 # Training opportunity uses only predeclared natural warm training scenes.
 run('eval_atomic_feedback.py',['--out',a.root/'training_rule','--control','rule','--termination','phase','--start',761000,'--count',16])
elif a.part=='mechanism':
 run('probe_contact_sequence.py',['--out',a.root/'contact_sequence'])
 run('probe_learned_rescue.py',['--out',a.root/'learned_rescue','--checkpoint',a.root/'macro_bc64/CHECKPOINT_32.pt'])
 run('probe_progress_splice.py',['--out',a.root/'progress_splice'])
 run('probe_replan_clock.py',['--out',a.root/'progress_controls','--seeds',759205,761100,761101,761102,761103])
else:
 run('program_native_charger_pace.py',['--out',a.root/'charger200','--seeds',761600,761601,761602,761603,'--angles',15,30,'--policies','nominal','feedback'])
