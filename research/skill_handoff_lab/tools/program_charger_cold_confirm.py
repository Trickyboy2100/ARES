"""Reproduce EXP-001 frozen independent confirmation; never overwrites runs."""
import argparse,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--python',required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
for seed in range(726000,726008):
 for variant in ['frozen_stop','servo_stop','servo_local_hold_stop']:
  name=f'{variant}_{seed}'
  cmd=['bash','research/skill_handoff_lab/tools/handoff_offline.sh',a.python,'research/skill_handoff_lab/tools/program_charger.py','--out',str(a.out/name),'--variant',variant,'--seeds',str(seed),'--single-angle','15','--single-distance','.025']
  with (a.out/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  print(name,flush=True)
