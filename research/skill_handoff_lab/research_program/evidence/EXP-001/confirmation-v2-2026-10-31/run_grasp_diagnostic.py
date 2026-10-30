import subprocess
from pathlib import Path
root=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-001')
for seed in [726005,726000]:
 for angle in [-15,45]:
  name=f'grasp_diagnostic_{seed}_a{angle}'
  cmd=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python','research/skill_handoff_lab/tools/program_charger.py','--out',str(root/name),'--variant','servo_local_hold_stop','--seeds',str(seed),'--single-angle',str(angle),'--single-distance','.025']
  with (root/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  print(name,flush=True)
