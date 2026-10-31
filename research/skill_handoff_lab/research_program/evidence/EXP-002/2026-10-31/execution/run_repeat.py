import subprocess
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
b=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python','research/skill_handoff_lab/tools/program_adaptive_contact.py']
for bias in [-1.2,1.2]:
 name=f'repeat_controls_{bias}'
 with (r/f'{name}.log').open('w') as f:subprocess.run(b+['--bias-y-mm',str(bias),'--seeds','729000','--policies','geometry,search','--out',str(r/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)
