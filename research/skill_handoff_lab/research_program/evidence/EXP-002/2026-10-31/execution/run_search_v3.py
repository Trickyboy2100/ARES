import subprocess
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
for sign in [-1,1]:
 name=f'search_v3_{sign}'
 cmd=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python','research/skill_handoff_lab/tools/program_adaptive_contact.py','--out',str(r/name),'--seeds','727000,727002','--bias-y-mm',str(sign*1.2),'--policies','search,rls']
 with (r/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)
