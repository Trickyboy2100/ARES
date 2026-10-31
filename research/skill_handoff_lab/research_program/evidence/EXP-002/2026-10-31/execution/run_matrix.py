import subprocess
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
configs=[('bias_plus12_v2',['--bias-y-mm','1.2','--seeds','727000,727002','--policies','geometry,fixed_plus,fixed_minus,search,rls']),('light_lowfriction',['--density','.5','--friction','.2','--seeds','727000,727002','--policies','geometry,search,rls']),('heavy_highfriction',['--density','2','--friction','1','--seeds','727000,727002','--policies','geometry,search,rls'])]
for name,args in configs:
 cmd=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python','research/skill_handoff_lab/tools/program_adaptive_contact.py','--out',str(r/name)]+args
 with (r/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)
