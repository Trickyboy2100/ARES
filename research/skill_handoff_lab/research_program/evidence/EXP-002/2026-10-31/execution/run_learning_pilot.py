import subprocess
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
for method in ['sac','effect_sac']:
 name=f'{method}_pilot_1024_retry'
 cmd=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python','research/skill_handoff_lab/tools/program_adaptive_rl.py','--sb3-root','/home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0','--out',str(r/name),'--method',method,'--steps','1024','--alpha','.02']
 with (r/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)
