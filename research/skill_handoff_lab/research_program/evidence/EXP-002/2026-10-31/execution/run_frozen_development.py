"""Predeclared new-scene development comparison. Never select checkpoints on this set."""
import subprocess
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
base=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python']
for method in ['sac','effect_sac']:
 name=f'eval_{method}'
 cmd=base+['research/skill_handoff_lab/tools/program_adaptive_rl.py','--sb3-root','/home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0','--mode','eval','--out',str(r/name),'--method',method,'--seeds','729000,729001,729002,729003','--model',str(r/f'{method}_pilot_1024_retry/MODEL.zip')]
 with (r/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)
for bias in [-1.2,0,1.2]:
 name=f'eval_controls_{bias}'
 cmd=base+['research/skill_handoff_lab/tools/program_adaptive_contact.py','--out',str(r/name),'--seeds','729000,729001,729002,729003','--bias-y-mm',str(bias),'--policies','geometry,search,rls']
 with (r/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)

name='eval_effect_prior_only'
cmd=base+['research/skill_handoff_lab/tools/program_adaptive_rl.py','--sb3-root','/home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0','--mode','eval','--out',str(r/name),'--method','effect_sac','--prior-only','--seeds','729000,729001,729002,729003','--model',str(r/'effect_sac_pilot_1024_retry/MODEL.zip')]
with (r/f'{name}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
print(name,flush=True)
