import subprocess
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
b=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python','research/skill_handoff_lab/tools/program_adaptive_rl.py','--sb3-root','/home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0']
for name,args in [('ln_sac_alpha002',['--mode','train','--method','ln_sac','--steps','1024','--alpha','.02']),('eval_ln_sac_alpha002',['--mode','eval','--method','ln_sac','--alpha','.02','--seeds','729000,729001,729002,729003','--model',str(r/'ln_sac_alpha002/MODEL.zip')])]:
 with (r/f'{name}.log').open('w') as f:subprocess.run(b+args+['--out',str(r/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)
