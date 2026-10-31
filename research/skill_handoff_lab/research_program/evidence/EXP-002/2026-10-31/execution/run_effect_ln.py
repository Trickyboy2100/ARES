import subprocess
from pathlib import Path
r=Path('/home/andyee/robot-research/ARES-Skilllab/runs/research_program/EXP-002')
b=['bash','research/skill_handoff_lab/tools/handoff_offline.sh','/home/andyee/robot-research/ARES-Skilllab/envs/method-evolution-20261028/bin/python','research/skill_handoff_lab/tools/program_adaptive_rl.py','--sb3-root','/home/andyee/robot-research/ARES-Skilllab/external/sb3-2.7.0','--method','effect_ln_sac','--alpha','.02']
for name,args in [('effect_ln_sac_alpha002',['--mode','train','--steps','1024']),('eval_effect_ln_sac',['--mode','eval','--seeds','729000,729001,729002,729003','--model',str(r/'effect_ln_sac_alpha002/MODEL.zip')]),('eval_effect_ln_prior_only',['--mode','eval','--prior-only','--seeds','729000,729001,729002,729003','--model',str(r/'effect_ln_sac_alpha002/MODEL.zip')])]:
 with (r/f'{name}.log').open('w') as f:subprocess.run(b+args+['--out',str(r/name)],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,flush=True)
